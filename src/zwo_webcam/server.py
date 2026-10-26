"""Web server: MJPEG stream, snapshots, and the camera control API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

from .camera import IMG_NAMES, CaptureWorker, SimCamera, ZwoCamera

log = logging.getLogger("zwo_webcam")
STATIC = Path(__file__).parent / "static"
BOUNDARY = "zwoframe"
STAMP_MODES = ("off", "local", "utc")
STAMP_POSITIONS = ("tl", "tr", "bl", "br")


class Hub:
    """Holds the latest frame and encodes JPEGs on demand.

    Encoding happens only when someone asks, and the result is shared between
    clients that use the same quality and width.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.frame: np.ndarray | None = None
        self.frame_time = 0.0
        self.seq = 0
        self.stamp_mode = "off"  # off | local | utc
        self.stamp_pos = "bl"  # tl | tr | bl | br
        self._event = asyncio.Event()
        self._cache: OrderedDict[tuple, asyncio.Future] = OrderedDict()
        self.bytes_out = 0
        self.kbps = 0.0
        self.clients: dict[int, dict] = {}

    # from the capture thread
    def push(self, img: np.ndarray, seq: int, t: float):
        self.loop.call_soon_threadsafe(self._set, img, seq, t)

    def _set(self, img, seq, t):
        self.frame, self.seq, self.frame_time = img, seq, t
        self._event.set()
        self._event = asyncio.Event()

    async def wait_newer(self, seq: int, timeout: float) -> bool:
        if self.seq > seq:
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def jpeg(self, quality: int, max_width: int) -> tuple[bytes, int]:
        seq, img = self.seq, self.frame
        if img is None:
            raise web.HTTPServiceUnavailable(text="No frame from the camera yet")
        stamp = (self.stamp_mode, self.stamp_pos, self.frame_time)
        key = (seq, quality, max_width, stamp[:2])
        fut = self._cache.get(key)
        if fut is None:
            fut = asyncio.ensure_future(asyncio.to_thread(encode_jpeg, img, quality, max_width, stamp))
            self._cache[key] = fut
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
        return await fut, seq

    async def meter(self):
        last, t0 = 0, time.monotonic()
        while True:
            await asyncio.sleep(1)
            t = time.monotonic()
            self.kbps = 0.7 * self.kbps + 0.3 * (self.bytes_out - last) * 8 / 1000 / (t - t0)
            last, t0 = self.bytes_out, t


def format_stamp(t: float, mode: str) -> str:
    if mode == "utc":
        dt = datetime.fromtimestamp(t, timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d} UTC"
    dt = datetime.fromtimestamp(t).astimezone()
    off = dt.strftime("%z")  # +0700 -> UTC+07:00
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d} UTC{off[:3]}:{off[3:]}"


def draw_stamp(img: np.ndarray, text: str, pos: str) -> np.ndarray:
    """Draw the text on a dark plate in a corner, sized to the output width."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, w / 1600)
    thick = max(1, round(scale * 1.6))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = max(4, round(th * 0.45))
    margin = max(6, round(w / 120))
    x = margin if pos in ("tl", "bl") else w - tw - 2 * pad - margin
    y = margin if pos in ("tl", "tr") else h - th - base - 2 * pad - margin
    x, y = max(0, x), max(0, y)
    plate = img[y:y + th + base + 2 * pad, x:x + tw + 2 * pad]
    plate[:] = (plate * 0.35).astype(img.dtype)  # translucent dark plate
    white = 255 if img.ndim == 2 else (255, 255, 255)
    cv2.putText(img, text, (x + pad, y + pad + th), font, scale, white, thick, cv2.LINE_AA)
    return img


def encode_jpeg(img: np.ndarray, quality: int, max_width: int, stamp=("off", "bl", 0.0)) -> bytes:
    mode, pos, t = stamp
    h, w = img.shape[:2]
    if max_width and w > max_width:
        nh = max(2, round(h * max_width / w))
        img = cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)
    elif mode != "off":
        img = img.copy()  # the shared frame may be read-only and is reused by other encodes
    if mode != "off" and t:
        img = draw_stamp(img, format_stamp(t, mode), pos)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def stream_params(req: web.Request, defaults: dict) -> tuple[float, int, int]:
    q = req.query

    def num(name, cast, lo, hi):
        try:
            return max(lo, min(hi, cast(q.get(name, defaults[name]))))
        except ValueError:
            raise web.HTTPBadRequest(text=f"Invalid {name} parameter")

    return num("fps", float, 0.1, 60), num("q", int, 10, 95), num("w", int, 0, 10000)


# ---------- handlers ----------

async def index(req):
    return web.FileResponse(STATIC / "index.html")


async def stream(req: web.Request):
    hub: Hub = req.app["hub"]
    fps, quality, width = stream_params(req, req.app["defaults"])
    resp = web.StreamResponse(headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(req)
    cid = id(resp)
    hub.clients[cid] = {"peer": req.remote, "fps": fps, "q": quality, "w": width, "since": time.time()}
    sent_seq, next_t = 0, 0.0
    try:
        while True:
            # Send only new frames; without one, resend the last frame every 5 s as a keepalive.
            await hub.wait_newer(sent_seq, 5.0)
            if hub.frame is None:
                continue
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            data, seq = await hub.jpeg(quality, width)
            part = (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                    ).encode() + data + b"\r\n"
            # write() waits for the socket to drain, so slow clients skip frames instead of building up lag
            await resp.write(part)
            hub.bytes_out += len(part)
            sent_seq = seq
            next_t = max(time.monotonic(), next_t + 1 / fps)
    except (ConnectionError, RuntimeError):
        pass
    finally:
        hub.clients.pop(cid, None)
    return resp


async def snapshot(req: web.Request):
    """A single JPEG. `after=<seq>` waits for a newer frame (for frame-by-frame mode)."""
    hub: Hub = req.app["hub"]
    _, quality, width = stream_params(req, {**req.app["defaults"], "q": 90, "w": 0})
    try:
        after = int(req.query.get("after", 0))
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid after parameter")
    await hub.wait_newer(after, 10.0)
    data, seq = await hub.jpeg(quality, width)
    hub.bytes_out += len(data)
    headers = {"Cache-Control": "no-store", "X-Frame-Seq": str(seq)}
    if "download" in req.query:
        headers["Content-Disposition"] = f'attachment; filename="zwo_{time.strftime("%Y%m%d_%H%M%S")}.jpg"'
    return web.Response(body=data, content_type="image/jpeg", headers=headers)


async def status(req: web.Request):
    worker: CaptureWorker = req.app["worker"]
    hub: Hub = req.app["hub"]

    def read(cam):
        values = cam.get_control_values()
        controls = []
        for name, c in cam.controls.items():
            v = values.get(name)
            controls.append({**c, "value": v[0] if v else None, "auto": v[1] if v else False})
        return {
            "camera": cam.info,
            "format": {"width": cam.width, "height": cam.height, "bin": cam.binning,
                       "img_type": cam.img_type, "img_name": IMG_NAMES[cam.img_type]},
            "formats": [{"id": f, "name": IMG_NAMES[f]} for f in cam.info["formats"]],
            "controls": controls,
            "dropped": cam.dropped_frames(),
        }

    try:
        data = await asyncio.to_thread(worker.call, read)
    except RuntimeError as e:
        data = {"camera": None}
        worker.error = worker.error or str(e)
    data["stats"] = {
        "fps": round(worker.fps, 2),
        "seq": hub.seq,
        "kbps": round(hub.kbps, 1),
        "clients": list(hub.clients.values()),
        "error": worker.error,
    }
    data["process"] = {"stretch": worker.stretch, "debayer": worker.debayer, "bayer": worker.bayer_override,
                       "timestamp": hub.stamp_mode, "timestamp_pos": hub.stamp_pos}
    data["defaults"] = req.app["defaults"]
    return web.json_response(data)


async def set_control(req: web.Request):
    body = await req.json()
    worker: CaptureWorker = req.app["worker"]
    try:
        await asyncio.to_thread(worker.call, lambda cam: cam.set_control(
            str(body["name"]), int(body["value"]), bool(body.get("auto", False))))
    except (KeyError, ValueError) as e:
        raise web.HTTPBadRequest(text=str(e))
    except Exception as e:
        raise web.HTTPInternalServerError(text=str(e))
    return web.json_response({"ok": True})


async def set_format(req: web.Request):
    body = await req.json()
    worker: CaptureWorker = req.app["worker"]
    try:
        await asyncio.to_thread(worker.call, lambda cam: cam.set_format(int(body["bin"]), int(body["img_type"])))
    except (KeyError, ValueError) as e:
        raise web.HTTPBadRequest(text=str(e))
    except Exception as e:
        raise web.HTTPInternalServerError(text=str(e))
    return web.json_response({"ok": True})


async def set_process(req: web.Request):
    body = await req.json()
    hub: Hub = req.app["hub"]
    if "timestamp" in body:
        if body["timestamp"] not in STAMP_MODES:
            raise web.HTTPBadRequest(text=f"timestamp must be one of {', '.join(STAMP_MODES)}")
        hub.stamp_mode = body["timestamp"]
    if "timestamp_pos" in body:
        if body["timestamp_pos"] not in STAMP_POSITIONS:
            raise web.HTTPBadRequest(text=f"timestamp_pos must be one of {', '.join(STAMP_POSITIONS)}")
        hub.stamp_pos = body["timestamp_pos"]
    worker: CaptureWorker = req.app["worker"]
    if "stretch" in body:
        worker.stretch = bool(body["stretch"])
    if "debayer" in body:
        worker.debayer = bool(body["debayer"])
    if "bayer" in body:
        b = body["bayer"]
        worker.bayer_override = None if b in (None, "", "auto") else int(b) % 4
    return web.json_response({"ok": True})


def basic_auth(user: str, password: str):
    expected = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    @web.middleware
    async def mw(req, handler):
        if req.headers.get("Authorization") != expected:
            raise web.HTTPUnauthorized(headers={"WWW-Authenticate": 'Basic realm="ZWO Webcam"'})
        return await handler(req)

    return mw


def build_app(args) -> web.Application:
    middlewares = []
    if args.auth:
        user, _, pw = args.auth.partition(":")
        middlewares.append(basic_auth(user, pw))
    app = web.Application(middlewares=middlewares)
    app["defaults"] = {"fps": args.fps, "q": args.quality, "w": args.width}

    async def on_startup(app):
        hub = Hub(asyncio.get_running_loop())
        factory = (lambda: SimCamera(args.camera)) if args.sim else (lambda: ZwoCamera(args.camera))
        worker = CaptureWorker(factory, hub.push)
        worker.stretch = args.stretch
        hub.stamp_mode, hub.stamp_pos = args.timestamp, args.timestamp_pos
        worker.start()
        app["hub"], app["worker"] = hub, worker
        app["meter"] = asyncio.create_task(hub.meter())

    async def on_cleanup(app):
        app["meter"].cancel()
        app["worker"].stop()
        await asyncio.to_thread(app["worker"].join, 3)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/stream.mjpg", stream)
    app.router.add_get("/snapshot.jpg", snapshot)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/control", set_control)
    app.router.add_post("/api/format", set_format)
    app.router.add_post("/api/process", set_process)
    app.router.add_static("/static", STATIC)
    return app


def main():
    p = argparse.ArgumentParser(description="Turn a ZWO ASI camera into a webcam you can watch and control from a browser")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0 = all networks)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--camera", type=int, default=0, help="camera index when more than one is connected")
    p.add_argument("--sim", action="store_true", help="use a simulated camera (no hardware)")
    p.add_argument("--fps", type=float, default=10, help="default stream frame rate")
    p.add_argument("--quality", type=int, default=70, help="default JPEG quality (10-95)")
    p.add_argument("--width", type=int, default=1280, help="default max stream width, 0 = native resolution")
    p.add_argument("--stretch", action="store_true", help="start with contrast auto-stretch enabled")
    p.add_argument("--timestamp", choices=STAMP_MODES, default="local",
                   help="burn the capture time into every frame (default: local)")
    p.add_argument("--timestamp-pos", choices=STAMP_POSITIONS, default="bl",
                   help="timestamp corner: tl, tr, bl or br (default: bl)")
    p.add_argument("--auth", metavar="USER:PASS", help="require HTTP Basic Auth")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    print(f"ZWO Webcam: open http://localhost:{args.port}/  (stream: /stream.mjpg)")
    web.run_app(build_app(args), host=args.host, port=args.port, print=None)
