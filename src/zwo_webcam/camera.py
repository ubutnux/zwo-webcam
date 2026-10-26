"""ZWO camera access through pyzwoasi, plus a simulated camera for testing without hardware."""

from __future__ import annotations

import logging
import math
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ASI_IMG_TYPE enum values (same as pyzwoasi.ASIImageType)
IMG_RAW8, IMG_RGB24, IMG_RAW16, IMG_Y8 = 0, 1, 2, 3
IMG_NAMES = {IMG_RAW8: "RAW8", IMG_RGB24: "RGB24", IMG_RAW16: "RAW16", IMG_Y8: "Y8"}
BAYER_NAMES = {0: "RG", 1: "BG", 2: "GR", 3: "GB"}


def _bayer_code(pattern: int) -> int:
    """cv2 code to debayer an ASI pattern (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG) to BGR.

    OpenCV's legacy constant names refer to the second row, so RGGB = COLOR_BayerBG2BGR.
    """
    names = {
        0: ("COLOR_BayerRGGB2BGR", "COLOR_BayerBG2BGR"),
        1: ("COLOR_BayerBGGR2BGR", "COLOR_BayerRG2BGR"),
        2: ("COLOR_BayerGRBG2BGR", "COLOR_BayerGB2BGR"),
        3: ("COLOR_BayerGBRG2BGR", "COLOR_BayerGR2BGR"),
    }[pattern]
    for n in names:
        if hasattr(cv2, n):
            return getattr(cv2, n)
    raise RuntimeError("This OpenCV build cannot debayer")


def _video_formats(arr) -> list[int]:
    """The SDK format list ends at ASI_IMG_END (-1); anything after it is padding."""
    out = []
    for f in arr:
        if f not in IMG_NAMES:
            break
        if f not in out:
            out.append(f)
    return out


class FrameTimeout(Exception):
    """No frame arrived within the short wait; not an error."""


class ZwoCamera:
    """A ZWO ASI camera in video (continuous capture) mode."""

    def __init__(self, index: int = 0):
        import pyzwoasi as asi  # imported here so simulation mode works without the DLL/driver

        self.asi = asi
        n = asi.getNumOfConnectedCameras()
        if n == 0:
            raise RuntimeError("No ZWO camera detected")
        if index >= n:
            raise RuntimeError(f"Camera index {index} not found ({n} camera(s) detected)")
        info = asi.getCameraProperty(index)
        self.id = info.CameraID
        self.info = {
            "name": info.Name.decode(errors="replace"),
            "max_width": int(info.MaxWidth),
            "max_height": int(info.MaxHeight),
            "is_color": bool(info.IsColorCam),
            "bayer": int(info.BayerPattern),
            "bins": [b for b in info.SupportedBins if b != 0],
            "formats": _video_formats(info.SupportedVideoFormat),
            "pixel_size": round(float(info.PixelSize), 2),
            "bit_depth": int(info.BitDepth),
            "usb3": bool(info.IsUSB3Camera),
            "usb3_host": bool(info.IsUSB3Host),
            "cooler": bool(info.IsCoolerCam),
        }
        asi.openCamera(self.id)
        try:
            self._setup()
        except Exception:
            asi.closeCamera(self.id)  # so the next attempt starts from a clean state
            raise

    def _setup(self):
        asi = self.asi
        asi.initCamera(self.id)
        try:
            self.info["serial"] = asi.getSerialNumber(self.id)
        except Exception:
            pass

        self.controls: dict[str, dict] = {}
        for i in range(asi.getNumOfControls(self.id)):
            c = asi.getControlCaps(self.id, i)
            name = c.Name.decode(errors="replace")
            self.controls[name] = {
                "name": name,
                "description": c.Description.decode(errors="replace"),
                "min": int(c.MinValue),
                "max": int(c.MaxValue),
                "default": int(c.DefaultValue),
                "auto_supported": bool(c.IsAutoSupported),
                "writable": bool(c.IsWritable),
                "type": int(c.ControlType),
            }

        self.capturing = False
        self.binning = 1
        self.img_type = IMG_RGB24 if self.info["is_color"] and IMG_RGB24 in self.info["formats"] else IMG_RAW8
        if not self.info["is_color"]:
            self.img_type = IMG_RAW8
        # Sensible starting values for webcam use
        if "BandWidth" in self.controls:
            self._set("BandWidth", self.controls["BandWidth"]["default"], False)
        if "Exposure" in self.controls:
            self._set("Exposure", 20_000, False)
        self.set_format(self.binning, self.img_type)

    # --- controls ---
    def _set(self, name: str, value: int, auto: bool):
        c = self.controls[name]
        value = max(c["min"], min(c["max"], int(value)))
        self.asi.setControlValue(self.id, c["type"], value, bool(auto) and c["auto_supported"])

    def set_control(self, name: str, value: int, auto: bool = False):
        if name not in self.controls:
            raise KeyError(f"Unknown control: {name}")
        if not self.controls[name]["writable"]:
            raise ValueError(f"{name} is read-only")
        self._set(name, value, auto)

    def get_control_values(self) -> dict[str, tuple[int, bool]]:
        out = {}
        for name, c in self.controls.items():
            try:
                v, a = self.asi.getControlValue(self.id, c["type"])
                out[name] = (int(v), bool(a))
            except Exception:
                pass
        return out

    def exposure_us(self) -> int:
        if "Exposure" not in self.controls:
            return 0
        return int(self.asi.getControlValue(self.id, self.controls["Exposure"]["type"])[0])

    # --- format / ROI ---
    def set_format(self, binning: int, img_type: int):
        if binning not in self.info["bins"]:
            raise ValueError(f"Binning {binning} is not supported")
        if img_type not in self.info["formats"]:
            raise ValueError(f"Format {img_type} is not supported")
        was = self.capturing
        if was:
            self.stop()
        w = (self.info["max_width"] // binning) // 8 * 8
        h = (self.info["max_height"] // binning) // 2 * 2
        self.asi.setROIFormat(self.id, w, h, binning, img_type)
        self.width, self.height, self.binning, self.img_type = w, h, binning, img_type
        if was:
            self.start()

    @property
    def buffer_size(self) -> int:
        bpp = {IMG_RAW8: 1, IMG_Y8: 1, IMG_RAW16: 2, IMG_RGB24: 3}[self.img_type]
        return self.width * self.height * bpp

    # --- capture ---
    def start(self):
        if not self.capturing:
            self.asi.startVideoCapture(self.id)
            self.capturing = True

    def stop(self):
        if self.capturing:
            self.asi.stopVideoCapture(self.id)
            self.capturing = False

    def read_raw(self, wait_ms: int) -> np.ndarray:
        """Fetch one raw (not yet debayered) frame. Raises FrameTimeout if none is ready."""
        try:
            data = self.asi.getVideoData(self.id, self.buffer_size, wait_ms)
        except self.asi.ASIError as e:
            if len(e.args) > 1 and e.args[1] == self.asi.ASIErrorCode.ASI_ERROR_TIMEOUT:
                raise FrameTimeout() from None
            raise
        h, w = self.height, self.width
        if self.img_type == IMG_RGB24:
            return np.frombuffer(data, np.uint8).reshape(h, w, 3)  # the SDK delivers BGR order
        if self.img_type == IMG_RAW16:
            return np.frombuffer(data, np.uint16).reshape(h, w)
        return np.frombuffer(data, np.uint8).reshape(h, w)

    def dropped_frames(self) -> int:
        try:
            return int(self.asi.getDroppedFrames(self.id))
        except Exception:
            return 0

    def close(self):
        try:
            self.stop()
        finally:
            self.asi.closeCamera(self.id)


class SimCamera:
    """Fake camera with the same interface as ZwoCamera, for testing without hardware."""

    def __init__(self, index: int = 0):
        self.id = index
        self.info = {
            "name": "Simulated ASI (no hardware)",
            "max_width": 1936,
            "max_height": 1096,
            "is_color": True,
            "bayer": 0,
            "bins": [1, 2, 4],
            "formats": [IMG_RAW8, IMG_RGB24, IMG_RAW16, IMG_Y8],
            "pixel_size": 2.9,
            "bit_depth": 12,
            "usb3": True,
            "usb3_host": True,
            "cooler": False,
            "serial": "SIM0000",
        }

        def ctl(name, desc, lo, hi, default, auto=False, writable=True):
            return {"name": name, "description": desc, "min": lo, "max": hi, "default": default,
                    "auto_supported": auto, "writable": writable, "type": 0}

        self.controls = {c["name"]: c for c in [
            ctl("Gain", "Gain", 0, 570, 100, True),
            ctl("Exposure", "Exposure Time(us)", 32, 2_000_000_000, 20_000, True),
            ctl("Offset", "offset", 0, 80, 8),
            ctl("BandWidth", "The total data transfer rate percentage", 40, 100, 50, True),
            ctl("Flip", "Flip: 0->None 1->Horiz 2->Vert 3->Both", 0, 3, 0),
            ctl("WB_R", "White balance: Red component", 1, 99, 52, True),
            ctl("WB_B", "White balance: Blue component", 1, 99, 95, True),
            ctl("Temperature", "Sensor temperature(degrees Celsius)", -500, 1000, 20, writable=False),
            ctl("HighSpeedMode", "Is high speed mode:0->No 1->Yes", 0, 1, 0),
        ]}
        self.values = {n: [c["default"], False] for n, c in self.controls.items()}
        self.capturing = False
        self._next = 0.0
        self._t0 = time.monotonic()
        self.set_format(1, IMG_RGB24)

    def set_control(self, name, value, auto=False):
        c = self.controls[name]
        if not c["writable"]:
            raise ValueError(f"{name} is read-only")
        self.values[name] = [max(c["min"], min(c["max"], int(value))), bool(auto) and c["auto_supported"]]

    def get_control_values(self):
        self.values["Temperature"][0] = 200 + int(10 * math.sin(time.monotonic() / 30))
        return {n: (v[0], v[1]) for n, v in self.values.items()}

    def exposure_us(self):
        return self.values["Exposure"][0]

    def set_format(self, binning, img_type):
        self.width = (self.info["max_width"] // binning) // 8 * 8
        self.height = (self.info["max_height"] // binning) // 2 * 2
        self.binning, self.img_type = binning, img_type
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self._bg = np.dstack([
            (xx * 255 // self.width), (yy * 255 // self.height), np.full_like(xx, 96)
        ]).astype(np.float32)

    def start(self):
        self.capturing = True

    def stop(self):
        self.capturing = False

    def read_raw(self, wait_ms):
        # Honour the exposure time (capped at ~30 fps) like a real camera
        period = max(self.exposure_us() / 1e6, 1 / 30)
        now = time.monotonic()
        if self._next == 0.0 or self._next < now - period:
            self._next = now
        wait = self._next + period - now
        if wait > wait_ms / 1000:
            time.sleep(wait_ms / 1000)
            raise FrameTimeout()
        time.sleep(max(0.0, wait))
        self._next += period

        t = time.monotonic() - self._t0
        gain = 10 ** (self.values["Gain"][0] / 200)  # 0.1 dB per unit, like ZWO
        bright = min(4.0, self.exposure_us() / 20_000 * gain / 3)
        img = self._bg * bright + np.random.normal(0, 4 * gain, self._bg.shape[:2])[..., None]
        bgr = np.clip(img, 0, 255).astype(np.uint8)
        cx = int((0.5 + 0.35 * math.sin(t)) * self.width)
        cy = int((0.5 + 0.35 * math.cos(t * 0.7)) * self.height)
        cv2.circle(bgr, (cx, cy), self.height // 10, (255, 255, 255), -1)
        cv2.putText(bgr, time.strftime("%H:%M:%S"), (20, self.height // 8), cv2.FONT_HERSHEY_SIMPLEX,
                    self.height / 300, (255, 255, 255), max(1, self.height // 200))
        flip = self.values["Flip"][0]
        if flip:
            bgr = cv2.flip(bgr, {1: 1, 2: 0, 3: -1}[flip])
        if self.img_type == IMG_RGB24:
            return bgr
        if self.img_type == IMG_Y8:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # RGGB mosaic
        raw = np.empty(bgr.shape[:2], np.uint8)
        raw[0::2, 0::2] = bgr[0::2, 0::2, 2]
        raw[0::2, 1::2] = bgr[0::2, 1::2, 1]
        raw[1::2, 0::2] = bgr[1::2, 0::2, 1]
        raw[1::2, 1::2] = bgr[1::2, 1::2, 0]
        if self.img_type == IMG_RAW16:
            return raw.astype(np.uint16) << 8
        return raw

    def dropped_frames(self):
        return 0

    def close(self):
        self.capturing = False


def to_bgr8(raw: np.ndarray, img_type: int, bayer: int | None) -> np.ndarray:
    """Turn a raw frame into an 8-bit image (BGR or grayscale) ready to encode.

    `bayer` None means show it as-is without debayering (mono).
    """
    if img_type == IMG_RGB24:
        return raw
    if img_type == IMG_RAW16:
        raw = (raw >> 8).astype(np.uint8)  # ZWO RAW16 data is MSB-aligned
    if img_type == IMG_Y8 or bayer is None:
        return raw
    return cv2.cvtColor(raw, _bayer_code(bayer))


def auto_stretch(img: np.ndarray) -> np.ndarray:
    """Stretch contrast between the 0.5% and 99.8% percentiles (useful for faint/astro targets)."""
    small = img[::8, ::8]
    lo, hi = np.percentile(small, (0.5, 99.8))
    if hi - lo < 2:
        return img
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


class CaptureWorker(threading.Thread):
    """Thread that keeps pulling frames from the camera and keeps the latest one.

    All camera access goes through `self.lock`, so web commands (control or format
    changes) slot in between the deliberately short getVideoData polls.
    """

    POLL_MS = 200

    def __init__(self, factory, on_frame):
        super().__init__(daemon=True, name="capture")
        self.factory = factory
        self.on_frame = on_frame  # called as (frame_bgr8, seq, unix_time) from this thread
        self.lock = threading.RLock()
        self.cam = None
        self.error: str | None = None
        self.seq = 0
        self.fps = 0.0
        self.running = True
        self.stretch = False
        self.debayer = True
        self.bayer_override: int | None = None
        self._win_start = time.monotonic()
        self._win_count = 0
        self._waiters = 0

    # called from web threads
    def call(self, fn):
        self._waiters += 1  # ask the capture loop to yield
        with self.lock:
            self._waiters -= 1
            if self.cam is None:
                raise RuntimeError(self.error or "Camera not ready")
            return fn(self.cam)

    def stop(self):
        self.running = False

    def _connect(self):
        while self.running:
            try:
                cam = self.factory()
                cam.start()
                with self.lock:
                    self.cam = cam
                    self.error = None
                log.info("Camera ready: %s %dx%d", cam.info["name"], cam.width, cam.height)
                return
            except Exception as e:
                self.error = f"Could not open the camera: {e}"
                log.warning("%s; retrying in 3 s", self.error)
                time.sleep(3)

    def run(self):
        self._connect()
        waited = stalls = 0
        while self.running:
            while self._waiters > 0:
                time.sleep(0.002)
            try:
                with self.lock:
                    cam = self.cam
                    raw = cam.read_raw(self.POLL_MS)
                    captured_at = time.time()
                    img_type = cam.img_type
                    bayer = None
                    if cam.info["is_color"] and self.debayer:
                        bayer = cam.info["bayer"] if self.bayer_override is None else self.bayer_override
                waited = stalls = 0
            except FrameTimeout:
                # No frame for well past the exposure time: treat the camera as stuck
                waited += self.POLL_MS
                if time.monotonic() - self._win_start >= 2.0:  # long exposures: let fps drop
                    self.fps = self._win_count / (time.monotonic() - self._win_start)
                    self._win_start, self._win_count = time.monotonic(), 0
                try:
                    with self.lock:
                        limit = self.cam.exposure_us() / 1000 * 2 + 5000
                    if waited > limit:
                        waited = 0
                        stalls += 1
                        if stalls > 1:
                            # Restarting capture did not help: close and reopen the camera
                            raise RuntimeError("camera stopped sending frames")
                        log.warning("No frame for %d ms, restarting capture", limit)
                        with self.lock:
                            self.cam.stop()
                            self.cam.start()
                except Exception as e:
                    stalls = 0
                    self._lost(e)
                continue
            except Exception as e:
                self._lost(e)
                continue

            img = to_bgr8(raw, img_type, bayer)
            if self.stretch:
                img = auto_stretch(img)
            # fps = frames per ~1 s window (per-frame 1/dt is too noisy)
            now = time.monotonic()
            self._win_count += 1
            if now - self._win_start >= 1.0:
                self.fps = self._win_count / (now - self._win_start)
                self._win_start, self._win_count = now, 0
            self.seq += 1
            self.on_frame(img, self.seq, captured_at)

        with self.lock:
            if self.cam:
                self.cam.close()
                self.cam = None

    def _lost(self, e: Exception):
        log.error("Camera error: %s", e)
        with self.lock:
            try:
                if self.cam:
                    self.cam.close()
            except Exception:
                pass
            self.cam = None
            self.error = f"Camera disconnected: {e}"
        self.fps = 0.0
        time.sleep(1)
        self._connect()
