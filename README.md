# ZWO Webcam

Turns a ZWO ASI camera into a webcam you can watch and control from any browser.
Camera access uses [pyzwoasi](https://github.com/fmargall/pyzwoasi); the web server uses aiohttp.

## Running

You need [uv](https://docs.astral.sh/uv/) and the ZWO USB camera driver (on Windows, download it from zwoastro.com).
pyzwoasi ships the `ASICamera2` SDK library.

```bash
uv run zwo-webcam
```

Open `http://localhost:8080/`, or `http://<computer-ip>:8080/` from another device on the network.

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8080 | HTTP port |
| `--host` | 0.0.0.0 | bind address; `127.0.0.1` = this computer only |
| `--camera` | 0 | camera index when several are connected |
| `--fps` / `--quality` / `--width` | 10 / 70 / 1280 | default stream settings for `/stream.mjpg` |
| `--stretch` | off | start with contrast auto-stretch on (astro / faint targets) |
| `--timestamp` | local | burn the capture time into every frame: `off`, `local` or `utc` |
| `--timestamp-pos` | bl | timestamp corner: `tl`, `tr`, `bl`, `br` |
| `--auth USER:PASS` | – | require HTTP Basic Auth (use this when exposed to the internet) |
| `--sim` | – | simulated camera, for trying it without hardware |

## The web page

- Live view with a framing crosshair, full-resolution snapshot and full screen (keys `C`, `S`, `F`).
- Quality presets (Data saver, Balanced, Sharp, Full or Custom) and two delivery modes:
  **Live stream** (MJPEG) or **Frame by frame**, which fetches one image at a time and holds up best on slow or mobile links.
- A live histogram with clipping warning and exposure presets.
- A capture-time timestamp (local time or UTC, millisecond precision) burned into the stream, snapshots and the shared link, in any corner.
- Every control the camera reports, grouped by purpose: exposure, gain, offset, white balance, flip, auto-exposure limits, USB bandwidth, cooling.
- Night mode (`N`) turns the whole interface dim red to protect dark-adapted eyes.

## Endpoints

- `/stream.mjpg?fps=10&q=70&w=1280`: MJPEG stream; opens directly in VLC, OBS (Media Source / Browser), Home Assistant, etc.
- `/snapshot.jpg?q=90&w=0`: one JPEG (`w=0` = native resolution, add `&download=1` to download)
- `/api/status`, `POST /api/control` `{name, value, auto}`, `POST /api/format` `{bin, img_type}`, `POST /api/process` `{stretch, debayer, bayer, timestamp, timestamp_pos}`

## Bandwidth

- Frames are JPEG-encoded only when someone is watching, and one encode is shared by all viewers with the same settings.
- Each viewer picks their own frame rate, JPEG quality and maximum width.
- The server waits for the socket to drain before sending the next frame, so slow viewers skip frames instead of falling behind.

## Notes

- Colour cameras default to `RGB24`, debayered and white-balanced by the SDK. `RAW8`/`RAW16` are debayered by OpenCV on the server; if red and blue look swapped, change the Bayer pattern.
- If the camera is unplugged, the server tries to reconnect every 3 seconds.
