from __future__ import annotations

import os
import time
import threading
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# --- env loader ---
def _load_env_vars() -> None:
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            print(f"[PREVIEW] Loaded environment from: {env_path}")
        else:
            print(f"[PREVIEW] WARNING: app.env not found at {env_path}, using system env vars")
    except ImportError:
        print("[PREVIEW] WARNING: python-dotenv not installed, using system env vars only")

_load_env_vars()

# --- settings ---
PLATE_CAMERA_RTSP = os.getenv("PLATE_CAMERA_RTSP", "")
SNOW_CAMERA_RTSP = os.getenv("SNOW_CAMERA_RTSP", "")

PLATE_LINE_Y_POSITION = float(os.getenv("PLATE_LINE_Y_POSITION", "0.65"))
SNOW_LINE_Y_POSITION = float(os.getenv("SNOW_LINE_Y_POSITION", "0.82"))

PLATE_LINE_DIRECTION = os.getenv("PLATE_LINE_DIRECTION", "down").strip().lower()
SNOW_LINE_DIRECTION = os.getenv("SNOW_LINE_DIRECTION", "down").strip().lower()

FFMPEG_OUT_W = int(os.getenv("FFMPEG_OUT_W", "1280"))
FFMPEG_OUT_H = int(os.getenv("FFMPEG_OUT_H", "720"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
MIN_CONFIDENCE = float(os.getenv("STREAM_MIN_CONFIDENCE", "0.50"))
MIN_BBOX_AREA = int(os.getenv("STREAM_MIN_BBOX_AREA", "10000"))
DETECTION_INTERVAL = int(os.getenv("STREAM_DETECTION_INTERVAL", "3"))

FFMPEG_BIN_ENV = os.getenv("FFMPEG_BIN", "").strip()

def _resolve_ffmpeg_bin() -> Optional[str]:
    if FFMPEG_BIN_ENV and os.path.exists(FFMPEG_BIN_ENV):
        return FFMPEG_BIN_ENV
    p = shutil.which("ffmpeg")
    return p

# --- ffmpeg reader ---
class FFmpegRTSPReader:
    def __init__(self, rtsp_url: str, name: str, width: int, height: int):
        self.rtsp_url = rtsp_url
        self.name = name
        self.width = width
        self.height = height
        self.frame_size = self.width * self.height * 3

        self._ffmpeg_bin = _resolve_ffmpeg_bin()
        self.process: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._last_frame: Optional[np.ndarray] = None

    def start(self) -> bool:
        if not self._ffmpeg_bin:
            print("[PREVIEW] ERROR: ffmpeg not found. Set FFMPEG_BIN in app.env (where.exe ffmpeg).")
            return False

        vf = f"scale={self.width}:{self.height}"
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
            "-rtsp_transport", "tcp",
            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-err_detect", "ignore_err",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", self.rtsp_url,
            "-an",
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]

        creationflags = 0
        if os.name == "nt":
            try:
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            except Exception:
                creationflags = 0

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 4,
            creationflags=creationflags,
        )
        if self.process.stdout is None:
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ffmpeg-preview-{self.name}")
        self._thread.start()
        print(f"[PREVIEW] ffmpeg started for {self.name}: {self.width}x{self.height}")
        return True

    def _loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        stdout = self.process.stdout
        need = self.frame_size

        while not self._stop.is_set():
            if self.process.poll() is not None:
                break

            buf = bytearray(need)
            mv = memoryview(buf)
            got = 0
            try:
                while got < need and not self._stop.is_set():
                    chunk = stdout.read(need - got)
                    if not chunk:
                        break
                    mv[got:got + len(chunk)] = chunk
                    got += len(chunk)
            except Exception:
                break

            if got != need:
                time.sleep(0.01)
                continue

            try:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
                with self._lock:
                    self._last_frame = frame
                self._has_frame.set()
            except Exception:
                continue

        self._has_frame.set()

    def read(self, timeout_s: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        if self.process is None or self.process.poll() is not None:
            return False, None
        if not self._has_frame.wait(timeout=timeout_s):
            return False, None
        with self._lock:
            if self._last_frame is None:
                return False, None
            return True, self._last_frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._has_frame.set()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None

# --- yolo ---
def _load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_MODEL_PATH)

def _detect(yolo, frame: np.ndarray):
    # car=2, truck=7
    results = yolo(frame, classes=[2, 7], conf=MIN_CONFIDENCE, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BBOX_AREA:
                continue
            dets.append((x1, y1, x2, y2, conf))
    return dets

# --- drawing ---
def _draw_overlay(frame: np.ndarray, line_ratio: float, direction: str, dets, title: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    line_y = int(h * line_ratio)

    out = frame.copy()

    # линия
    cv2.line(out, (0, line_y), (w - 1, line_y), (0, 255, 255), 2)
    cv2.putText(out, f"line_y_ratio={line_ratio:.3f} dir={direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # bbox
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # заголовок + fps
    cv2.putText(out, f"{title} | FPS={fps:.1f}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out

def main():
    if not PLATE_CAMERA_RTSP or not SNOW_CAMERA_RTSP:
        print("[PREVIEW] ERROR: PLATE_CAMERA_RTSP / SNOW_CAMERA_RTSP not set in app.env")
        return

    print("[PREVIEW] Controls:")
    print("  Q / ESC  -> quit")
    print("  1 / 2    -> select Plate(1) or Snow(2) line")
    print("  W / S    -> move selected line up/down (small step)")
    print("  E / D    -> move selected line up/down (big step)")
    print("  R        -> reload app.env (ratios)")
    print("")
    print(f"[PREVIEW] Plate ratio={PLATE_LINE_Y_POSITION:.3f}, Snow ratio={SNOW_LINE_Y_POSITION:.3f}")

    yolo = _load_yolo()

    plate = FFmpegRTSPReader(PLATE_CAMERA_RTSP, "plate", FFMPEG_OUT_W, FFMPEG_OUT_H)
    snow = FFmpegRTSPReader(SNOW_CAMERA_RTSP, "snow", FFMPEG_OUT_W, FFMPEG_OUT_H)
    if not plate.start() or not snow.start():
        plate.stop()
        snow.stop()
        return

    sel = 1  # 1=plate, 2=snow
    plate_ratio = PLATE_LINE_Y_POSITION
    snow_ratio = SNOW_LINE_Y_POSITION

    last_t = time.time()
    fps = 0.0
    fc = 0

    frame_i = 0
    try:
        while True:
            ok1, f1 = plate.read(timeout_s=1.0)
            ok2, f2 = snow.read(timeout_s=1.0)
            if not ok1 or f1 is None or not ok2 or f2 is None:
                continue

            frame_i += 1
            dets1 = []
            dets2 = []
            if frame_i % max(1, DETECTION_INTERVAL) == 0:
                dets1 = _detect(yolo, f1)
                dets2 = _detect(yolo, f2)

            fc += 1
            now = time.time()
            if now - last_t >= 1.0:
                fps = fc / (now - last_t)
                fc = 0
                last_t = now

            vis1 = _draw_overlay(f1, plate_ratio, PLATE_LINE_DIRECTION, dets1, "PLATE (press 1)", fps)
            vis2 = _draw_overlay(f2, snow_ratio, SNOW_LINE_DIRECTION, dets2, "SNOW  (press 2)", fps)

            # подпись какая линия выбрана
            if sel == 1:
                cv2.putText(vis1, "SELECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(vis2, "SELECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            stacked = np.vstack([vis1, vis2])
            cv2.imshow("RTSP Preview (Plate top, Snow bottom)", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("1"):
                sel = 1
            if key == ord("2"):
                sel = 2

            # steps
            small = 0.005
            big = 0.02

            if key == ord("w"):
                if sel == 1: plate_ratio = max(0.05, plate_ratio - small)
                else: snow_ratio = max(0.05, snow_ratio - small)
            if key == ord("s"):
                if sel == 1: plate_ratio = min(0.95, plate_ratio + small)
                else: snow_ratio = min(0.95, snow_ratio + small)

            if key == ord("e"):
                if sel == 1: plate_ratio = max(0.05, plate_ratio - big)
                else: snow_ratio = max(0.05, snow_ratio - big)
            if key == ord("d"):
                if sel == 1: plate_ratio = min(0.95, plate_ratio + big)
                else: snow_ratio = min(0.95, snow_ratio + big)

            if key == ord("r"):
                _load_env_vars()
                plate_ratio = float(os.getenv("PLATE_LINE_Y_POSITION", f"{plate_ratio:.3f}"))
                snow_ratio = float(os.getenv("SNOW_LINE_Y_POSITION", f"{snow_ratio:.3f}"))
                print(f"[PREVIEW] Reloaded env: Plate={plate_ratio:.3f}, Snow={snow_ratio:.3f}")

            # печать текущих
            if key in (ord("w"), ord("s"), ord("e"), ord("d")):
                print(f"[PREVIEW] Plate={plate_ratio:.3f} | Snow={snow_ratio:.3f}")

    finally:
        plate.stop()
        snow.stop()
        cv2.destroyAllWindows()

        print("\n[PREVIEW] Final ratios you should put into app.env:")
        print(f"PLATE_LINE_Y_POSITION={plate_ratio:.3f}")
        print(f"SNOW_LINE_Y_POSITION={snow_ratio:.3f}")

if __name__ == "__main__":
    main()
