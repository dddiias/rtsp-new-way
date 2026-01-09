from __future__ import annotations

import os
import json
import threading
import time
import subprocess
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque, Union

import cv2
import numpy as np
import httpx


# =========================
# 0) Env loader (app.env)
# =========================

def _load_env_vars() -> Optional[str]:
    """Загружает переменные окружения из app.env с override=True."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            return env_path
        else:
            print(f"[STREAM] WARNING: app.env not found at {env_path}, using system env vars")
            return None
    except ImportError:
        print("[STREAM] WARNING: python-dotenv not installed, using system env vars only")
        return None

_env_path = _load_env_vars()
if _env_path:
    print(f"[STREAM] Loaded environment from: {_env_path}")


def _get_env_float(key: str, fallback_key: str | None = None, default: float = 0.6) -> float:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        print(f"[STREAM] WARNING: Invalid float value for {key}: {v}, using default {default}")
        return default


def _get_env_str(key: str, fallback_key: str | None = None, default: str = "down") -> str:
    v = os.getenv(key)
    if v is None and fallback_key:
        v = os.getenv(fallback_key)
    if v is None:
        return default
    return str(v).strip().lower()


# =========================
# 1) Settings
# =========================

PLATE_CAMERA_RTSP = os.getenv("PLATE_CAMERA_RTSP", "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")
SNOW_CAMERA_RTSP  = os.getenv("SNOW_CAMERA_RTSP",  "rtsp://USER:PASSWORD@HOST:554/Streaming/Channels/101")

PLATE_LINE_Y_POSITION = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
PLATE_LINE_DIRECTION  = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")

SNOW_LINE_Y_POSITION  = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
SNOW_LINE_DIRECTION   = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")

MIN_CONFIDENCE = float(os.getenv("STREAM_MIN_CONFIDENCE", "0.5"))
MIN_BBOX_AREA  = int(os.getenv("STREAM_MIN_BBOX_AREA", "10000"))
DETECTION_INTERVAL = int(os.getenv("STREAM_DETECTION_INTERVAL", "3"))

TRACK_MAX_AGE          = int(os.getenv("TRACK_MAX_AGE", "30"))
TRACK_MIN_HITS         = int(os.getenv("TRACK_MIN_HITS", "3"))
TRACK_IOU_THRESHOLD    = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
TRACK_CROSS_COOLDOWN_S = float(os.getenv("TRACK_CROSS_COOLDOWN_S", "1.0"))

DEDUP_WINDOW_SECONDS = float(os.getenv("STREAM_DEDUP_WINDOW_SECONDS", "5.0"))

SHOW_STREAM_WINDOW = os.getenv("SHOW_STREAM_WINDOW", "false").strip().lower() == "true"

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events")
PLATE_CAMERA_ID = os.getenv("PLATE_CAMERA_ID", "camera-001")

FFMPEG_OUT_W = int(os.getenv("FFMPEG_OUT_W", "1280"))
FFMPEG_OUT_H = int(os.getenv("FFMPEG_OUT_H", "720"))

USE_FFMPEG_DIRECT = os.getenv("USE_FFMPEG_DIRECT", "false").strip().lower() == "true"

FFMPEG_BIN_ENV = os.getenv("FFMPEG_BIN", "").strip()

print(f"[STREAM] Plate camera line: Y={PLATE_LINE_Y_POSITION}, direction={PLATE_LINE_DIRECTION}")
print(f"[STREAM] Snow camera line:  Y={SNOW_LINE_Y_POSITION}, direction={SNOW_LINE_DIRECTION}")
print(f"[STREAM] USE_FFMPEG_DIRECT={USE_FFMPEG_DIRECT}, FFMPEG_OUT={FFMPEG_OUT_W}x{FFMPEG_OUT_H}")


def _resolve_ffmpeg_bin() -> Optional[str]:
    """
    Решаем проблему 'ffmpeg виден в одном терминале, но не виден в Cursor/venv':
    - если задан FFMPEG_BIN -> используем его
    - иначе пробуем shutil.which("ffmpeg")
    """
    if FFMPEG_BIN_ENV:
        if os.path.exists(FFMPEG_BIN_ENV):
            return FFMPEG_BIN_ENV
        print(f"[FFMPEG] WARNING: FFMPEG_BIN is set but file not found: {FFMPEG_BIN_ENV}")

    p = shutil.which("ffmpeg")
    if p:
        return p
    return None


# =========================
# 2) Models
# =========================

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]      # x1,y1,x2,y2
    center: Tuple[int, int]              # cx,cy
    confidence: float
    age: int
    hits: int
    last_seen_ts: float
    crossed: bool
    direction: Optional[str]
    last_cross_ts: float = 0.0


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float


# =========================
# 3) Line crossing detector
# =========================

class LineCrossingDetector:
    """Простой IOU-трекер + детектор пересечения горизонтальной линии по центру bbox."""
    def __init__(self, line_y_ratio: float, direction: str = "down"):
        self.line_y_ratio = float(line_y_ratio)
        self.direction = direction.strip().lower()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, frame: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> list[Track]:
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        now_ts = time.time()
        line_y = int(h * self.line_y_ratio)

        # состариваем
        for tr in self.tracks.values():
            tr.age += 1

        matched_dets: set[int] = set()
        crossed_now_tracks: list[Track] = []

        sorted_tracks = sorted(self.tracks.items(), key=lambda kv: (kv[1].hits, -kv[1].age), reverse=True)

        for track_id, tr in sorted_tracks:
            best_iou = 0.0
            best_det_idx: Optional[int] = None

            for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                iou = self._iou(tr.bbox, (x1, y1, x2, y2))
                if iou > best_iou and iou >= TRACK_IOU_THRESHOLD:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is None:
                continue

            x1, y1, x2, y2, conf = detections[best_det_idx]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            prev_cy = tr.center[1]
            crossed_now = False

            if (not tr.crossed) and (now_ts - tr.last_cross_ts >= TRACK_CROSS_COOLDOWN_S):
                if self.direction == "down":
                    if prev_cy < line_y <= cy:
                        crossed_now = True
                else:
                    if prev_cy > line_y >= cy:
                        crossed_now = True

            tr.bbox = (x1, y1, x2, y2)
            tr.center = (cx, cy)
            tr.confidence = conf
            tr.age = 0
            tr.hits += 1
            tr.last_seen_ts = now_ts

            if crossed_now:
                tr.crossed = True
                tr.direction = self.direction
                tr.last_cross_ts = now_ts
                crossed_now_tracks.append(tr)

            matched_dets.add(best_det_idx)

        # новые треки
        for det_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            if det_idx in matched_dets:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tr = Track(
                track_id=self.next_track_id,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
                age=0,
                hits=1,
                last_seen_ts=now_ts,
                crossed=False,
                direction=None,
                last_cross_ts=0.0,
            )
            self.tracks[self.next_track_id] = tr
            self.next_track_id += 1

        # чистим старые
        to_remove = []
        for tid, tr in self.tracks.items():
            if tr.age > TRACK_MAX_AGE:
                to_remove.append(tid)
            elif tr.hits < TRACK_MIN_HITS and tr.age > 5:
                to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return crossed_now_tracks


# =========================
# 4) FFmpeg RTSP Reader (STABLE)
# =========================

class FFmpegRTSPReader:
    """
    Надёжный RTSP reader через ffmpeg:
      - используем явный путь к ffmpeg (FFMPEG_BIN), чтобы не зависеть от PATH Cursor/venv
      - stderr -> DEVNULL (иначе ffmpeg может зависнуть по буферу stderr)
      - stdout читаем в отдельном thread и держим last_frame
      - read() ждёт кадр ограниченное время
      - масштабируем до FFMPEG_OUT_W x FFMPEG_OUT_H
    """
    def __init__(self, rtsp_url: str, name: str):
        self.rtsp_url = rtsp_url
        self.name = name

        self.width = FFMPEG_OUT_W
        self.height = FFMPEG_OUT_H
        self.frame_size = self.width * self.height * 3  # bgr24

        self.process: Optional[subprocess.Popen] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._has_frame = threading.Event()

        self._ffmpeg_bin = _resolve_ffmpeg_bin()

    def start(self) -> bool:
        if not self._ffmpeg_bin:
            print("[FFMPEG] ERROR: ffmpeg not found for this process.")
            print("[FFMPEG] Fix: set FFMPEG_BIN in app.env using: where.exe ffmpeg")
            return False

        try:
            vf = f"scale={self.width}:{self.height}"

            cmd = [
                self._ffmpeg_bin,
                "-hide_banner",
                "-loglevel", "error",
                "-nostats",

                "-rtsp_transport", "tcp",

                # делаем поток более терпимым к битому h264
                "-fflags", "+nobuffer+discardcorrupt",
                "-flags", "low_delay",
                "-err_detect", "ignore_err",

                # (опционально) уменьшить задержки анализа
                "-analyzeduration", "0",
                "-probesize", "32",

                "-i", self.rtsp_url,
                "-an",
                "-vf", vf,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-"
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
                self.release()
                return False

            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ffmpeg-reader-{self.name}")
            self._thread.start()

            print(f"[FFMPEG:{self.name}] started via: {self._ffmpeg_bin} output={self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"[FFMPEG:{self.name}] start failed: {e}")
            self.release()
            return False

    def _loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None

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
                time.sleep(0.02)
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

    def isOpened(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def release(self) -> None:
        self._stop.set()
        try:
            self._has_frame.set()
        except Exception:
            pass

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

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None


RTSPHandle = Union[cv2.VideoCapture, FFmpegRTSPReader]


# =========================
# 5) StreamProcessor
# =========================

class StreamProcessor:
    def __init__(self, merger):
        self.merger = merger
        _load_env_vars()

        self.use_ffmpeg_direct = USE_FFMPEG_DIRECT

        plate_y = _get_env_float("PLATE_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        plate_dir = _get_env_str("PLATE_LINE_DIRECTION", "LINE_DIRECTION", "down")

        snow_y = _get_env_float("SNOW_LINE_Y_POSITION", "LINE_Y_POSITION", 0.6)
        snow_dir = _get_env_str("SNOW_LINE_DIRECTION", "LINE_DIRECTION", "down")

        self.plate_detector = LineCrossingDetector(plate_y, plate_dir)
        self.snow_detector  = LineCrossingDetector(snow_y, snow_dir)

        print("[STREAM] Initialized detectors:")
        print(f"[STREAM]   Plate: Y={plate_y}, direction={plate_dir}")
        print(f"[STREAM]   Snow:  Y={snow_y}, direction={snow_dir}")

        self.plate_cap: Optional[RTSPHandle] = None
        self.snow_cap: Optional[RTSPHandle] = None

        self._stop_event = threading.Event()
        self._snow_thread: Optional[threading.Thread] = None
        self._plate_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._snow_frame_buffer: Deque[TimestampedFrame] = deque(maxlen=120)
        self._snow_crossing_frames: Deque[dict] = deque(maxlen=20)
        self._snow_buffer_lock = threading.Lock()
        self._snow_cross_lock = threading.Lock()

        self._processed_plates: Dict[str, float] = {}
        self._plates_lock = threading.Lock()

        self._task_queue: "deque[dict]" = deque()
        self._task_lock = threading.Lock()
        self._task_signal = threading.Event()

        self.yolo_model = None
        self._yolo_lock = threading.Lock()
        self._load_yolo_model()

    # --------- YOLO ---------

    def _load_yolo_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[STREAM] YOLO model loaded: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[STREAM] ERROR: Failed to load YOLO model: {e}")
            self.yolo_model = None

    def _detect_vehicles(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self.yolo_model is None:
            return []
        try:
            with self._yolo_lock:
                # COCO: car=2, truck=7
                results = self.yolo_model(frame, classes=[2, 7], conf=MIN_CONFIDENCE, verbose=False)

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_BBOX_AREA:
                        continue
                    detections.append((x1, y1, x2, y2, conf))
            return detections
        except Exception as e:
            print(f"[STREAM] Error in vehicle detection: {e}")
            return []

    # --------- frame helpers ---------

    def _validate_frame(self, frame: np.ndarray) -> bool:
        try:
            if frame is None or frame.size == 0:
                return False
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return False
            if frame.dtype != np.uint8:
                return False
            if np.all(frame == 0):
                return False
            if np.any(np.isnan(frame)):
                return False
            return True
        except Exception:
            return False

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if not self._validate_frame(frame):
            return None
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None
            b = buf.tobytes()
            return b if b else None
        except Exception as e:
            print(f"[STREAM] Error encoding frame: {e}")
            return None

    def _get_snow_frame(self, prefer_crossing: bool = True) -> Optional[np.ndarray]:
        now_ts = time.time()

        if prefer_crossing:
            with self._snow_cross_lock:
                while self._snow_crossing_frames and (now_ts - self._snow_crossing_frames[0]["timestamp"] > 2.0):
                    self._snow_crossing_frames.popleft()

                if self._snow_crossing_frames:
                    item = self._snow_crossing_frames[-1]
                    fr = item["frame"]
                    if self._validate_frame(fr):
                        return fr.copy()

        with self._snow_buffer_lock:
            if not self._snow_frame_buffer:
                return None
            for i in range(len(self._snow_frame_buffer) - 1, -1, -1):
                fr = self._snow_frame_buffer[i].frame
                if self._validate_frame(fr):
                    return fr.copy()

        return None

    # --------- task queue ---------

    def _push_task(self, task: dict) -> None:
        with self._task_lock:
            self._task_queue.append(task)
            self._task_signal.set()

    def _pop_task(self) -> Optional[dict]:
        with self._task_lock:
            if not self._task_queue:
                self._task_signal.clear()
                return None
            return self._task_queue.popleft()

    # --------- RTSP open/read ---------

    @staticmethod
    def _mask_url(rtsp_url: str) -> str:
        if "@" in rtsp_url:
            a, b = rtsp_url.split("@", 1)
            return f"rtsp://***@{b}"
        return rtsp_url

    def _open_rtsp(self, rtsp_url: str, name: str, retries: int = 5) -> Optional[RTSPHandle]:
        print(f"[STREAM] Opening {name}: {self._mask_url(rtsp_url)}")

        if self.use_ffmpeg_direct:
            for attempt in range(1, retries + 1):
                reader = FFmpegRTSPReader(rtsp_url, name=name.replace(" ", "_").lower())
                if reader.start():
                    ok, fr = reader.read(timeout_s=3.0)
                    if ok and fr is not None and fr.size > 0:
                        h, w = fr.shape[:2]
                        print(f"[STREAM] ✓ {name} FFmpeg OK: {w}x{h}")
                        return reader
                    print(f"[STREAM] ⚠ {name}: ffmpeg started but no frames yet (attempt {attempt})")
                    reader.release()
                time.sleep(1.2 * attempt)

            print(f"[STREAM] ✗ {name}: FFmpeg failed after {retries} attempts")
            return None

        # OpenCV fallback (если вдруг захочешь выключить USE_FFMPEG_DIRECT)
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            time.sleep(1.0)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                print(f"[STREAM] ✗ {name}: OpenCV isOpened=False (attempt {attempt})")
                time.sleep(1.2 * attempt)
                continue

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ok_any = False
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok_any = True
                    h, w = frame.shape[:2]
                    print(f"[STREAM] ✓ {name} OpenCV OK: {w}x{h}")
                    break
                time.sleep(0.2)

            if ok_any:
                return cap

            try:
                cap.release()
            except Exception:
                pass

            print(f"[STREAM] ⚠ {name}: OpenCV opened but no frames (attempt {attempt})")
            time.sleep(1.2 * attempt)

        print(f"[STREAM] ✗ {name}: OpenCV failed after {retries} attempts")
        return None

    def _read_frame(self, handle: RTSPHandle) -> Tuple[bool, Optional[np.ndarray]]:
        if isinstance(handle, FFmpegRTSPReader):
            return handle.read(timeout_s=1.0)

        if handle is None or not handle.isOpened():
            return False, None

        for _ in range(3):
            ret, frame = handle.read()
            if ret and frame is not None and frame.size > 0 and len(frame.shape) == 3:
                return True, frame
            time.sleep(0.02)

        return False, None

    def _close_handle(self, handle: Optional[RTSPHandle]) -> None:
        if handle is None:
            return
        try:
            if isinstance(handle, FFmpegRTSPReader):
                handle.release()
            else:
                handle.release()
        except Exception:
            pass

    # --------- async processing ---------

    async def _process_crossing_async(self, plate_frame: np.ndarray) -> None:
        now_ts = time.time()

        snow_frame = self._get_snow_frame(prefer_crossing=True)
        if snow_frame is None:
            print("[STREAM] No snow frame available, skipping")
            return

        plate_bytes = self._encode_frame_to_jpeg(plate_frame)
        snow_bytes  = self._encode_frame_to_jpeg(snow_frame)
        if plate_bytes is None or snow_bytes is None:
            print("[STREAM] Failed to encode frames, skipping")
            return

        try:
            gemini_result = await self.merger.analyze_with_gemini(
                snow_photo=snow_bytes,
                plate_photo_1=plate_bytes,
                plate_photo_2=None,
                camera_plate=None,
            )
            print(f"[STREAM] Gemini result: {gemini_result}")
        except Exception as e:
            print(f"[STREAM] Gemini error: {e}")
            return

        plate = (gemini_result or {}).get("plate")
        plate_conf = float((gemini_result or {}).get("plate_confidence", 0.0) or 0.0)

        if plate:
            plate = str(plate).strip().upper()
            with self._plates_lock:
                old = [p for p, ts in self._processed_plates.items() if (now_ts - ts) > DEDUP_WINDOW_SECONDS]
                for p in old:
                    self._processed_plates.pop(p, None)

                if plate in self._processed_plates:
                    print(f"[STREAM] Duplicate plate (Gemini): {plate}, skipping")
                    return

                self._processed_plates[plate] = now_ts

        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        event_data = {
            "camera_id": PLATE_CAMERA_ID,
            "event_time": now_iso,
            "plate": plate,
            "confidence": plate_conf,
            "direction": self.plate_detector.direction,
            "lane": 0,
            "vehicle": {},
            "plate_source": "gemini",
            "snow_volume_percentage": float((gemini_result or {}).get("snow_percentage", 0.0) or 0.0),
            "snow_volume_confidence": float((gemini_result or {}).get("snow_confidence", 0.0) or 0.0),
            "matched_snow": True,
            "gemini_result": gemini_result,
            "timestamp": now_iso,
        }

        try:
            data = {"event": json.dumps(event_data, ensure_ascii=False)}
            files = [
                ("photos", ("detectionPicture.jpg", plate_bytes, "image/jpeg")),
                ("photos", ("snowSnapshot.jpg", snow_bytes, "image/jpeg")),
            ]
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(UPSTREAM_URL, data=data, files=files)
            print(f"[STREAM] Upstream: status={resp.status_code}, ok={resp.is_success}, body={resp.text[:200]}")
        except Exception as e:
            print(f"[STREAM] Upstream send error: {e}")

    def _worker_loop(self) -> None:
        print("[STREAM] Worker thread started")
        while not self._stop_event.is_set():
            if not self._task_signal.wait(timeout=0.2):
                continue

            task = self._pop_task()
            if task is None:
                continue

            plate_frame = task.get("plate_frame")
            if plate_frame is None:
                continue

            try:
                import asyncio
                asyncio.run(self._process_crossing_async(plate_frame))
            except Exception as e:
                print(f"[STREAM] Worker error: {e}")

        print("[STREAM] Worker thread stopped")

    # --------- loops ---------

    def _snow_processing_loop(self) -> None:
        print("[STREAM] Starting snow stream processing...")
        self.snow_cap = self._open_rtsp(SNOW_CAMERA_RTSP, "Snow Camera", retries=5)
        if self.snow_cap is None:
            print("[STREAM] ERROR: Cannot open snow camera")
            return

        fail = 0
        frame_counter = 0

        while not self._stop_event.is_set():
            opened = self.snow_cap.isOpened() if not isinstance(self.snow_cap, FFmpegRTSPReader) else self.snow_cap.isOpened()
            if not opened:
                print("[STREAM] Snow stream closed, reconnecting...")
                self._close_handle(self.snow_cap)
                time.sleep(2)
                self.snow_cap = self._open_rtsp(SNOW_CAMERA_RTSP, "Snow Camera", retries=5)
                if self.snow_cap is None:
                    time.sleep(2)
                    continue

            ret, frame = self._read_frame(self.snow_cap)
            if not ret or frame is None or not self._validate_frame(frame):
                fail += 1
                if fail % 20 == 0:
                    print(f"[STREAM] Snow: {fail} failed reads...")
                time.sleep(0.02)
                continue

            fail = 0
            now_ts = time.time()

            with self._snow_buffer_lock:
                while self._snow_frame_buffer and (now_ts - self._snow_frame_buffer[0].timestamp > 3.0):
                    self._snow_frame_buffer.popleft()
                self._snow_frame_buffer.append(TimestampedFrame(frame=frame.copy(), timestamp=now_ts))

            frame_counter += 1
            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                dets = self._detect_vehicles(frame)
                if dets:
                    crossed = self.snow_detector.process_frame(frame, dets)
                    for tr in crossed:
                        h, _w = frame.shape[:2]
                        line_y = int(h * self.snow_detector.line_y_ratio)
                        cy = tr.center[1]
                        ok = (self.snow_detector.direction == "down" and cy >= line_y) or (
                            self.snow_detector.direction == "up" and cy <= line_y
                        )
                        if ok:
                            with self._snow_cross_lock:
                                self._snow_crossing_frames.append({
                                    "frame": frame.copy(),
                                    "timestamp": now_ts,
                                    "track_id": tr.track_id,
                                })
                            print(f"[STREAM] Snow crossing: track_id={tr.track_id}, cy={cy}, line_y={line_y}")

            time.sleep(0.005)

        self._close_handle(self.snow_cap)
        print("[STREAM] Snow processing loop stopped")

    def _plate_processing_loop(self) -> None:
        print("[STREAM] Starting plate stream processing...")
        self.plate_cap = self._open_rtsp(PLATE_CAMERA_RTSP, "Plate Camera", retries=5)
        if self.plate_cap is None:
            print("[STREAM] ERROR: Cannot open plate camera")
            return

        fail = 0
        frame_counter = 0

        while not self._stop_event.is_set():
            opened = self.plate_cap.isOpened() if not isinstance(self.plate_cap, FFmpegRTSPReader) else self.plate_cap.isOpened()
            if not opened:
                print("[STREAM] Plate stream closed, reconnecting...")
                self._close_handle(self.plate_cap)
                time.sleep(2)
                self.plate_cap = self._open_rtsp(PLATE_CAMERA_RTSP, "Plate Camera", retries=5)
                if self.plate_cap is None:
                    time.sleep(2)
                    continue

            ret, frame = self._read_frame(self.plate_cap)
            if not ret or frame is None or not self._validate_frame(frame):
                fail += 1
                if fail % 20 == 0:
                    print(f"[STREAM] Plate: {fail} failed reads...")
                time.sleep(0.02)
                continue

            fail = 0
            frame_counter += 1

            if frame_counter % max(1, DETECTION_INTERVAL) == 0:
                dets = self._detect_vehicles(frame)
                if dets:
                    crossed = self.plate_detector.process_frame(frame, dets)
                    for tr in crossed:
                        print(f"[STREAM] Plate crossing: track_id={tr.track_id}, bbox={tr.bbox}")
                        self._push_task({"plate_frame": frame.copy()})

            time.sleep(0.005)

        self._close_handle(self.plate_cap)
        print("[STREAM] Plate processing loop stopped")

    # --------- Public API ---------

    def start(self) -> None:
        if self._plate_thread and self._plate_thread.is_alive():
            print("[STREAM] Already running")
            return

        self._stop_event.clear()

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="stream-worker")
        self._worker_thread.start()

        self._snow_thread = threading.Thread(target=self._snow_processing_loop, daemon=True, name="snow-processor")
        self._snow_thread.start()

        time.sleep(1)

        self._plate_thread = threading.Thread(target=self._plate_processing_loop, daemon=True, name="plate-processor")
        self._plate_thread.start()

        print("[STREAM] Stream processor started (snow + plate + worker)")

    def stop(self) -> None:
        self._stop_event.set()
        self._task_signal.set()

        for th in [self._plate_thread, self._snow_thread, self._worker_thread]:
            if th:
                th.join(timeout=5)

        self._plate_thread = None
        self._snow_thread = None
        self._worker_thread = None

        print("[STREAM] Stream processor stopped")


# =========================
# Singleton helpers
# =========================

_stream_processor: Optional[StreamProcessor] = None

def init_stream_processor(merger) -> StreamProcessor:
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(merger)
    return _stream_processor

def get_stream_processor() -> Optional[StreamProcessor]:
    return _stream_processor
