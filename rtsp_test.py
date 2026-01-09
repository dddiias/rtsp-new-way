import os
import time

# ВАЖНО: ДО import cv2
# Попробуем сначала TCP (как у тебя), но без буферов и с явными таймаутами
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;10000000|rw_timeout;10000000|fflags;nobuffer|flags;low_delay|max_delay;0"
)

import cv2  # noqa: E402

RTSP = "rtsp://admin:Armat456321@178.22.170.253:554/Streaming/Channels/101"


def _open_rtsp_stream_soft(self, rtsp_url: str, stream_name: str, max_retries: int = 3):
    display_url = rtsp_url
    if '@' in rtsp_url:
        a, b = rtsp_url.split('@', 1)
        display_url = f"rtsp://***@{b}"

    print(f"[STREAM] Attempting to connect to {stream_name} (soft mode): {display_url}")

    # ✅ 0) Если включили USE_FFMPEG_DIRECT — сразу пробуем ffmpeg и не трогаем OpenCV
    if self.use_ffmpeg_direct:
        for attempt in range(1, max_retries + 1):
            print(f"[STREAM] FFmpeg-direct mode (attempt {attempt}/{max_retries})...")
            r = FFmpegRTSPReader(rtsp_url)
            if r.start():
                time.sleep(1.0)
                ok, fr = r.read()
                if ok and fr is not None and fr.size > 0:
                    h, w = fr.shape[:2]
                    print(f"[STREAM] ✓✓ FFmpeg-direct works for {stream_name}! Frame: {w}x{h}")
                    return r
                r.release()
            time.sleep(1.0)
        print(f"[STREAM] ✗✗ FFmpeg-direct failed for {stream_name}")
        return None

    # дальше — твой старый OpenCV способ (если ffmpeg-direct выключен)
    ...


cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)

# если поддерживается твоим OpenCV — хорошо, если нет — просто пропустится
try:
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
except Exception:
    pass

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("isOpened:", cap.isOpened())

cv2.namedWindow("rtsp", cv2.WINDOW_NORMAL)

last_print = time.time()
while True:
    # read() может блокироваться, но с таймаутами должно отпускать быстрее
    ret, frame = cap.read()

    if ret and frame is not None and frame.size > 0:
        cv2.imshow("rtsp", frame)
    else:
        if time.time() - last_print > 2:
            print("no frame yet...")
            last_print = time.time()

    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
