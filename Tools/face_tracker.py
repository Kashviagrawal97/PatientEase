import cv2
import numpy as np
from typing import Optional, Tuple

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_prev_center_y = None
_baseline_y = None
_last_known_bbox = None
_frame_counter = 0
_calibration_frames = []
_is_calibrated = False

MOVEMENT_THRESHOLD = 18    # pixels — ignore micro-movements
CALIBRATION_FRAMES = 30    # frames to establish neutral position
SCROLL_AMOUNT = 3          # scroll steps per detection

def get_face_center_y(bbox: Tuple) -> int:
    """Returns Y-coordinate of face center."""
    x, y, w, h = bbox
    return y + h // 2

def calibrate(center_y: int):
    """Build baseline neutral position from first N frames."""
    global _baseline_y, _is_calibrated, _calibration_frames
    _calibration_frames.append(center_y)
    if len(_calibration_frames) >= CALIBRATION_FRAMES:
        _baseline_y = int(np.mean(_calibration_frames))
        _is_calibrated = True
        print(f"[Calibrated] Neutral Y = {_baseline_y}px")

def detect_face(frame: np.ndarray) -> Optional[Tuple]:
    """Detect largest face — runs every frame for accurate tracking."""
    global _last_known_bbox

    try:
        scale_percent = 0.5                          # slightly larger for better tracking
        w = int(frame.shape[1] * scale_percent)
        h = int(frame.shape[0] * scale_percent)
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)                # improves contrast in low light

        faces = _cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,                          # stricter = fewer false detections
            minSize=(60, 60),                        # ignore tiny false positives
        )

        if len(faces) == 0:
            return _last_known_bbox                  # hold last position if lost briefly

        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, bw, bh = largest
        bbox = (
            int(x / scale_percent),
            int(y / scale_percent),
            int(bw / scale_percent),
            int(bh / scale_percent),
        )
        _last_known_bbox = bbox
        return bbox

    except Exception:
        return _last_known_bbox

def get_movement(frame: np.ndarray):
    """
    Returns movement direction: 'UP', 'DOWN', or None.
    Also returns (bbox, delta_y, baseline_y) for visualization.
    """
    global _prev_center_y

    bbox = detect_face(frame)
    if bbox is None:
        return None, None, None, None

    center_y = get_face_center_y(bbox)

    if not _is_calibrated:
        calibrate(center_y)
        return None, bbox, 0, None

    delta_y = center_y - _baseline_y               # negative = moved UP, positive = DOWN

    if abs(delta_y) < MOVEMENT_THRESHOLD:
        direction = None                            # inside dead zone
    elif delta_y < 0:
        direction = "UP"
    else:
        direction = "DOWN"

    return direction, bbox, delta_y, _baseline_y

def draw_face_box(frame: np.ndarray, bbox: Tuple, delta_y: int = 0, direction: str = None) -> np.ndarray:
    """Draw tracking box + movement indicator."""
    if bbox is None:
        return frame

    out = frame.copy()
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2

    # Color based on direction
    if direction == "UP":
        color = (100, 220, 100)       # green
        label = "UP - Scrolling Up"
    elif direction == "DOWN":
        color = (100, 100, 220)       # blue
        label = "DOWN - Scrolling Down"
    else:
        color = (83, 119, 221)        # neutral
        label = "NEUTRAL"

    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Delta Y indicator
    if _baseline_y is not None:
        cv2.putText(out, f"dy={delta_y:+d}px", (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Visual arrow showing movement magnitude
        arrow_len = min(abs(delta_y), 60)
        arrow_dir = -1 if direction == "UP" else 1
        if direction in ("UP", "DOWN"):
            cv2.arrowedLine(out, (cx, cy), (cx, cy + arrow_dir * arrow_len),
                            color, 2, tipLength=0.4)

    # Calibration status
    if not _is_calibrated:
        cv2.putText(out, f"Calibrating... {len(_calibration_frames)}/{CALIBRATION_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return out


# ── Main loop ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pyautogui
    pyautogui.FAILSAFE = True

    cap = cv2.VideoCapture(0)
    print("Look straight at the camera to calibrate neutral position...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)                  # mirror for natural feel
        direction, bbox, delta_y, baseline = get_movement(frame)

        # Browser scroll
        if direction == "UP":
            pyautogui.scroll(SCROLL_AMOUNT)
        elif direction == "DOWN":
            pyautogui.scroll(-SCROLL_AMOUNT)

        vis = draw_face_box(frame, bbox, delta_y or 0, direction)
        cv2.imshow("Face Movement - Head Control", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()