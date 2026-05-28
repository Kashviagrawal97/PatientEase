import cv2
import mediapipe as mp
import numpy as np
import time

# ── MediaPipe setup ───────────────────────────────────────────────────────────
_mp_mesh = mp.solutions.face_mesh
_face_mesh = _mp_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices
NOSE_TIP          = 4
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# ── Calibration state ─────────────────────────────────────────────────────────
_calib = {
    "nose_x": None, "nose_y": None,
    "iris_x": None, "iris_y": None,
}

# ── Tunable parameters ────────────────────────────────────────────────────────
HEAD_THRESHOLD = 0.018   # normalized units — increase if too jittery
GAZE_SCALE     = 8.0     # 15 was too high — cursor flew to edges instantly
DWELL_SECONDS  = 0.8     # seconds to trigger a dwell click
SMOOTH_ALPHA   = 0.35    # exponential smoothing — lower = smoother but slower

# ── Smoothing state ───────────────────────────────────────────────────────────
_smooth = {"gaze_x": 0.5, "gaze_y": 0.5}

# ── Dwell state ───────────────────────────────────────────────────────────────
_dwell_zone  = None
_dwell_start = None

# ── 3×3 screen zones ──────────────────────────────────────────────────────────
ZONES = {
    "top-left":   (0.00, 0.00, 0.33, 0.33),
    "top-center": (0.33, 0.00, 0.66, 0.33),
    "top-right":  (0.66, 0.00, 1.00, 0.33),
    "mid-left":   (0.00, 0.33, 0.33, 0.66),
    "center":     (0.33, 0.33, 0.66, 0.66),
    "mid-right":  (0.66, 0.33, 1.00, 0.66),
    "bot-left":   (0.00, 0.66, 0.33, 1.00),
    "bot-center": (0.33, 0.66, 0.66, 1.00),
    "bot-right":  (0.66, 0.66, 1.00, 1.00),
}

def calibrate(nose_x: float, nose_y: float, iris_x: float, iris_y: float):
    _calib["nose_x"] = nose_x
    _calib["nose_y"] = nose_y
    _calib["iris_x"] = iris_x
    _calib["iris_y"] = iris_y
    # Reset smoothing to calibration point
    _smooth["gaze_x"] = 0.5
    _smooth["gaze_y"] = 0.5

def is_calibrated() -> bool:
    return _calib["iris_x"] is not None

def _zone(gx: float, gy: float) -> str:
    for name, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= gx < x2 and y1 <= gy < y2:
            return name
    return "center"

def process_frame(frame: np.ndarray) -> dict:
    global _dwell_zone, _dwell_start, _smooth

    h, w = frame.shape[:2]
    annotated = frame.copy()

    result = {
        "status": "ok",
        "head_direction": "neutral",
        "gaze_x": 0.5, "gaze_y": 0.5,
        "gaze_zone": "center",
        "iris_raw_x": 0.5, "iris_raw_y": 0.5,
        "nose_x_norm": 0.5, "nose_y_norm": 0.5,
        "dwell_progress": 0.0,
        "dwell_trigger": False,
        "annotated_frame": annotated,
    }

    # ── Process full frame (no skipping) ─────────────────────────────────────
    # FIX: landmarks are in normalized coords (0-1), so no need to scale frame.
    # Scaling caused iris dots to appear at wrong positions.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_res = _face_mesh.process(rgb)

    if not mesh_res.multi_face_landmarks:
        result["status"] = "no_landmarks"
        return result

    lm = mesh_res.multi_face_landmarks[0].landmark

    # ── Nose tip ──────────────────────────────────────────────────────────────
    nose = lm[NOSE_TIP]
    nose_x, nose_y = nose.x, nose.y
    result["nose_x_norm"] = round(nose_x, 4)
    result["nose_y_norm"] = round(nose_y, 4)

    # Draw nose dot at correct position
    cv2.circle(annotated, (int(nose_x * w), int(nose_y * h)), 5, (239, 159, 39), -1)

    # ── Head direction ────────────────────────────────────────────────────────
    if not is_calibrated():
        result["status"] = "calibrating"
        result["head_direction"] = "calibrating"
    else:
        dx = nose_x - _calib["nose_x"]
        dy = nose_y - _calib["nose_y"]
        # Determine dominant axis
        if abs(dx) > abs(dy):
            if dx > HEAD_THRESHOLD:
                result["head_direction"] = "right"
            elif dx < -HEAD_THRESHOLD:
                result["head_direction"] = "left"
        else:
            if dy > HEAD_THRESHOLD:
                result["head_direction"] = "down"
            elif dy < -HEAD_THRESHOLD:
                result["head_direction"] = "up"

    # ── Iris tracking ─────────────────────────────────────────────────────────
    li = lm[LEFT_IRIS_CENTER]
    ri = lm[RIGHT_IRIS_CENTER]
    iris_x = (li.x + ri.x) / 2.0
    iris_y = (li.y + ri.y) / 2.0
    result["iris_raw_x"] = round(iris_x, 4)
    result["iris_raw_y"] = round(iris_y, 4)

    # Draw iris dots at correct positions (FIX: was using scaled coords before)
    cv2.circle(annotated, (int(li.x * w), int(li.y * h)), 5, (93, 202, 165), -1)
    cv2.circle(annotated, (int(ri.x * w), int(ri.y * h)), 5, (93, 202, 165), -1)

    # ── Gaze calculation with smoothing ───────────────────────────────────────
    if is_calibrated():
        raw_gx = float(np.clip(0.5 + (iris_x - _calib["iris_x"]) * GAZE_SCALE, 0, 1))
        raw_gy = float(np.clip(0.5 + (iris_y - _calib["iris_y"]) * GAZE_SCALE, 0, 1))
    else:
        raw_gx, raw_gy = iris_x, iris_y

    # FIX: Exponential smoothing — removes jitter without adding lag
    _smooth["gaze_x"] = SMOOTH_ALPHA * raw_gx + (1 - SMOOTH_ALPHA) * _smooth["gaze_x"]
    _smooth["gaze_y"] = SMOOTH_ALPHA * raw_gy + (1 - SMOOTH_ALPHA) * _smooth["gaze_y"]

    gaze_x = round(_smooth["gaze_x"], 3)
    gaze_y = round(_smooth["gaze_y"], 3)
    result["gaze_x"] = gaze_x
    result["gaze_y"] = gaze_y
    result["gaze_zone"] = _zone(gaze_x, gaze_y)

    # ── Dwell detection ───────────────────────────────────────────────────────
    zone = result["gaze_zone"]
    now  = time.time()

    if zone == _dwell_zone:
        elapsed  = now - (_dwell_start or now)
        progress = min(elapsed / DWELL_SECONDS, 1.0)
        result["dwell_progress"] = round(progress, 3)

        if elapsed >= DWELL_SECONDS:
            result["dwell_trigger"] = True
            # FIX: Reset after trigger so it doesn't fire every frame
            _dwell_zone  = None
            _dwell_start = None
    else:
        _dwell_zone  = zone
        _dwell_start = now
        result["dwell_progress"] = 0.0

    # ── Draw dwell arc ────────────────────────────────────────────────────────
    if result["dwell_progress"] > 0:
        cx = int(iris_x * w)
        cy = int(iris_y * h)
        angle = int(360 * result["dwell_progress"])
        cv2.ellipse(annotated, (cx, cy), (22, 22), -90, 0, angle, (239, 159, 39), 2)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    status_color = (100, 220, 100) if is_calibrated() else (0, 200, 255)
    cv2.putText(annotated,
                f"Zone: {result['gaze_zone']}  Dir: {result['head_direction']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
    cv2.putText(annotated,
                f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    if not is_calibrated():
        cv2.putText(annotated, "Look straight — press C to calibrate",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    result["annotated_frame"] = annotated
    return result


# ── Main loop (for standalone testing) ───────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press C = calibrate | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        data  = process_frame(frame)

        if data.get("dwell_trigger"):
            print(f"DWELL CLICK → zone: {data['gaze_zone']}")

        cv2.imshow("Eye Tracker", data["annotated_frame"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and data["status"] != "no_landmarks":
            calibrate(
                data["nose_x_norm"], data["nose_y_norm"],
                data["iris_raw_x"],  data["iris_raw_y"],
            )
            print(f"Calibrated! Nose=({data['nose_x_norm']}, {data['nose_y_norm']})")

    cap.release()
    cv2.destroyAllWindows()