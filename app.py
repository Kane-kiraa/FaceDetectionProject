from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import sys
import os
import threading
import numpy as np
import atexit
import time
import datetime

# ── 📂 កំណត់ Path និងសញ្ញាណសម្គាល់ម៉ូដែលទាំងអស់ ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ១. Gender Model (OpenCV DNN / Caffe)
GENDER_MODEL_DIR = os.path.join(BASE_DIR, "gender_model")
GENDER_PROTO     = os.path.join(GENDER_MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL     = os.path.join(GENDER_MODEL_DIR, "gender_net.caffemodel")
GENDER_LIST      = ["MALE", "FEMALE"]
GENDER_MEAN      = (78.4263377603, 87.7689143744, 114.895847746)

# ២. Age Model (OpenCV DNN / Caffe)
AGE_MODEL_DIR = os.path.join(BASE_DIR, "age_model")
AGE_PROTO     = os.path.join(AGE_MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL     = os.path.join(AGE_MODEL_DIR, "age_net.caffemodel")
AGE_LIST      = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# ៣. Emotion Model (OpenCV DNN / ONNX)
EMOTION_MODEL_DIR  = os.path.join(BASE_DIR, "emotion_model")
EMOTION_MODEL_PATH = os.path.join(EMOTION_MODEL_DIR, "emotion_model.onnx")
EMOTION_LIST       = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']

# ── 🚀 ឡូដម៉ូដែលចូលទៅក្នុងប្រព័ន្ធ ──────────────────────────────────────────
gender_net = None
if os.path.isfile(GENDER_PROTO) and os.path.isfile(GENDER_MODEL):
    try:
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        print("✅ Gender model loaded.")
    except Exception as e:
        print(f"⚠️  Gender model load failed: {e}", file=sys.stderr)
else:
    print("⚠️  Gender model files not found.", file=sys.stderr)

age_net = None
if os.path.isfile(AGE_PROTO) and os.path.isfile(AGE_MODEL):
    try:
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        print("✅ Age model loaded.")
    except Exception as e:
        print(f"⚠️  Age model load failed: {e}", file=sys.stderr)
else:
    print("⚠️  Age model files not found.", file=sys.stderr)

emotion_net = None
if os.path.isfile(EMOTION_MODEL_PATH):
    try:
        emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL_PATH)
        print("✅ Emotion model loaded.")
    except Exception as e:
        print(f"⚠️  Emotion model load failed: {e}", file=sys.stderr)
else:
    print("⚠️  Emotion model file not found.", file=sys.stderr)


app = Flask(__name__)

# ── Thread-safe camera management ─────────────────────────────────────────────
_camera_lock   = threading.Lock()
camera         = None
camera_source  = None
camera_fps     = None

known_face_encodings = []
known_face_names     = []
faces_folder         = "known_faces"

RECOGNITION_SCALE = 0.25
JPEG_QUALITY      = 78
JPEG_QUALITY_60FPS = 65  # Lower quality for 60 FPS to improve encoding speed


def open_camera(source=0, fps=30):
    """Open a camera device and configure FPS. Returns cap or None.

    Strategy:
      1. Try V4L2 first (best for USB / Iriun virtual cameras on Linux).
      2. If that fails, fall back to the OS-default backend (CAP_ANY) which
         handles built-in laptop webcams on Linux, Windows (DSHOW/MSMF),
         and macOS (AVFoundation) automatically.
    """
    # ── attempt 1: explicit V4L2 (Linux USB / virtual cam) ──────────────────
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(source, backend)
        except Exception:
            continue

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "AUTO"
            print(f"✅ Camera opened [{backend_name}]: {actual_w}x{actual_h} @ {fps}fps")
            return cap

        cap.release()

    print(f"❌ Cannot open camera source={source} with any backend.", file=sys.stderr)
    return None


def load_known_faces():
    if not os.path.isdir(faces_folder):
        print(f"⚠️  Known faces folder '{faces_folder}' not found. Creating...", file=sys.stderr)
        os.makedirs(faces_folder)
        return
    for filename in os.listdir(faces_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(faces_folder, filename)
            try:
                image     = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if not encodings:
                    print(f"⚠️  No face found in '{filename}', skipping.", file=sys.stderr)
                    continue
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0].replace("_", " "))
            except Exception as e:
                print(f"❌ Error loading '{filename}': {e}", file=sys.stderr)

    if known_face_names:
        print(f"✅ Loaded {len(known_face_names)} known face(s): {', '.join(known_face_names)}")
    else:
        print("⚠️  No known faces loaded. Everyone will be labelled 'Unknown'.", file=sys.stderr)


def get_camera(source=0, fps=30):
    """Return (and lazily create) the global camera — thread-safe."""
    global camera, camera_source, camera_fps
    with _camera_lock:
        need_new = (
            camera is None
            or not camera.isOpened()
            or camera_source != source
            or camera_fps != fps
        )
        if need_new:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
            camera = camera_source = camera_fps = None
            cap = open_camera(source, fps)
            if cap is not None and cap.isOpened():
                camera        = cap
                camera_source = source
                camera_fps    = fps
            else:
                print("❌ ERROR: Cannot open camera.", file=sys.stderr)
    return camera


def release_camera():
    global camera, camera_source, camera_fps
    with _camera_lock:
        if camera is not None:
            camera.release()
            camera = camera_source = camera_fps = None


atexit.register(release_camera)
load_known_faces()


# ── Face access logging ────────────────────────────────────────────────────────
ACCESS_LOG_FILE = "access_log.txt"
_access_log_lock = threading.Lock()
_last_logged_faces: dict[str, float] = {}  # {name: timestamp} to avoid duplicate logs

def log_face_access(name: str):
    """Log a known face detection with timestamp to access_log.txt.
    
    Avoids duplicate logs for the same person within 10 seconds.
    """
    global _last_logged_faces
    
    if name == "Unknown":
        return  # Don't log unknown faces
    
    current_time = time.time()
    
    # Skip if this person was logged recently (within 10 seconds)
    if name in _last_logged_faces:
        if current_time - _last_logged_faces[name] < 10:
            return
    
    with _access_log_lock:
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | {name}\n"
            with open(ACCESS_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)
            _last_logged_faces[name] = current_time
        except Exception as e:
            print(f"⚠️ Error logging face access: {e}", file=sys.stderr)


def estimate_emotion_from_landmarks(frame, top, right, bottom, left):
    face_crop = frame[top:bottom, left:right]
    if face_crop.size == 0:
        return None

    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(rgb_face)
    if not landmarks:
        return None

    mouth_top = np.array(landmarks[0].get('top_lip', []))
    mouth_bot = np.array(landmarks[0].get('bottom_lip', []))
    if mouth_top.shape[0] < 6 or mouth_bot.shape[0] < 6:
        return None

    left_corner = mouth_top[0]
    right_corner = mouth_top[6] if mouth_top.shape[0] > 6 else mouth_top[-1]
    top_center = np.mean(mouth_top[3:9], axis=0)
    bottom_center = np.mean(mouth_bot[3:9], axis=0)

    mouth_width = np.linalg.norm(right_corner - left_corner)
    mouth_height = abs(bottom_center[1] - top_center[1])
    if mouth_width <= 0:
        return None

    ratio = mouth_height / mouth_width
    if ratio < 0.16:
        return 'Happy'
    if ratio > 0.45:
        return 'Surprise'
    return None


# ── Face attribute prediction ──────────────────────────────────────────────────
def predict_face_attributes(frame, top, right, bottom, left):
    gender_res = age_res = emotion_res = "?"
    h, w = frame.shape[:2]

    pad = 35
    t = max(0, top  - pad)
    b = min(h, bottom + pad)
    l = max(0, left  - pad)
    r = min(w, right  + pad)
    face_crop = frame[t:b, l:r]

    if face_crop.size == 0:
        return gender_res, age_res, emotion_res

    # 1. Gender & Age (Caffe)
    try:
        if gender_net is not None or age_net is not None:
            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227),
                GENDER_MEAN, swapRB=False        # Caffe models expect BGR input + mean subtraction
            )
            if gender_net is not None:
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender_res   = GENDER_LIST[gender_preds[0].argmax()]

            if age_net is not None:
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age_res   = AGE_LIST[age_preds[0].argmax()]
    except Exception as e:
        print(f"⚠️ Gender/Age prediction error: {e}", file=sys.stderr)

    # 2. Emotion (ONNX)
    try:
        if emotion_net is not None:
            gray_face    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray_face    = cv2.resize(gray_face, (64, 64))
            emotion_blob = cv2.dnn.blobFromImage(
                gray_face, 1.0 / 255.0, (64, 64), (0,), swapRB=False, crop=False
            )
            emotion_net.setInput(emotion_blob)
            emotion_preds = emotion_net.forward()
            emotion_res   = EMOTION_LIST[emotion_preds[0].argmax()]

            if emotion_res == 'Neutral':
                fallback = estimate_emotion_from_landmarks(frame, t, r, b, l)
                if fallback is not None:
                    emotion_res = fallback
    except Exception as e:
        print(f"⚠️ Emotion prediction error: {e}", file=sys.stderr)

    return gender_res, age_res, emotion_res


# ── HUD drawing ────────────────────────────────────────────────────────────────
def draw_hud_face_box(frame, top, right, bottom, left, name,
                      is_known, gender="?", age="?", emotion="?"):
    fh, fw = frame.shape[:2]

    color_main   = (0, 232, 160) if is_known else (0, 0, 255)
    color_text   = (255, 255, 255)
    color_accent = (180, 180, 180)
    dash_color   = (120, 120, 120) if is_known else (0, 0, 150)

    w   = right - left
    h   = bottom - top
    arm = max(16, int(min(w, h) * 0.20))
    thick = 2

    # Corner brackets
    def corner(cx, cy, dx, dy):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color_main, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color_main, thick, cv2.LINE_AA)

    corner(left,  top,     +1, +1)
    corner(right, top,     -1, +1)
    corner(left,  bottom,  +1, -1)
    corner(right, bottom,  -1, -1)

    # Dashed border
    dash_len, gap_len = 6, 5

    def dashed_hline(y, x0, x1):
        x = x0 + arm
        while x < x1 - arm:
            xe = min(x + dash_len, x1 - arm)
            cv2.line(frame, (x, y), (xe, y), dash_color, 1, cv2.LINE_AA)
            x += dash_len + gap_len

    def dashed_vline(x, y0, y1):
        y = y0 + arm
        while y < y1 - arm:
            ye = min(y + dash_len, y1 - arm)
            cv2.line(frame, (x, y), (x, ye), dash_color, 1, cv2.LINE_AA)
            y += dash_len + gap_len

    dashed_hline(top,    left, right)
    dashed_hline(bottom, left, right)
    dashed_vline(left,   top,  bottom)
    dashed_vline(right,  top,  bottom)

    # Centre crosshair
    cx_c, cy_c = (left + right) // 2, (top + bottom) // 2
    ch = 8
    cv2.line(frame,   (cx_c - ch, cy_c), (cx_c + ch, cy_c), color_main, 1, cv2.LINE_AA)
    cv2.line(frame,   (cx_c, cy_c - ch), (cx_c, cy_c + ch), color_main, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx_c, cy_c), 3, color_main, 1, cv2.LINE_AA)

    # Info panel
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.42
    line_h     = 22
    panel_pad  = 10
    status_str = "CONFIRMED" if is_known else "UNKNOWN ALERT"

    lines = [
        ("NAME",    name.upper()),
        ("STATUS",  status_str),
        ("GENDER",  gender),
        ("AGE",     age),
        ("EMOTION", emotion.upper()),
    ]

    max_w = 0
    for label, value in lines:
        text = f"{label}: {value}"
        (tw, _), _ = cv2.getTextSize(text, font, font_small, 1)
        max_w = max(max_w, tw)

    panel_w = max_w + panel_pad * 2 + 10
    panel_h = len(lines) * line_h + panel_pad * 2

    # ✅ Clamp panel to frame bounds so it never renders off-screen
    panel_x = right + 12
    panel_y = top
    if panel_x + panel_w > fw:
        panel_x = max(0, left - panel_w - 12)   # flip to left side
    if panel_y + panel_h > fh:
        panel_y = max(0, fh - panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), color_main, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 4,       panel_y + panel_h), color_main, -1)

    for i, (label, value) in enumerate(lines):
        ty        = panel_y + panel_pad + (i + 1) * line_h - 4
        label_str = label + ": "
        (lw, _), _ = cv2.getTextSize(label_str, font, font_small, 1)
        cv2.putText(frame, label_str, (panel_x + panel_pad + 6, ty),          font, font_small, color_accent, 1, cv2.LINE_AA)
        cv2.putText(frame, value,     (panel_x + panel_pad + 6 + lw, ty),     font, font_small, color_text,   1, cv2.LINE_AA)

    # Connector line from box to panel
    cv2.line(frame, (right, top + arm), (panel_x, panel_y + arm), dash_color, 1, cv2.LINE_AA)


# ── Frame generator ────────────────────────────────────────────────────────────
def _rotate_frame(frame, rotate):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def generate_frames(source=0, rotate=0, mirror=False, fps=30):
    cam = get_camera(source, fps)
    if cam is None:
        return

    # ✅ process_interval: analyse every Nth frame for performance
    # At 60 fps: process every 6 frames (10 fps AI, 60 fps video output)
    # At 30 fps: process every 3 frames (10 fps AI, 30 fps video output)
    process_interval = 6 if fps >= 60 else 3
    # ✅ Use round() to avoid off-by-one from float precision (e.g. 1/0.25 = 4.0)
    scale_multiplier = round(1.0 / RECOGNITION_SCALE)
    # Use lower JPEG quality for 60 FPS to maintain throughput
    quality = JPEG_QUALITY_60FPS if fps >= 60 else JPEG_QUALITY
    encode_params    = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    frame_index      = 0
    cached_faces     = []       # list of (top, right, bottom, left, name, is_known)
    # ✅ Key is a unique per-face tuple (top, right, bottom, left) instead of just (top, left)
    #    which could collide when two faces share the same top or left coordinate.
    cached_attributes: dict[tuple, tuple] = {}

    while True:
        with _camera_lock:
            if cam is None or not cam.isOpened():
                break
            success, frame = cam.read()

        if not success:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # ── Detection & recognition (every Nth frame) ──────────────────────
        if frame_index % process_interval == 0:
            small_frame     = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations  = face_recognition.face_locations(rgb_small_frame, model='hog')

            face_names = []
            face_known = []

            # ✅ Only call face_encodings when there are faces AND known encodings
            if face_locations and known_face_encodings:
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations, num_jitters=1
                )
                for face_encoding in face_encodings:
                    name  = "Unknown"
                    known = False
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(distances) > 0:
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= 0.55:
                            name  = known_face_names[best_idx]
                            known = True
                            log_face_access(name)  # ✅ Log known face detection with timestamp
                    face_names.append(name)
                    face_known.append(known)
            else:
                face_names = ["Unknown"] * len(face_locations)
                face_known = [False]     * len(face_locations)

            s = scale_multiplier
            new_cached_faces     = []
            new_cached_attributes: dict[tuple, tuple] = {}

            for (top, right, bottom, left), name, is_known in zip(face_locations, face_names, face_known):
                t, r, b, l = top * s, right * s, bottom * s, left * s
                new_cached_faces.append((t, r, b, l, name, is_known))

                # ✅ Use full bounding box as key (no coordinate collisions)
                key = (t, r, b, l)
                if key in cached_attributes:
                    new_cached_attributes[key] = cached_attributes[key]
                else:
                    new_cached_attributes[key] = predict_face_attributes(frame, t, r, b, l)

            # ✅ Replace old cache — prevents unbounded memory growth
            cached_faces      = new_cached_faces
            cached_attributes = new_cached_attributes

        # ── Draw HUD ────────────────────────────────────────────────────────
        for top, right, bottom, left, name, is_known in cached_faces:
            key = (top, right, bottom, left)
            gender, age, emotion = cached_attributes.get(key, ("?", "?", "?"))
            draw_hud_face_box(frame, top, right, bottom, left, name, is_known,
                              gender=gender, age=age, emotion=emotion)

        # ✅ Rotate BEFORE encoding so the JPEG contains the correctly oriented image
        if rotate:
            frame = _rotate_frame(frame, rotate)

        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            frame_index += 1
            continue

        frame_index += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    source_arg = request.args.get('source', default='0')
    try:
        source = int(source_arg)
    except ValueError:
        source = 0

    rotate_arg = request.args.get('rotate', default='0')
    try:
        rotate = int(rotate_arg) % 360
        if rotate not in (0, 90, 180, 270):
            rotate = 0
    except ValueError:
        rotate = 0

    mirror_arg = request.args.get('mirror', default='0')
    mirror = str(mirror_arg).lower() in ('1', 'true', 'yes', 'on')

    fps_arg = request.args.get('fps', default='30')
    try:
        fps = int(fps_arg)
        if fps not in (30, 60):
            fps = 30
    except ValueError:
        fps = 30

    if get_camera(source, fps) is None:
        return "Unable to open camera. Check connection.", 503

    return Response(
        generate_frames(source=source, rotate=rotate, mirror=mirror, fps=fps),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/cameras')
def api_cameras():
    """Probe camera indices 0–3 and return those that open successfully.
    Uses the same V4L2 -> CAP_ANY fallback as open_camera() so that both
    USB cameras and built-in laptop webcams are discovered correctly.
    """
    available = []
    for i in range(4):
        found = False
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    found = True
                cap.release()
            except Exception:
                pass
            if found:
                break
    return jsonify(available)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)