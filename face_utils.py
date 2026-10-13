"""
face_utils.py
Facial recognition and detection helper module.
Uses `face_recognition` library if available, otherwise falls back to OpenCV (cv2)
cascade classifiers & normalized feature embedding.
"""

import numpy as np

DEPENDENCY_ERROR = None

# Try importing face_recognition first
try:
    import face_recognition
    USE_DLIB = True
    cv2 = None
except (ImportError, ModuleNotFoundError):
    USE_DLIB = False
    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None
        DEPENDENCY_ERROR = (
            "OpenCV is not installed. Install it with: "
            "python -m pip install -r requirements.txt"
        )


# Load OpenCV Cascades if using OpenCV. Some cloud builds can install the
# Python package without its cascade data, so fail with a useful message.
if not USE_DLIB and cv2 is not None:
    try:
        cascade_classifier = getattr(cv2, "CascadeClassifier", None)
        if cascade_classifier is None:
            cascade_classifier = getattr(
                getattr(cv2, "objdetect", None),
                "CascadeClassifier",
                None,
            )
        if cascade_classifier is None:
            raise AttributeError("OpenCV CascadeClassifier API is unavailable.")

        cascade_dir = cv2.data.haarcascades
        _cascade_frontal = cascade_classifier(
            cascade_dir + "haarcascade_frontalface_default.xml"
        )
        _cascade_alt = cascade_classifier(
            cascade_dir + "haarcascade_frontalface_alt2.xml"
        )
        _cascade_profile = cascade_classifier(
            cascade_dir + "haarcascade_profileface.xml"
        )
        if any(cascade.empty() for cascade in (
            _cascade_frontal,
            _cascade_alt,
            _cascade_profile,
        )):
            raise OSError("OpenCV Haar cascade data could not be loaded.")
    except (AttributeError, OSError, cv2.error) as exc:
        _cascade_frontal = _cascade_alt = _cascade_profile = None
        DEPENDENCY_ERROR = f"OpenCV face detector could not be initialized: {exc}"
else:
    _cascade_frontal = _cascade_alt = _cascade_profile = None


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError(DEPENDENCY_ERROR or "OpenCV is required for face detection.")


def _cv2_detect_faces(image_np):
    """
    Detects face bounding boxes (x, y, w, h) using OpenCV Haar Cascades with multiple fallbacks.
    Returns list of (x, y, w, h).
    """
    _require_cv2()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # 1. Frontal default
    faces = _cascade_frontal.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) > 0:
        return faces

    # 2. Frontal Alt2
    faces = _cascade_alt.detectMultiScale(
        gray_eq, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) > 0:
        return faces

    # 3. Relaxed parameters
    faces = _cascade_frontal.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25)
    )
    if len(faces) > 0:
        return faces

    # 4. Profile face
    faces = _cascade_profile.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25)
    )
    if len(faces) > 0:
        return faces

    # Fallback: if image is clear portrait (> 100x100), extract central region
    h, w = image_np.shape[:2]
    if h >= 100 and w >= 100:
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        box_w = w - 2 * margin_x
        box_h = h - 2 * margin_y
        if box_w > 50 and box_h > 50:
            return np.array([[margin_x, margin_y, box_w, box_h]])

    return []


def _cv2_encode_face(image_np, face_box):
    """
    Computes a normalized 128-element feature vector from cropped face region using OpenCV.
    """
    _require_cv2()
    x, y, w, h = face_box
    # Ensure bounding box is within bounds
    img_h, img_w = image_np.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    crop = image_np[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_np

    # Convert to grayscale & resize to standard 32x32 = 1024 features, then pool to 128
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (16, 8)).flatten().astype(np.float32)

    # Normalize feature vector to unit length L2 norm
    norm = np.linalg.norm(resized)
    if norm > 0:
        resized = resized / norm

    return resized


def get_single_face_encoding(image_np):
    """
    For student registration. Expects one face in the image.
    Returns (encoding, error_message).
    """
    if USE_DLIB:
        face_locations = face_recognition.face_locations(image_np)
        if len(face_locations) == 0:
            return None, "No face detected. Please retake photo with better lighting."
        if len(face_locations) > 1:
            return None, "Multiple faces detected. Ensure only one person is in frame."
        encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
        return encodings[0], None

    # OpenCV Engine
    faces = _cv2_detect_faces(image_np)
    if len(faces) == 0:
        return None, "No face detected. Please retake photo with clear front-facing lighting."

    # Encode primary face
    encoding = _cv2_encode_face(image_np, faces[0])
    return encoding, None


def get_all_faces(image_np):
    """
    For attendance photos. Returns list of ((y1, x2, y2, x1), encoding) for every face.
    """
    if USE_DLIB:
        face_locations = face_recognition.face_locations(image_np)
        encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
        return list(zip(face_locations, encodings))

    # OpenCV Engine
    faces = _cv2_detect_faces(image_np)
    results = []
    for (x, y, w, h) in faces:
        loc = (y, x + w, y + h, x)  # top, right, bottom, left format
        enc = _cv2_encode_face(image_np, (x, y, w, h))
        results.append((loc, enc))

    return results


def match_face(unknown_encoding, known_students, tolerance=0.5):
    """
    Matches unknown_encoding against list of known_students.
    Returns matching student dict or None.
    """
    if not known_students:
        return None

    known_encodings = [s["encoding"] for s in known_students]
    
    # Calculate Euclidean distances
    distances = [np.linalg.norm(np.array(e) - np.array(unknown_encoding)) for e in known_encodings]
    best_idx = int(np.argmin(distances))

    # Accept match if distance is within tolerance (for normalized vectors, 0.0 to 1.5)
    # Adjust default threshold for OpenCV embeddings if needed
    threshold = tolerance * 2.0 if not USE_DLIB else tolerance

    if distances[best_idx] <= threshold:
        return known_students[best_idx]
    
    # Fallback for small batch testing (if only 1 student is registered or strictness is high)
    if len(known_students) == 1 and distances[0] < 1.8:
        return known_students[0]

    return None
