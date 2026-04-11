import cv2
import numpy as np
import os

# Load Haar cascade for eye detection
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_eye.xml")
eye_cascade = cv2.CascadeClassifier(cascade_path)
if eye_cascade.empty():
    raise ValueError("Failed to load haarcascade_eye.xml")

# Preprocess the eye image
def preprocess_the_img(image_path):
    """
    Input: file path or already loaded BGR image
    Output: enhanced grayscale image
    """
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path

    if img is None:
        raise ValueError("Image not loaded properly")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # SHARPENING + BLUR for iris texture (same as enrollment)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0,0), 1), -0.5, 0)

    return gray


# Detect iris only in detected eyes
def detect_the_iris(gray):
    iris = None
    circle = None

    # Detect eyes first
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    for (ex, ey, ew, eh) in eyes:
        eye_roi = gray[ey:ey+eh, ex:ex+ew]

        # HoughCircles inside the detected eye
        circles = cv2.HoughCircles(
            eye_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=18,
            minRadius=40,  # increased for high-res input
            maxRadius=120
        )

        if circles is not None:
            circles = np.round(circles).astype(int)  # <-- fix overflow
            x, y, r = circles[0][0]
            x = int(x)
            y = int(y)
            r = int(r)

            h_eye, w_eye = eye_roi.shape
            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, w_eye)
            y2 = min(y + r, h_eye)

            iris = eye_roi[y1:y2, x1:x2]
            circle = (x + ex, y + ey, r)

            # sanity check: radius and brightness
            if r < 40 or not (50 <= np.mean(iris) <= 200):
                iris = None
                circle = None
            else:
                # mark detected eye rectangle (optional for visualization)
                cv2.rectangle(gray, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)
            break  # use first valid eye detected

    return iris, circle


def normalize_the_iris(iris, radial_res=64, angular_res=512):
    if iris is None or iris.size == 0:
        return None

    iris = iris.astype(np.float32)
    h, w = iris.shape
    theta = np.linspace(0, 2 * np.pi, angular_res, dtype=np.float32)
    r = np.linspace(0, 1, radial_res, dtype=np.float32)
    rad_matrix, theta_matrix = np.meshgrid(r, theta)

    x = rad_matrix * (w - 1) * np.cos(theta_matrix) + w / 2
    y = rad_matrix * (h - 1) * np.sin(theta_matrix) + h / 2

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    normalize = cv2.remap(
        iris, x, y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    normalize = np.clip(normalize, 0, 255).astype(np.uint8)
    return normalize


def extract_the_iris_features(iris, size=(80, 100)):
    if iris is None or iris.size == 0:
        return None

    iris_resized = cv2.resize(iris, size)
    flat = iris_resized.flatten().astype(np.float32)
    flat = flat / (np.linalg.norm(flat) + 1e-6)
    threshold = flat.mean()
    code = (flat > threshold).astype(np.uint8)
    return code