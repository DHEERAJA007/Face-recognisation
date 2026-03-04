import os
import cv2

def get_haar_cascade():
    # Use OpenCV's built-in path to Haar cascades
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("Haar cascade not found at " + cascade_path)
    return cv2.CascadeClassifier(cascade_path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
