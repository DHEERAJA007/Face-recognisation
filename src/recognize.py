import os
import cv2
import json
import argparse
from pathlib import Path
from utils import get_haar_cascade

def load_labels(labels_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)   # id(str) -> name
    # Convert keys to int
    id_map_int = {int(k): v for k, v in id_map.items()}
    return id_map_int

def main():
    parser = argparse.ArgumentParser(description="Realtime face recognition from webcam using LBPH.")
    parser.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "models"),
                        help="Path to models dir (default: ./models)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="LBPH distance threshold (lower is stricter; default: 60)")
    args = parser.parse_args()

    models_dir = Path(args.models)
    model_path = models_dir / "face_lbph.xml"
    labels_path = models_dir / "labels.json"

    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Model or labels not found. Train first with src/train.py")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    id_map = load_labels(labels_path)
    face_cascade = get_haar_cascade()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            label_id, distance = recognizer.predict(roi)
            name = id_map.get(label_id, "Unknown")

            if distance <= args.threshold:
                display_name = f"{name} ({distance:.1f})"
                color = (0, 255, 0)
            else:
                display_name = f"Unknown ({distance:.1f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition — Press 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
