import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

def load_dataset(data_root: Path):
    images = []
    labels = []
    label_map = {}      # name -> id
    id_map = {}         # id -> name
    next_id = 0

    for person_name in sorted(os.listdir(data_root)):
        person_dir = data_root / person_name
        if not person_dir.is_dir():
            continue

        if person_name not in label_map:
            label_map[person_name] = next_id
            id_map[next_id] = person_name
            next_id += 1

        label_id = label_map[person_name]

        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = person_dir / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Ensure consistent size
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(label_id)

    return images, np.array(labels, dtype=np.int32), id_map

def main():
    parser = argparse.ArgumentParser(description="Train an LBPH face recognizer from collected images.")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"),
                        help="Path to dataset root (default: ./data)")
    parser.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "models"),
                        help="Path to models dir (default: ./models)")
    args = parser.parse_args()

    data_root = Path(args.data)
    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    images, labels, id_map = load_dataset(data_root)
    if len(images) == 0:
        raise RuntimeError("No training images found in data/. Run collect_faces.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, labels)

    model_path = models_dir / "face_lbph.xml"
    recognizer.write(str(model_path))

    labels_path = models_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved model to: {model_path}")
    print(f"[OK] Saved label map to: {labels_path}")
    print(f"[INFO] People: {list(id_map.values())}")

if __name__ == "__main__":
    main()
