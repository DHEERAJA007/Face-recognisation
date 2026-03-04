import os
import cv2
import time
import argparse
from pathlib import Path
from utils import get_haar_cascade, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Collect face images from webcam for a given person.")
    parser.add_argument("--name", required=True, help="Person name (folder will be created under data/<name>)")
    parser.add_argument("--num", type=int, default=80, help="Number of images to collect (default: 80)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between auto-captures")
    args = parser.parse_args()

    person_name = args.name.strip()
    data_dir = Path(__file__).resolve().parents[1] / "data" / person_name
    ensure_dir(str(data_dir))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    face_cascade = get_haar_cascade()
    count = 0
    last_capture = 0.0

    print("[INFO] Press 'c' to capture manually, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"{person_name}: {count}/{args.num} captured", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Collect Faces - Press 'c' to capture, 'q' to quit", frame)

        # Auto-capture at intervals if a face is present
        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        should_capture = False
        if key == ord('c'):
            should_capture = True
        elif len(faces) > 0 and (now - last_capture) >= args.delay:
            should_capture = True

        if should_capture and len(faces) > 0:
            # Take the largest detected face (closest to camera)
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            img_path = data_dir / f"{person_name}_{count:04d}.png"
            cv2.imwrite(str(img_path), face_img)
            count += 1
            last_capture = now

            if count >= args.num:
                print("[INFO] Done collecting.")
                break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
