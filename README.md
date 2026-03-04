# Face Recognition (Webcam) — OpenCV LBPH

A minimal, reliable face-recognition project using your laptop camera.
It uses OpenCV's **LBPH** face recognizer and Haar Cascades for detection (works offline).

## Features
- Collect face images from your webcam for any number of people
- Train an LBPH recognizer
- Realtime recognition with on-screen labels
- Simple CLI scripts

## Project Structure
```
face-recognition-project/
├─ data/                 # Collected face images (auto-created)
├─ models/               # Trained model + label mapping
├─ src/
│  ├─ collect_faces.py   # Capture faces for a person
│  ├─ train.py           # Train LBPH model
│  ├─ recognize.py       # Realtime recognition
│  └─ utils.py           # Helpers: loading dataset, cascade path
├─ requirements.txt
└─ README.md
```

## 1) Create & Activate Environment (recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install Dependencies
```bash
pip install -r requirements.txt
```

> NOTE: We require **opencv-contrib-python** (not just opencv-python) for LBPH.

## 3) Collect Face Images
Run this once per person you want to recognize.
```bash
# Example: enroll a person named "Dheeraj"
python src/collect_faces.py --name Dheeraj --num 80
```
**How it works:**
- Press **C** to capture when your face is well-framed (default auto-captures too).
- Press **Q** to quit early.

Tips:
- Make sure your face is well-lit and vary angles/look directions a little.
- Aim for **50-120** images per person.

## 4) Train the Model
```bash
python src/train.py
```
This will produce:
- `models/face_lbph.xml` — the trained LBPH model
- `models/labels.json` — mapping of numeric labels to names

## 5) Run Realtime Recognition
```bash
python src/recognize.py --threshold 60
```
- A **lower** LBPH distance is **better** (more confident). The default threshold (60) is a good start. If you see many false positives, try lowering it (e.g., 50). If it misses too often, try raising it (e.g., 75).

## Common Issues
- **Camera not opening**: ensure no other app is using the webcam; try `--camera 1` (or 2).
- **No faces detected**: ensure good lighting and face is frontal; adjust webcam position.
- **Poor recognition**: collect more varied images per person and retrain; tweak `--threshold`.

## Commands Overview
```bash
# Enroll a person
python src/collect_faces.py --name NAME --num 80

# Train model after collecting
python src/train.py

# Realtime recognition
python src/recognize.py --threshold 60 --camera 0
```
