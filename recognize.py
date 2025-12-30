#!/usr/bin/env python
import cv2
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    model_path = models_dir / "model.yml"
    labels_path = models_dir / "labels.json"

    # Load model and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))

    with open(labels_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)  # id -> name (keys are strings in JSON)
    # JSON may load keys as strings; convert to int keys for safety
    id_map = {int(k): v for k, v in id_map.items()}

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            label_id, confidence = recognizer.predict(roi)
            name = id_map.get(label_id, "Unknown")

            # Lower confidence value means a better match with LBPH.
            text = f"{name} ({confidence:.1f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Recognition - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
