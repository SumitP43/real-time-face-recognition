#!/usr/bin/env python
import cv2
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Capture face images from webcam and save to dataset/<name>")
    parser.add_argument("--name", required=True, help="Person name (folder will be dataset/<name>)")
    parser.add_argument("--num", type=int, default=120, help="Number of face images to capture (default: 120)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "dataset" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer OpenCV's built-in cascade path to avoid local file issues
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --cam 1 or check permissions.")

    print(f"[INFO] Saving faces to: {out_dir}")
    count = 0
    while count < args.num:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, skipping...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            filename = out_dir / f"{args.name}_{count:04d}.png"
            cv2.imwrite(str(filename), roi)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{args.num}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if count >= args.num:
                break

        cv2.imshow("Capture Faces - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
