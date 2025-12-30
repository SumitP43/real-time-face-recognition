#!/usr/bin/env python
import cv2
import os
import json
from pathlib import Path

def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    label_map = {}   # name -> id
    id_map = {}      # id -> name
    current_id = 0

    for person_dir in sorted(Path(dataset_dir).glob("*")):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = current_id
            id_map[current_id] = name
            current_id += 1

        for img_path in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label_map[name])

    return images, labels, id_map

def main():
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "dataset"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    images, labels, id_map = load_images_and_labels(dataset_dir)
    if len(images) == 0:
        raise RuntimeError("No images found in dataset/. Capture faces first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=12, grid_x=8, grid_y=8)
    recognizer.train(images, cv2.UMat(labels))

    model_path = models_dir / "model.yml"
    recognizer.save(str(model_path))

    with open(models_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Trained on {len(images)} images across {len(id_map)} person(s).")
    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved labels to: {models_dir / 'labels.json'}")

if __name__ == "__main__":
    main()
