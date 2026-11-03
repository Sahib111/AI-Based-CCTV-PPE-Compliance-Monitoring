import os
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "data", "data_v4.yaml")
V2_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")

def main():
    model = YOLO(V2_WEIGHTS)
    results = model.train(
        data=DATA_YAML,
        project=os.path.join(BASE_DIR, "ppe_v4_experiments", "models"),
        name="test_mem",
        exist_ok=True,
        epochs=1,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        plots=False
    )
    print("1 epoch completed successfully with workers=0!")

if __name__ == "__main__":
    main()
