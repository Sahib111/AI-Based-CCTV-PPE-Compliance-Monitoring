import os
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "data", "data_v4.yaml")
V2_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
PROJECT_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")

def main():
    print("==================================================")
    print("LAUNCHING EXPERIMENT A: Fine-tune YOLO11n (35 Epochs)")
    print(f"Data: {DATA_YAML}")
    print(f"Base Weights: {V2_WEIGHTS}")
    print("==================================================")
    
    model = YOLO(V2_WEIGHTS)
    results = model.train(
        data=DATA_YAML,
        project=PROJECT_DIR,
        name="v4_expA_yolo11n_finetune",
        exist_ok=True,
        epochs=35,
        patience=10,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        optimizer="auto",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        cls=1.2,
        box=7.5,
        dfl=1.5,
        mosaic=0.7,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=5,
        save=True,
        save_period=5,
        plots=True,
        verbose=True
    )
    print("Experiment A Training Complete!")

if __name__ == "__main__":
    main()
