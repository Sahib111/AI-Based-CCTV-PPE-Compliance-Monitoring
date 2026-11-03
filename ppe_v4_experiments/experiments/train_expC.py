import os
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "data", "data_v4.yaml")
V2_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
PROJECT_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")

def main():
    print("==================================================")
    print("LAUNCHING EXPERIMENT C: High-Resolution (768px) & Conservative Industrial Augmentation")
    print(f"Data: {DATA_YAML}")
    print("==================================================")
    
    # Start from V2 weights or best candidate
    best_weights = os.path.join(PROJECT_DIR, "v4_expA_yolo11n_finetune", "weights", "best.pt")
    start_weights = best_weights if os.path.exists(best_weights) else V2_WEIGHTS
    
    model = YOLO(start_weights)
    results = model.train(
        data=DATA_YAML,
        project=PROJECT_DIR,
        name="v4_expC_hires768_aug",
        exist_ok=True,
        epochs=30,
        patience=10,
        batch=4, # batch 4 for 768px to stay well within 4GB VRAM
        imgsz=768,
        device=0,
        workers=0,
        optimizer="auto",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        cls=1.3,
        box=7.5,
        dfl=1.5,
        mosaic=0.5, # conservative mosaic to prevent unrealistic scale warping
        scale=0.4,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=5,
        save=True,
        save_period=5,
        plots=True,
        verbose=True
    )
    print("Experiment C Training Complete!")

if __name__ == "__main__":
    main()

