import os
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "data", "data_v4.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")

def main():
    print("==================================================")
    print("LAUNCHING EXPERIMENT D: Class Imbalance Compensation (NoVest Focus)")
    print(f"Data: {DATA_YAML}")
    print("==================================================")
    
    # Base on best candidate from Exp A or Exp B
    best_exp_a = os.path.join(PROJECT_DIR, "v4_expA_yolo11n_finetune", "weights", "best.pt")
    v2_weights = os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
    start_weights = best_exp_a if os.path.exists(best_exp_a) else v2_weights
    
    model = YOLO(start_weights)
    results = model.train(
        data=DATA_YAML,
        project=PROJECT_DIR,
        name="v4_expD_class_imbalance_loss",
        exist_ok=True,
        epochs=30,
        patience=10,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        optimizer="auto",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        cls=1.8, # Elevated class loss to penalize NoVest and Helmet misclassifications heavily
        box=7.5,
        dfl=1.5,
        mosaic=0.6,
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
    print("Experiment D Training Complete!")

if __name__ == "__main__":
    main()

