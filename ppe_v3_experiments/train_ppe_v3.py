import os, sys, time, json
import torch
from ultralytics import YOLO

def train_v3():
    # Set PyTorch thread count for optimal CPU performance on i7-12650H
    torch.set_num_threads(8)
    
    base_dir = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
    v2_weights = os.path.join(base_dir, "runs", "detect", "ppe_v2", "weights", "best.pt")
    data_yaml = os.path.join(base_dir, "ppe_v3_experiments", "data", "data_v3.yaml")
    project_dir = os.path.join(base_dir, "ppe_v3_experiments", "models")
    
    print(f"Loading pretrained PPE v2 weights: {v2_weights}")
    model = YOLO(v2_weights)
    
    print("Initiating fine-tuning for PPE v3...")
    t0 = time.time()
    results = model.train(
        data=data_yaml,
        project=project_dir,
        name="ppe_v3_finetune",
        exist_ok=True,
        epochs=3,              # 3 fine-tuning epochs feasible on CPU
        batch=8,               # batch size 8 for CPU throughput
        imgsz=640,
        device="cpu",
        workers=2,
        lr0=0.002,             # lower fine-tuning learning rate
        lrf=0.01,
        warmup_epochs=0.5,
        cls=1.2,               # increased class loss weight (from 0.5) to prioritize NoVest / Helmet precision
        box=7.5,
        dfl=1.5,
        mosaic=0.5,            # gentler mosaic augmentation to avoid small-target degradation
        close_mosaic=1,
        save=True,
        plots=True,
        val=True
    )
    total_time = time.time() - t0
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} mins)")
    return results

if __name__ == "__main__":
    train_v3()
