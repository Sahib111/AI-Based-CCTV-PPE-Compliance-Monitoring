"""
PPE v4 Experiment 1: Extended Fine-Tuning from PPE v2 Best Checkpoint
======================================================================
Strategy:
- Start from PPE v2 best.pt (the strongest known checkpoint)
- Train for 30 epochs with early stopping (patience=10)
- Use cosine LR schedule with warm-up
- Moderate augmentation tuned for small PPE targets
- Slightly elevated cls loss to help NoVest without hurting Helmet
- CPU-optimised batch=8, imgsz=640
- Threads set to 10 to saturate i7-12650H

Estimated runtime on i7-12650H CPU:
  ~2 min/epoch * 30 epochs = ~60 min max (early stop likely fires earlier)
"""
import os, time, json, torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
V2_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
DATA_YAML  = os.path.join(BASE_DIR, "ppe_v4_experiments", "datasets", "data_v4.yaml")
PROJECT    = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")
RUN_NAME   = "v4_exp1_n640_30ep"

torch.set_num_threads(10)

print(f"[EXP1] Loading PPE v2 checkpoint: {V2_WEIGHTS}")
model = YOLO(V2_WEIGHTS)

t0 = time.time()
results = model.train(
    data=DATA_YAML,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,

    # --- Core ---
    epochs=30,
    patience=10,          # Early stopping: stop if no mAP50 improvement for 10 epochs
    batch=8,
    imgsz=640,
    device="cpu",
    workers=2,

    # --- LR schedule ---
    lr0=0.001,            # conservative LR for fine-tuning (v2 trained with default 0.01)
    lrf=0.01,             # final LR = lr0 * lrf = 1e-5
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    cos_lr=True,          # cosine LR annealing

    # --- Loss weights ---
    box=7.5,
    cls=1.5,              # boosted from default 0.5 to give NoVest/Helmet more signal
    dfl=1.5,

    # --- Augmentation (tuned for small PPE objects) ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,          # no rotation - hardhat orientation matters
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,           # horizontal flip is safe for PPE
    mosaic=0.8,           # mosaic helps with small-object co-occurrence
    mixup=0.1,
    copy_paste=0.0,
    close_mosaic=5,       # disable mosaic for last 5 epochs for stability

    # --- Output ---
    save=True,
    save_period=5,        # save checkpoint every 5 epochs
    plots=True,
    val=True,
)
elapsed = time.time() - t0
print(f"\n[EXP1] Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"[EXP1] Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

# Save config log
config = {
    "experiment": RUN_NAME,
    "base_weights": V2_WEIGHTS,
    "data": DATA_YAML,
    "epochs": 30,
    "patience": 10,
    "batch": 8,
    "imgsz": 640,
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "cls": 1.5,
    "box": 7.5,
    "mosaic": 0.8,
    "mixup": 0.1,
    "elapsed_s": elapsed,
}
log_path = os.path.join(PROJECT, RUN_NAME, "train_config.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[EXP1] Config saved to: {log_path}")
