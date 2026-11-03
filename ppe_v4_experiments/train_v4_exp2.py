"""
PPE v4 Experiment 2: YOLO11n @ 768 Resolution from Exp1 Best Checkpoint
=========================================================================
Strategy:
- Start from the best checkpoint produced by Experiment 1
- Train at 768x768 to improve small-object detection of distant hardhats/vests
- Fewer epochs (20) since we're fine-tuning a well-trained checkpoint
- Early stopping patience=8
- Slightly lower LR to preserve Exp1 knowledge

Estimated runtime: ~2.5 min/epoch x 20 = ~50 min (768 is ~1.4x slower than 640)
"""
import os, time, json, torch
from ultralytics import YOLO

BASE_DIR   = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
EXP1_BEST  = os.path.join(BASE_DIR, "ppe_v4_experiments", "models", "v4_exp1_n640_30ep", "weights", "best.pt")
DATA_YAML  = os.path.join(BASE_DIR, "ppe_v4_experiments", "datasets", "data_v4.yaml")
PROJECT    = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")
RUN_NAME   = "v4_exp2_n768_20ep"

if not os.path.exists(EXP1_BEST):
    raise FileNotFoundError(f"Exp1 best.pt not found: {EXP1_BEST}\nRun Experiment 1 first.")

torch.set_num_threads(10)

print(f"[EXP2] Loading Exp1 best checkpoint: {EXP1_BEST}")
model = YOLO(EXP1_BEST)

t0 = time.time()
results = model.train(
    data=DATA_YAML,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,

    # --- Core ---
    epochs=20,
    patience=8,
    batch=4,         # batch=4 for 768 to stay within CPU memory
    imgsz=768,       # higher resolution
    device="cpu",
    workers=2,

    # --- LR (lower than Exp1 since starting from a trained checkpoint) ---
    lr0=0.0005,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=2.0,

    # --- Loss ---
    box=7.5,
    cls=1.5,
    dfl=1.5,

    # --- Augmentation (same as Exp1) ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.0,
    close_mosaic=4,

    # --- Output ---
    save=True,
    save_period=5,
    plots=True,
    val=True,
)
elapsed = time.time() - t0
print(f"\n[EXP2] Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"[EXP2] Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

config = {
    "experiment": RUN_NAME, "base_weights": EXP1_BEST, "data": DATA_YAML,
    "epochs": 20, "patience": 8, "batch": 4, "imgsz": 768,
    "lr0": 0.0005, "cos_lr": True, "cls": 1.5, "elapsed_s": elapsed,
}
log_path = os.path.join(PROJECT, RUN_NAME, "train_config.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[EXP2] Config saved to: {log_path}")
