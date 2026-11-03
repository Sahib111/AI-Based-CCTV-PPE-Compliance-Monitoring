"""
PPE v4 Experiment 3: YOLO11s Architecture @ 640
=================================================
Strategy:
- YOLO11s has ~4x parameters vs YOLO11n — higher capacity ceiling
- Start from COCO pretrained yolo11s.pt and fine-tune on PPE dataset
- 25 epochs (more needed since starting from COCO, not domain-specific checkpoint)
- CPU inference ~90ms/image — acceptable for system prototype

Estimated runtime: ~4 min/epoch x 25 = ~100 min

Note: Download yolo11s.pt automatically if not present.
"""
import os, time, json, torch
from ultralytics import YOLO

BASE_DIR  = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "datasets", "data_v4.yaml")
PROJECT   = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")
RUN_NAME  = "v4_exp3_s640_25ep"

torch.set_num_threads(10)

print("[EXP3] Loading YOLO11s (COCO pretrained)...")
model = YOLO("yolo11s.pt")   # auto-downloads if not present

t0 = time.time()
results = model.train(
    data=DATA_YAML,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,

    # --- Core ---
    epochs=25,
    patience=10,
    batch=4,         # YOLO11s is 4x heavier; batch=4 for CPU
    imgsz=640,
    device="cpu",
    workers=2,

    # --- LR ---
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3.0,

    # --- Loss ---
    box=7.5,
    cls=1.5,
    dfl=1.5,

    # --- Augmentation ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=0.9,      # higher mosaic for larger model
    mixup=0.15,
    copy_paste=0.0,
    close_mosaic=5,

    # --- Output ---
    save=True,
    save_period=5,
    plots=True,
    val=True,
)
elapsed = time.time() - t0
print(f"\n[EXP3] Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"[EXP3] Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

config = {
    "experiment": RUN_NAME, "base_weights": "yolo11s.pt (COCO)", "data": DATA_YAML,
    "epochs": 25, "patience": 10, "batch": 4, "imgsz": 640,
    "lr0": 0.001, "cos_lr": True, "cls": 1.5, "elapsed_s": elapsed,
}
log_path = os.path.join(PROJECT, RUN_NAME, "train_config.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[EXP3] Config saved to: {log_path}")
