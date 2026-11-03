import os
import glob
import json
import csv
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
CALIB_YAML = os.path.join(DATA_DIR, "data_calib.yaml")
OUT_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "evaluation", "threshold_calibration")
os.makedirs(OUT_DIR, exist_ok=True)

def setup_calibration_split():
    # Pick a deterministic 150-image subset from train for threshold calibration
    # to avoid contaminating the 454 trusted validation set
    all_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.*")))
    np.random.seed(42)
    calib_imgs = sorted(list(np.random.choice(all_imgs, size=150, replace=False)))
    
    calib_txt_path = os.path.join(DATA_DIR, "calib_images.txt")
    with open(calib_txt_path, "w", encoding="utf-8") as f:
        for p in calib_imgs:
            f.write(p.replace("\\", "/") + "\n")
            
    calib_yaml_content = f"""path: {BASE_DIR.replace('\\', '/')}
train: ppe_v4_experiments/data/train/images
val: ppe_v4_experiments/data/calib_images.txt
test:

nc: 4
names: ['Helmet', 'NoHelmet', 'NoVest', 'Vest']
"""
    with open(CALIB_YAML, "w", encoding="utf-8") as f:
        f.write(calib_yaml_content)
    return CALIB_YAML

def calibrate_model(weights_path, model_name):
    print(f"\n--- Starting Calibration Grid Search for {model_name} ---")
    setup_calibration_split()
    model = YOLO(weights_path)
    
    conf_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    iou_range = [0.45, 0.50, 0.60, 0.70]
    
    results = []
    best_f1 = 0.0
    best_cfg = None
    
    for conf in conf_range:
        for iou in iou_range:
            metrics = model.val(
                data=CALIB_YAML,
                batch=8,
                imgsz=640,
                conf=conf,
                iou=iou,
                device=0 if torch.cuda.is_available() else "cpu",
                plots=False,
                verbose=False
            )
            p = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
            r = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
            map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            
            row = {
                "conf": conf,
                "iou": iou,
                "precision": p,
                "recall": r,
                "f1": f1,
                "mAP50": map50,
                "mAP50_95": map50_95
            }
            results.append(row)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = row
                
    out_csv = os.path.join(OUT_DIR, f"{model_name}_threshold_sweep.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conf", "iou", "precision", "recall", "f1", "mAP50", "mAP50_95"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Optimal Calibration Thresholds for {model_name}:")
    print(f"  Confidence: {best_cfg['conf']}")
    print(f"  IoU NMS:    {best_cfg['iou']}")
    print(f"  Calib F1:   {best_cfg['f1']*100:.2f}% (P: {best_cfg['precision']*100:.2f}%, R: {best_cfg['recall']*100:.2f}%)")
    print(f"Saved grid sweep to: {out_csv}")
    return best_cfg

if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
    m = sys.argv[2] if len(sys.argv) > 2 else "baseline_v2"
    calibrate_model(w, m)

