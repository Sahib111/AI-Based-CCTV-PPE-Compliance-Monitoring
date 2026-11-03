import os, sys, json
import pandas as pd
from ultralytics import YOLO

def analyze_thresholds(weights_path, run_prefix):
    base_dir = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
    data_yaml = os.path.join(base_dir, "ppe_v3_experiments", "data", "data_trusted.yaml")
    project_dir = os.path.join(base_dir, "ppe_v3_experiments", "results")
    
    model = YOLO(weights_path)
    
    conf_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    records = []
    
    for conf in conf_thresholds:
        run_name = f"{run_prefix}_conf_{int(conf*100)}"
        print(f"--- Evaluating {weights_path} with conf={conf} ---")
        metrics = model.val(
            data=data_yaml,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            device="cpu",
            batch=8,
            imgsz=640,
            conf=conf,
            iou=0.6,
            plots=False
        )
        
        p = float(metrics.results_dict["metrics/precision(B)"])
        r = float(metrics.results_dict["metrics/recall(B)"])
        map50 = float(metrics.results_dict["metrics/mAP50(B)"])
        map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        
        row = {
            "model": run_prefix,
            "conf_thresh": conf,
            "iou_thresh": 0.6,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
        }
        
        class_names = ["Helmet", "NoHelmet", "NoVest", "Vest"]
        for i, name in enumerate(class_names):
            cp = float(metrics.box.p[i]) if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
            cr = float(metrics.box.r[i]) if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0
            cmap50 = float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
            cf1 = (2 * cp * cr) / (cp + cr) if (cp + cr) > 0 else 0
            row[f"{name}_P"] = round(cp, 4)
            row[f"{name}_R"] = round(cr, 4)
            row[f"{name}_F1"] = round(cf1, 4)
            row[f"{name}_mAP50"] = round(cmap50, 4)
            
        records.append(row)
        
    df = pd.DataFrame(records)
    out_csv = os.path.join(project_dir, f"{run_prefix}_threshold_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nThreshold sweep saved to: {out_csv}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else r"ppe_v3_experiments\models\ppe_v3_finetune\weights\best.pt"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "ppe_v3_sweep"
    analyze_thresholds(weights, prefix)
