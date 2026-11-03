import os
import sys
import time
import json
import csv
import torch
from ultralytics import YOLO

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "data", "data_v4.yaml")
EVAL_OUT_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "evaluation")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

def evaluate_model(weights_path, experiment_id, conf=0.001, iou=0.6, imgsz=640, device="0"):
    print(f"\n==================================================")
    print(f"EVALUATING: {experiment_id}")
    print(f"Weights: {weights_path}")
    print(f"Device: {device} (CUDA: {torch.cuda.is_available()})")
    print(f"Conf: {conf}, IoU: {iou}, Imgsz: {imgsz}")
    print(f"==================================================")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
        
    model = YOLO(weights_path)
    save_run_dir = os.path.join(EVAL_OUT_DIR, experiment_id)
    os.makedirs(save_run_dir, exist_ok=True)
    
    t0 = time.time()
    metrics = model.val(
        data=DATA_YAML,
        project=EVAL_OUT_DIR,
        name=experiment_id,
        exist_ok=True,
        device=device,
        batch=16 if torch.cuda.is_available() else 4,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=False
    )
    total_wall_time = time.time() - t0
    
    speed = metrics.speed
    p_all = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    r_all = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50_all = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95_all = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
    f1_all = (2 * p_all * r_all) / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
    
    class_names = ["Helmet", "NoHelmet", "NoVest", "Vest"]
    class_res = {}
    for i, name in enumerate(class_names):
        p = float(metrics.box.p[i]) if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
        r = float(metrics.box.r[i]) if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0
        ap50 = float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
        ap = float(metrics.box.ap[i]) if hasattr(metrics.box, "ap") and len(metrics.box.ap) > i else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        class_res[name] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP50": ap50,
            "mAP50_95": ap
        }
        
    summary = {
        "experiment_id": experiment_id,
        "weights": weights_path,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "imgsz": imgsz,
        "precision": p_all,
        "recall": r_all,
        "f1": f1_all,
        "mAP50": map50_all,
        "mAP50_95": map50_95_all,
        "latency_preprocess_ms": float(speed.get("preprocess", 0.0)),
        "latency_inference_ms": float(speed.get("inference", 0.0)),
        "latency_postprocess_ms": float(speed.get("postprocess", 0.0)),
        "latency_total_ms": float(speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)),
        "total_eval_time_s": total_wall_time,
        "class_metrics": class_res
    }
    
    # Save JSON summary
    out_json = os.path.join(save_run_dir, "metrics_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    # Save CSV per-class
    out_csv = os.path.join(save_run_dir, "class_wise_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1", "mAP50", "mAP50_95"])
        for cname, m in class_res.items():
            writer.writerow([cname, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['mAP50']:.4f}", f"{m['mAP50_95']:.4f}"])
        writer.writerow(["ALL", f"{p_all:.4f}", f"{r_all:.4f}", f"{f1_all:.4f}", f"{map50_all:.4f}", f"{map50_95_all:.4f}"])
        
    print(f"\n--- EVALUATION SUMMARY: {experiment_id} ---")
    print(f"mAP@50:    {map50_all*100:.2f}%")
    print(f"mAP@50-95: {map50_95_all*100:.2f}%")
    print(f"Precision: {p_all*100:.2f}%")
    print(f"Recall:    {r_all*100:.2f}%")
    print(f"F1 Score:  {f1_all*100:.2f}%")
    print(f"Latency:   {summary['latency_total_ms']:.2f} ms/frame ({summary['latency_inference_ms']:.2f} ms inference)")
    for cname, m in class_res.items():
        print(f"  {cname:10s} -> mAP@50: {m['mAP50']*100:.2f}% | F1: {m['f1']*100:.2f}% | R: {m['recall']*100:.2f}%")
    print("--------------------------------------------------\n")
    return summary

if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
    eid = sys.argv[2] if len(sys.argv) > 2 else "ppe_v2_baseline"
    dev = "0" if torch.cuda.is_available() else "cpu"
    evaluate_model(w, eid, conf=0.001, iou=0.6, imgsz=640, device=dev)

