"""
PPE v4 Shared Evaluation Script
================================
Evaluates any YOLO weights file against the trusted validation set.
Records all metrics and saves a metrics_summary.json alongside the val plots.

Usage:
  python ppe_v4_experiments/evaluate_v4.py <weights_path> <run_name> [conf] [iou]
"""
import os, sys, time, json
from ultralytics import YOLO

BASE_DIR  = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML = os.path.join(BASE_DIR, "ppe_v4_experiments", "datasets", "data_v4.yaml")
RESULTS   = os.path.join(BASE_DIR, "ppe_v4_experiments", "results")
CLASS_NAMES = ["Helmet", "NoHelmet", "NoVest", "Vest"]

def evaluate(weights_path, run_name, conf=0.001, iou=0.6):
    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name}")
    print(f"  Weights : {weights_path}")
    print(f"  conf={conf}, iou={iou}")
    print(f"{'='*60}")

    model = YOLO(weights_path)
    t0 = time.time()
    m = model.val(
        data=DATA_YAML,
        project=RESULTS,
        name=run_name,
        exist_ok=True,
        device="cpu",
        batch=8,
        imgsz=640,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=False,
    )
    elapsed = time.time() - t0

    p   = float(m.results_dict["metrics/precision(B)"])
    r   = float(m.results_dict["metrics/recall(B)"])
    m50 = float(m.results_dict["metrics/mAP50(B)"])
    m95 = float(m.results_dict["metrics/mAP50-95(B)"])
    f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    res = {
        "run_name":         run_name,
        "weights":          weights_path,
        "eval_conf":        conf,
        "eval_iou":         iou,
        "precision":        round(p, 4),
        "recall":           round(r, 4),
        "f1":               round(f1, 4),
        "mAP50":            round(m50, 4),
        "mAP50_95":         round(m95, 4),
        "speed_inference_ms": round(float(m.speed.get("inference", 0)), 2),
        "eval_time_s":      round(elapsed, 1),
        "class_metrics":    {},
    }

    for i, name in enumerate(CLASS_NAMES):
        cp   = float(m.box.p[i])   if len(m.box.p)   > i else 0.0
        cr   = float(m.box.r[i])   if len(m.box.r)   > i else 0.0
        ap50 = float(m.box.ap50[i]) if len(m.box.ap50) > i else 0.0
        ap   = float(m.box.ap[i])  if len(m.box.ap)  > i else 0.0
        cf1  = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        res["class_metrics"][name] = {
            "precision":  round(cp, 4),
            "recall":     round(cr, 4),
            "f1":         round(cf1, 4),
            "mAP50":      round(ap50, 4),
            "mAP50_95":   round(ap, 4),
        }

    # Print summary
    print(f"\n  Overall  | P={res['precision']:.4f} R={res['recall']:.4f} "
          f"F1={res['f1']:.4f} mAP50={res['mAP50']:.4f} mAP50-95={res['mAP50_95']:.4f}")
    for name, cm in res["class_metrics"].items():
        print(f"  {name:10s}| P={cm['precision']:.4f} R={cm['recall']:.4f} "
              f"F1={cm['f1']:.4f} mAP50={cm['mAP50']:.4f}")

    out_dir = os.path.join(RESULTS, run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "metrics_summary.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Saved: {out_json}")
    return res


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BASE_DIR, "runs", "detect", "ppe_v2", "weights", "best.pt")
    name    = sys.argv[2] if len(sys.argv) > 2 else "v4_eval_default"
    conf    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    iou     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    evaluate(weights, name, conf, iou)
