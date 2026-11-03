import os, sys, time, json
from ultralytics import YOLO

def evaluate(weights_path, run_name, conf=0.25, iou=0.6):
    base_dir = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
    data_yaml = os.path.join(base_dir, "ppe_v3_experiments", "data", "data_trusted.yaml")
    project_dir = os.path.join(base_dir, "ppe_v3_experiments", "results")
    
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    print(f"Running evaluation: {run_name} (conf={conf}, iou={iou})")
    t0 = time.time()
    metrics = model.val(
        data=data_yaml,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        device="cpu",
        batch=8,
        imgsz=640,
        conf=conf,
        iou=iou,
        plots=True
    )
    total_time = time.time() - t0
    
    speed = metrics.speed # inference, preprocess, loss, postprocess (ms)
    
    res = {
        "run_name": run_name,
        "weights": weights_path,
        "precision": float(metrics.results_dict["metrics/precision(B)"]),
        "recall": float(metrics.results_dict["metrics/recall(B)"]),
        "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
        "mAP50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
        "speed_preprocess_ms": float(speed.get("preprocess", 0)),
        "speed_inference_ms": float(speed.get("inference", 0)),
        "speed_loss_ms": float(speed.get("loss", 0)),
        "speed_postprocess_ms": float(speed.get("postprocess", 0)),
        "total_eval_time_s": total_time,
        "class_metrics": {}
    }
    
    class_names = ["Helmet", "NoHelmet", "NoVest", "Vest"]
    for i, name in enumerate(class_names):
        p = float(metrics.box.p[i]) if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
        r = float(metrics.box.r[i]) if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0
        ap50 = float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
        ap = float(metrics.box.ap[i]) if hasattr(metrics.box, "ap") and len(metrics.box.ap) > i else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        res["class_metrics"][name] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP50": ap50,
            "mAP50_95": ap
        }
        
    print(f"\n==========================================")
    print(f"Results for: {run_name}")
    print(f"==========================================")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall:    {res['recall']:.4f}")
    print(f"mAP@50:    {res['mAP50']:.4f}")
    print(f"mAP@50-95: {res['mAP50_95']:.4f}")
    print(f"Inference Time per image: {res['speed_inference_ms']:.2f} ms")
    print(f"Per-Class Metrics:")
    for name, m in res["class_metrics"].items():
        print(f"  {name:10s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | mAP@50: {m['mAP50']:.4f} | mAP@50-95: {m['mAP50_95']:.4f}")
    print(f"==========================================\n")
    
    out_json = os.path.join(project_dir, run_name, "metrics_summary.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Metrics saved to: {out_json}")
    return res

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else r"runs\detect\ppe_v2\weights\best.pt"
    name = sys.argv[2] if len(sys.argv) > 2 else "v2_baseline_trusted"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    iou = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    evaluate(weights, name, conf, iou)
