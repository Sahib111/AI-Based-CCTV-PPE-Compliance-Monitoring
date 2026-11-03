"""
PPE v4 Post-Training Pipeline
==============================
Run this after Experiment 1 completes.
1. Evaluates best.pt against trusted validation set
2. Runs confidence threshold sweep (0.20 to 0.45)
3. Updates experiment comparison table
4. Saves all results under ppe_v4_experiments/

Usage:
  python ppe_v4_experiments/post_train_pipeline.py <exp_dir_name>

Example:
  python ppe_v4_experiments/post_train_pipeline.py v4_exp1_n640_30ep
"""
import os, sys, json, time, glob
import pandas as pd
from ultralytics import YOLO

BASE_DIR    = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
DATA_YAML   = os.path.join(BASE_DIR, "ppe_v4_experiments", "datasets", "data_v4.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "results")
CLASS_NAMES = ["Helmet", "NoHelmet", "NoVest", "Vest"]


def full_eval(weights_path, run_name, conf=0.001, iou=0.6, plots=True):
    """Evaluate weights and save metrics JSON."""
    print(f"\nEvaluating: {run_name} (conf={conf})")
    model = YOLO(weights_path)
    t0 = time.time()
    m = model.val(
        data=DATA_YAML,
        project=RESULTS_DIR,
        name=run_name,
        exist_ok=True,
        device="cpu",
        batch=8,
        imgsz=640,
        conf=conf,
        iou=iou,
        plots=plots,
    )
    elapsed = time.time() - t0

    p   = float(m.results_dict["metrics/precision(B)"])
    r   = float(m.results_dict["metrics/recall(B)"])
    m50 = float(m.results_dict["metrics/mAP50(B)"])
    m95 = float(m.results_dict["metrics/mAP50-95(B)"])
    f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    res = {
        "run_name": run_name, "weights": weights_path,
        "eval_conf": conf, "eval_iou": iou,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(f1, 4), "mAP50": round(m50, 4), "mAP50_95": round(m95, 4),
        "speed_inference_ms": round(float(m.speed.get("inference", 0)), 2),
        "eval_time_s": round(elapsed, 1),
        "class_metrics": {},
    }
    for i, name in enumerate(CLASS_NAMES):
        cp   = float(m.box.p[i])    if len(m.box.p)    > i else 0.0
        cr   = float(m.box.r[i])    if len(m.box.r)    > i else 0.0
        ap50 = float(m.box.ap50[i]) if len(m.box.ap50) > i else 0.0
        ap   = float(m.box.ap[i])   if len(m.box.ap)   > i else 0.0
        cf1  = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        res["class_metrics"][name] = {
            "precision": round(cp, 4), "recall": round(cr, 4),
            "f1": round(cf1, 4), "mAP50": round(ap50, 4), "mAP50_95": round(ap, 4),
        }

    out_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"  P={p:.4f} R={r:.4f} F1={f1:.4f} mAP50={m50:.4f} mAP50-95={m95:.4f}")
    for nm, cm in res["class_metrics"].items():
        print(f"  {nm:10s}: mAP50={cm['mAP50']:.4f} F1={cm['f1']:.4f}")
    return res


def conf_sweep(weights_path, run_prefix):
    """Sweep confidence thresholds and find the best operational point."""
    print(f"\nConfidence sweep for: {run_prefix}")
    model = YOLO(weights_path)
    records = []
    for conf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        m = model.val(
            data=DATA_YAML,
            project=RESULTS_DIR,
            name=f"{run_prefix}_conf{int(conf*100)}",
            exist_ok=True,
            device="cpu", batch=8, imgsz=640,
            conf=conf, iou=0.6, plots=False,
        )
        p   = float(m.results_dict["metrics/precision(B)"])
        r   = float(m.results_dict["metrics/recall(B)"])
        m50 = float(m.results_dict["metrics/mAP50(B)"])
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        row = {"conf": conf, "precision": round(p,4), "recall": round(r,4),
               "f1": round(f1,4), "mAP50": round(m50,4)}
        for i, nm in enumerate(CLASS_NAMES):
            row[f"{nm}_F1"] = round(
                2*float(m.box.p[i])*float(m.box.r[i])/(float(m.box.p[i])+float(m.box.r[i])+1e-9), 4)
        records.append(row)
        print(f"  conf={conf:.2f} | P={p:.3f} R={r:.3f} F1={f1:.3f} mAP50={m50:.3f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, f"{run_prefix}_threshold_sweep.csv")
    df.to_csv(csv_path, index=False)
    best = df.loc[df["mAP50"].idxmax()]
    print(f"\n  Best operational conf: {best['conf']} (mAP50={best['mAP50']:.4f})")
    return df


def update_comparison():
    """Rebuild unified comparison CSV."""
    rows = []
    # Baselines
    baselines = [
        (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v2_standard_trusted", "metrics_summary.json"), "PPE v2 Baseline (conf=0.001)"),
        (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v3_standard_trusted", "metrics_summary.json"), "PPE v3 3ep (conf=0.001)"),
    ]
    for path, label in baselines:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        cm = d.get("class_metrics", {})
        rows.append({
            "Experiment": label,
            "mAP50": d.get("mAP50", 0), "mAP50_95": d.get("mAP50_95", 0),
            "Precision": d.get("precision", 0), "Recall": d.get("recall", 0), "F1": d.get("f1", 0),
            **{f"{n}_mAP50": cm.get(n, {}).get("mAP50", 0) for n in CLASS_NAMES},
            "Inference_ms": d.get("speed_inference_ms", 0),
        })
    # v4 results (standard conf=0.001 runs only)
    for json_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "metrics_summary.json"))):
        run_name = os.path.basename(os.path.dirname(json_path))
        if "_conf" in run_name:
            continue  # skip sweep sub-runs
        with open(json_path) as f:
            d = json.load(f)
        cm = d.get("class_metrics", {})
        rows.append({
            "Experiment": f"[v4] {run_name}",
            "mAP50": d.get("mAP50", 0), "mAP50_95": d.get("mAP50_95", 0),
            "Precision": d.get("precision", 0), "Recall": d.get("recall", 0), "F1": d.get("f1", 0),
            **{f"{n}_mAP50": cm.get(n, {}).get("mAP50", 0) for n in CLASS_NAMES},
            "Inference_ms": d.get("speed_inference_ms", 0),
        })

    df = pd.DataFrame(rows).sort_values("mAP50", ascending=False).reset_index(drop=True)
    df.index += 1
    out = os.path.join(BASE_DIR, "ppe_v4_experiments", "experiment_comparison.csv")
    df.to_csv(out, index=True)
    print(f"\n{'='*100}")
    print("EXPERIMENT COMPARISON (ranked by mAP50, trusted val 454 images, conf=0.001)")
    print(f"{'='*100}")
    cols = ["Experiment", "Precision", "Recall", "F1", "mAP50", "mAP50_95",
            "Helmet_mAP50", "NoHelmet_mAP50", "NoVest_mAP50", "Vest_mAP50"]
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.width", 200)
    print(df[cols].to_string())
    print(f"\nSaved: {out}")
    return df


if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "v4_exp1_n640_30ep"
    weights = os.path.join(BASE_DIR, "ppe_v4_experiments", "models", exp_name, "weights", "best.pt")

    if not os.path.exists(weights):
        print(f"ERROR: Weights not found at {weights}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"POST-TRAIN PIPELINE: {exp_name}")
    print(f"{'='*60}")

    # 1. Standard evaluation
    full_eval(weights, f"{exp_name}_eval", conf=0.001)

    # 2. Confidence sweep
    conf_sweep(weights, exp_name)

    # 3. Rebuild comparison table
    update_comparison()

    print(f"\nPost-train pipeline complete for {exp_name}")
