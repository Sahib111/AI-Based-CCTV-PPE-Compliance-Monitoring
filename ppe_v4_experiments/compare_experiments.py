"""
PPE v4 Experiment Tracker
==========================
Reads all metrics_summary.json files from ppe_v4_experiments/results/
and from ppe_v3_experiments/results/ (baselines) and builds a unified
experiment comparison table.

Usage:
  python ppe_v4_experiments/compare_experiments.py
"""
import os, json, glob
import pandas as pd

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
CLASS_NAMES = ["Helmet", "NoHelmet", "NoVest", "Vest"]


def load_metrics(json_path, label_override=None):
    with open(json_path) as f:
        d = json.load(f)
    cm = d.get("class_metrics", {})
    row = {
        "Experiment":      label_override or d.get("run_name", os.path.dirname(json_path)),
        "Weights":         d.get("weights", ""),
        "Eval_conf":       d.get("eval_conf", "—"),
        "Precision":       d.get("precision", 0),
        "Recall":          d.get("recall", 0),
        "F1":              d.get("f1", 0),
        "mAP50":           d.get("mAP50", 0),
        "mAP50_95":        d.get("mAP50_95", 0),
        "Inference_ms":    d.get("speed_inference_ms", d.get("speed_inference_ms", 0)),
    }
    for cls in CLASS_NAMES:
        row[f"{cls}_mAP50"] = cm.get(cls, {}).get("mAP50", 0)
        row[f"{cls}_F1"]    = cm.get(cls, {}).get("f1", 0)
    return row


rows = []

# --- Baselines from v3 experiments ---
baselines = [
    (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v2_standard_trusted",  "metrics_summary.json"), "PPE v2 Baseline (std conf=0.001)"),
    (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v3_standard_trusted",  "metrics_summary.json"), "PPE v3 Finetuned 3ep (std conf=0.001)"),
    (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v2_baseline_trusted",  "metrics_summary.json"), "PPE v2 Baseline (op conf=0.25)"),
    (os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v3_finetune_trusted",  "metrics_summary.json"), "PPE v3 Finetuned 3ep (op conf=0.25)"),
]
for path, label in baselines:
    if os.path.exists(path):
        rows.append(load_metrics(path, label))

# --- v4 experiment results ---
v4_results_dir = os.path.join(BASE_DIR, "ppe_v4_experiments", "results")
for json_path in sorted(glob.glob(os.path.join(v4_results_dir, "*", "metrics_summary.json"))):
    run_name = os.path.basename(os.path.dirname(json_path))
    rows.append(load_metrics(json_path, f"[v4] {run_name}"))

if not rows:
    print("No results found yet.")
else:
    df = pd.DataFrame(rows)
    df = df.sort_values("mAP50", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based ranking

    # Display
    cols = ["Experiment", "Precision", "Recall", "F1", "mAP50", "mAP50_95",
            "Helmet_mAP50", "NoHelmet_mAP50", "NoVest_mAP50", "Vest_mAP50", "Inference_ms"]
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    print("\n" + "="*120)
    print("PPE DETECTION EXPERIMENT COMPARISON (sorted by mAP50, evaluated on Trusted Valid 454 images)")
    print("="*120)
    print(df[cols].to_string())

    out = os.path.join(BASE_DIR, "ppe_v4_experiments", "experiment_comparison.csv")
    df.to_csv(out, index=True)
    print(f"\nSaved to: {out}")
