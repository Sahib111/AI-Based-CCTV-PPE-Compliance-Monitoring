import os, sys, json
import pandas as pd

def build_comparison():
    base_dir = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
    results_dir = os.path.join(base_dir, "ppe_v3_experiments", "results")
    
    # 1. PPE v2 on original uncleaned validation (historical reported)
    row_v2_orig = {
        "Experiment": "PPE v2 Baseline (Historical Reported)",
        "Evaluation Dataset": "Original Uncleaned Validation (457 images)",
        "Precision": 0.5990,
        "Recall": 0.7650,
        "F1_Score": round((2 * 0.599 * 0.765) / (0.599 + 0.765), 4),
        "mAP50": 0.7140,
        "mAP50_95": 0.5060,
        "Helmet_mAP50": 0.6790,
        "NoHelmet_mAP50": 0.7930,
        "NoVest_mAP50": 0.5840,
        "Vest_mAP50": 0.8020,
        "Inference_Speed_ms": 37.67,
        "Notes": "Artificially high due to missed ground-truth boxes in uncleaned valid set"
    }
    
    # 2. PPE v2 on Trusted Validation Ground Truth
    v2_json_path = os.path.join(results_dir, "v2_baseline_trusted", "metrics_summary.json")
    if os.path.exists(v2_json_path):
        with open(v2_json_path, "r") as f:
            v2_data = json.load(f)
        p = v2_data["precision"]
        r = v2_data["recall"]
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        cm = v2_data["class_metrics"]
        row_v2_trusted = {
            "Experiment": "PPE v2 Baseline (Measured on Trusted Ground Truth)",
            "Evaluation Dataset": "Trusted Clean Validation (454 images)",
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1_Score": round(f1, 4),
            "mAP50": round(v2_data["mAP50"], 4),
            "mAP50_95": round(v2_data["mAP50_95"], 4),
            "Helmet_mAP50": round(cm["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(cm["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(cm["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(cm["Vest"]["mAP50"], 4),
            "Inference_Speed_ms": round(v2_data["speed_inference_ms"], 2),
            "Notes": "Rigorous baseline; out-of-domain removed & missing annotations corrected"
        }
    else:
        row_v2_trusted = {}

    # 3. PPE v3 Finetuned on Cleaned Train set
    v3_json_path = os.path.join(results_dir, "v3_finetune_trusted", "metrics_summary.json")
    rows = [row_v2_orig, row_v2_trusted]
    
    if os.path.exists(v3_json_path):
        with open(v3_json_path, "r") as f:
            v3_data = json.load(f)
        p = v3_data["precision"]
        r = v3_data["recall"]
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        cm = v3_data["class_metrics"]
        row_v3 = {
            "Experiment": "PPE v3 Improved Model (Measured on Trusted Ground Truth)",
            "Evaluation Dataset": "Trusted Clean Validation (454 images)",
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1_Score": round(f1, 4),
            "mAP50": round(v3_data["mAP50"], 4),
            "mAP50_95": round(v3_data["mAP50_95"], 4),
            "Helmet_mAP50": round(cm["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(cm["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(cm["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(cm["Vest"]["mAP50"], 4),
            "Inference_Speed_ms": round(v3_data["speed_inference_ms"], 2),
            "Notes": "Transfer learning from v2 best.pt; cleaned training annotations; class weighting"
        }
        rows.append(row_v3)
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(base_dir, "ppe_v3_experiments", "experiment_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved experiment comparison to: {out_csv}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    build_comparison()
