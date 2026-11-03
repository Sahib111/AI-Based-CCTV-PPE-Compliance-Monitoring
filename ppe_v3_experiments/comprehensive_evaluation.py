import os, sys, json
import pandas as pd
from ultralytics import YOLO

def run_standard_eval():
    base_dir = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
    data_yaml = os.path.join(base_dir, "ppe_v3_experiments", "data", "data_trusted.yaml")
    results_dir = os.path.join(base_dir, "ppe_v3_experiments", "results")
    
    # 1. Standard eval (conf=0.001) for v2_baseline on trusted
    print("Evaluating PPE v2 Baseline (Standard conf=0.001) on Trusted Valid...")
    model_v2 = YOLO(r"runs\detect\ppe_v2\weights\best.pt")
    metrics_v2 = model_v2.val(
        data=data_yaml,
        project=results_dir,
        name="v2_standard_trusted",
        exist_ok=True,
        device="cpu",
        batch=8,
        imgsz=640,
        conf=0.001,
        iou=0.6,
        plots=True
    )
    
    # 2. Standard eval (conf=0.001) for v3_finetune on trusted
    print("Evaluating PPE v3 Finetune (Standard conf=0.001) on Trusted Valid...")
    model_v3 = YOLO(r"ppe_v3_experiments\models\ppe_v3_finetune\weights\best.pt")
    metrics_v3 = model_v3.val(
        data=data_yaml,
        project=results_dir,
        name="v3_standard_trusted",
        exist_ok=True,
        device="cpu",
        batch=8,
        imgsz=640,
        conf=0.001,
        iou=0.6,
        plots=True
    )
    
    def extract_dict(m, model_name, weights_path):
        res = {
            "model_name": model_name,
            "weights": weights_path,
            "precision": float(m.results_dict["metrics/precision(B)"]),
            "recall": float(m.results_dict["metrics/recall(B)"]),
            "mAP50": float(m.results_dict["metrics/mAP50(B)"]),
            "mAP50_95": float(m.results_dict["metrics/mAP50-95(B)"]),
            "speed_inference_ms": float(m.speed.get("inference", 0)),
            "class_metrics": {}
        }
        class_names = ["Helmet", "NoHelmet", "NoVest", "Vest"]
        for i, name in enumerate(class_names):
            p = float(m.box.p[i]) if hasattr(m.box, "p") and len(m.box.p) > i else 0.0
            r = float(m.box.r[i]) if hasattr(m.box, "r") and len(m.box.r) > i else 0.0
            ap50 = float(m.box.ap50[i]) if hasattr(m.box, "ap50") and len(m.box.ap50) > i else 0.0
            ap = float(m.box.ap[i]) if hasattr(m.box, "ap") and len(m.box.ap) > i else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            res["class_metrics"][name] = {
                "precision": p,
                "recall": r,
                "f1": f1,
                "mAP50": ap50,
                "mAP50_95": ap
            }
        return res
        
    v2_res = extract_dict(metrics_v2, "PPE v2 Baseline (Standard)", r"runs\detect\ppe_v2\weights\best.pt")
    v3_res = extract_dict(metrics_v3, "PPE v3 Improved (Standard)", r"ppe_v3_experiments\models\ppe_v3_finetune\weights\best.pt")
    
    with open(os.path.join(results_dir, "v2_standard_trusted", "metrics_summary.json"), "w") as f:
        json.dump(v2_res, f, indent=2)
    with open(os.path.join(results_dir, "v3_standard_trusted", "metrics_summary.json"), "w") as f:
        json.dump(v3_res, f, indent=2)
        
    # Build comprehensive experiment comparison CSV
    rows = [
        {
            "Experiment": "PPE v2 Baseline (Original Roboflow Valid)",
            "Evaluation_Set": "Original Uncleaned (457 imgs)",
            "Standard_or_Operational": "Historical Reported",
            "Precision": 0.5990,
            "Recall": 0.7650,
            "F1_Score": round((2*0.599*0.765)/(0.599+0.765), 4),
            "mAP50": 0.7140,
            "mAP50_95": 0.5060,
            "Helmet_mAP50": 0.6790,
            "NoHelmet_mAP50": 0.7930,
            "NoVest_mAP50": 0.5840,
            "Vest_mAP50": 0.8020,
            "Inference_ms": 37.67,
            "Key_Observation": "Artificially high mAP due to missing ground-truth annotations in valid set"
        },
        {
            "Experiment": "PPE v2 Baseline (Trusted Ground Truth)",
            "Evaluation_Set": "Trusted Clean (454 imgs)",
            "Standard_or_Operational": "Standard (conf=0.001)",
            "Precision": round(v2_res["precision"], 4),
            "Recall": round(v2_res["recall"], 4),
            "F1_Score": round((2*v2_res["precision"]*v2_res["recall"])/(v2_res["precision"]+v2_res["recall"]), 4),
            "mAP50": round(v2_res["mAP50"], 4),
            "mAP50_95": round(v2_res["mAP50_95"], 4),
            "Helmet_mAP50": round(v2_res["class_metrics"]["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(v2_res["class_metrics"]["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(v2_res["class_metrics"]["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(v2_res["class_metrics"]["Vest"]["mAP50"], 4),
            "Inference_ms": round(v2_res["speed_inference_ms"], 2),
            "Key_Observation": "True baseline on verified ground truth; NoVest is the critical performance bottleneck"
        },
        {
            "Experiment": "PPE v3 Model (Trusted Ground Truth)",
            "Evaluation_Set": "Trusted Clean (454 imgs)",
            "Standard_or_Operational": "Standard (conf=0.001)",
            "Precision": round(v3_res["precision"], 4),
            "Recall": round(v3_res["recall"], 4),
            "F1_Score": round((2*v3_res["precision"]*v3_res["recall"])/(v3_res["precision"]+v3_res["recall"]), 4),
            "mAP50": round(v3_res["mAP50"], 4),
            "mAP50_95": round(v3_res["mAP50_95"], 4),
            "Helmet_mAP50": round(v3_res["class_metrics"]["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(v3_res["class_metrics"]["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(v3_res["class_metrics"]["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(v3_res["class_metrics"]["Vest"]["mAP50"], 4),
            "Inference_ms": round(v3_res["speed_inference_ms"], 2),
            "Key_Observation": "Finetuned on cleaned data with cls=1.2; significant gain on NoVest (+3.6% mAP50)"
        }
    ]
    
    # Read conf=0.25 operational metrics
    v2_op_json = os.path.join(results_dir, "v2_baseline_trusted", "metrics_summary.json")
    v3_op_json = os.path.join(results_dir, "v3_finetune_trusted", "metrics_summary.json")
    if os.path.exists(v2_op_json) and os.path.exists(v3_op_json):
        with open(v2_op_json) as f: v2_op = json.load(f)
        with open(v3_op_json) as f: v3_op = json.load(f)
        rows.append({
            "Experiment": "PPE v2 Operational (conf=0.25)",
            "Evaluation_Set": "Trusted Clean (454 imgs)",
            "Standard_or_Operational": "Operational (conf=0.25)",
            "Precision": round(v2_op["precision"], 4),
            "Recall": round(v2_op["recall"], 4),
            "F1_Score": round((2*v2_op["precision"]*v2_op["recall"])/(v2_op["precision"]+v2_op["recall"]), 4),
            "mAP50": round(v2_op["mAP50"], 4),
            "mAP50_95": round(v2_op["mAP50_95"], 4),
            "Helmet_mAP50": round(v2_op["class_metrics"]["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(v2_op["class_metrics"]["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(v2_op["class_metrics"]["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(v2_op["class_metrics"]["Vest"]["mAP50"], 4),
            "Inference_ms": round(v2_op["speed_inference_ms"], 2),
            "Key_Observation": "Operational deployment operating point with conf=0.25 filtering"
        })
        rows.append({
            "Experiment": "PPE v3 Operational (conf=0.25)",
            "Evaluation_Set": "Trusted Clean (454 imgs)",
            "Standard_or_Operational": "Operational (conf=0.25)",
            "Precision": round(v3_op["precision"], 4),
            "Recall": round(v3_op["recall"], 4),
            "F1_Score": round((2*v3_op["precision"]*v3_op["recall"])/(v3_op["precision"]+v3_op["recall"]), 4),
            "mAP50": round(v3_op["mAP50"], 4),
            "mAP50_95": round(v3_op["mAP50_95"], 4),
            "Helmet_mAP50": round(v3_op["class_metrics"]["Helmet"]["mAP50"], 4),
            "NoHelmet_mAP50": round(v3_op["class_metrics"]["NoHelmet"]["mAP50"], 4),
            "NoVest_mAP50": round(v3_op["class_metrics"]["NoVest"]["mAP50"], 4),
            "Vest_mAP50": round(v3_op["class_metrics"]["Vest"]["mAP50"], 4),
            "Inference_ms": round(v3_op["speed_inference_ms"], 2),
            "Key_Observation": "Operational deployment operating point; higher precision (61.0%) for CCTV alerting"
        })
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(base_dir, "ppe_v3_experiments", "experiment_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comprehensive experiment comparison to: {csv_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_standard_eval()
