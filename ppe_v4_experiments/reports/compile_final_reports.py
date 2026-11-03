import os
import glob
import json
import csv

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
EVAL_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "evaluation")
MODELS_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def compile_reports():
    print("=== Compiling Comparative Reports & CSV Artifacts ===")
    
    # 1. Gather all evaluated models
    eval_dirs = sorted(glob.glob(os.path.join(EVAL_DIR, "*")))
    models_data = []
    
    for ed in eval_dirs:
        summary_path = os.path.join(ed, "metrics_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                models_data.append(json.load(f))
                
    print(f"Loaded {len(models_data)} evaluation records.")
    
    # 2. Generate V2_V3_V4_COMPARISON.csv
    comp_csv_path = os.path.join(REPORTS_DIR, "V2_V3_V4_COMPARISON.csv")
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Run", "Base_Architecture", "Confidence", "IoU", "Precision", "Recall", "F1_Score", "mAP50", "mAP50_95", "Inference_Latency_ms", "Status_vs_Baseline"])
        for m in models_data:
            eid = m["experiment_id"]
            p = m["precision"] * 100
            r = m["recall"] * 100
            f1 = m["f1"] * 100
            map50 = m["mAP50"] * 100
            map50_95 = m["mAP50_95"] * 100
            lat = m["latency_total_ms"]
            
            status = "PRIMARY_BASELINE" if "v2_baseline" in eid else (
                "REGRESSED" if map50 < 71.84 else "IMPROVED"
            )
            arch = "YOLO11s" if "11s" in eid else "YOLO11n"
            writer.writerow([eid, arch, m.get("conf_threshold", 0.001), m.get("iou_threshold", 0.6), f"{p:.2f}%", f"{r:.2f}%", f"{f1:.2f}%", f"{map50:.2f}%", f"{map50_95:.2f}%", f"{lat:.2f} ms", status])
            
    # 3. Generate CLASS_WISE_PERFORMANCE.csv
    class_csv_path = os.path.join(REPORTS_DIR, "CLASS_WISE_PERFORMANCE.csv")
    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Run", "Class_Name", "mAP50", "mAP50_95", "Precision", "Recall", "F1_Score"])
        for m in models_data:
            eid = m["experiment_id"]
            cm = m.get("class_metrics", {})
            for cname in ["Helmet", "NoHelmet", "NoVest", "Vest"]:
                if cname in cm:
                    ci = cm[cname]
                    writer.writerow([
                        eid,
                        cname,
                        f"{ci['mAP50']*100:.2f}%",
                        f"{ci['mAP50_95']*100:.2f}%",
                        f"{ci['precision']*100:.2f}%",
                        f"{ci['recall']*100:.2f}%",
                        f"{ci['f1']*100:.2f}%"
                    ])
                    
    # 4. Generate TRAINING_EXPERIMENT_LOG.csv
    log_csv_path = os.path.join(REPORTS_DIR, "TRAINING_EXPERIMENT_LOG.csv")
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment_ID", "Model", "Pretrained_Weights", "Dataset", "Epochs_Trained", "Batch", "ImgSz", "Optimizer", "Class_Loss_Weight", "Key_Augmentations", "Validation_mAP50"])
        writer.writerow(["PPE_V2", "YOLO11n", "yolo11n.pt", "data.yaml (original)", "20", "4", "640", "auto", "0.5", "Default (mosaic=1.0)", "71.84%"])
        writer.writerow(["PPE_V3_finetune", "YOLO11n", "PPE_V2 best.pt", "data_v3.yaml (cleaned)", "3", "8", "640", "auto", "1.2", "mosaic=0.5, lr0=0.002", "68.60%"])
        writer.writerow(["v4_expA_yolo11n", "YOLO11n", "PPE_V2 best.pt", "data_v4.yaml (leak-free)", "35", "8", "640", "AdamW", "1.2", "mosaic=0.7, scale=0.5, cos_lr=True", "70.44%"])
        
    print(f"Reports successfully generated in {REPORTS_DIR}")

if __name__ == "__main__":
    compile_reports()

