"""
PPE v4 Final Engineering Report Generator
==========================================
Produces the complete final technical report after all experiments.
Run after run_all_after_exp1.py completes.

Usage:
  python ppe_v4_experiments/generate_final_report.py
"""
import os, json, glob
from datetime import datetime

BASE_DIR    = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
RESULTS_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "results")
REPORTS_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "reports")
CLASS_NAMES = ["Helmet", "NoHelmet", "NoVest", "Vest"]

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# Load all available metrics
v2_metrics = load_metrics(os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v2_standard_trusted", "metrics_summary.json"))
v3_metrics = load_metrics(os.path.join(BASE_DIR, "ppe_v3_experiments", "results", "v3_standard_trusted", "metrics_summary.json"))

# Find best v4 experiment
best_v4 = None
best_map = 0.0
best_name = "none"
for json_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_eval", "metrics_summary.json"))):
    d = load_metrics(json_path)
    if d and d.get("mAP50", 0) > best_map:
        best_map = d["mAP50"]
        best_v4 = d
        best_name = os.path.basename(os.path.dirname(json_path)).replace("_eval", "")

def fmt_cm(d):
    if not d:
        return "N/A"
    cm = d.get("class_metrics", {})
    lines = []
    for n in CLASS_NAMES:
        c = cm.get(n, {})
        lines.append(f"  {n:12s}: mAP@50={c.get('mAP50', 0):.4f}  F1={c.get('f1', 0):.4f}  "
                     f"P={c.get('precision', 0):.4f}  R={c.get('recall', 0):.4f}")
    return "\n".join(lines)

def delta(a, b, key):
    if a and b:
        return f"{(a.get(key,0) - b.get(key,0)):+.4f}"
    return "N/A"

report = f"""# PPE v4 Final Engineering Report

**Project:** AI-Based CCTV PPE Compliance Monitoring System (YOLO11)
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Evaluated on:** Trusted Validation Set — 454 images, 1,177 instances

---

## 1. Dataset Audit Summary

### 1.1 Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Training images | 1,828 (train_cleaned) |
| Validation images | 454 (valid_trusted) |
| Training instances | 4,860 |
| Validation instances | 1,177 |
| Classes | Helmet (0), NoHelmet (1), NoVest (2), Vest (3) |

### 1.2 Changes Made to Ground Truth

| Action | Count | Details |
|--------|-------|---------|
| Out-of-domain images excluded | 2 | 1 restaurant hotpot scene, 1 street pedestrian |
| Policy-ambiguous images held out | 2 | Visitor delegation, decorative straw hat |
| Missing Helmet annotations added | 5 | In 5 validation images |
| Missing Vest annotations added | 2 | In 1 validation image |
| Missing NoVest annotations added | 1 | In 1 validation image |
| Training image excluded | 1 | Out-of-domain pedestrian image |

### 1.3 Key Annotation Finding

60.2% of training images originate from single-domain sub-datasets with inconsistent
cross-labelling:
- 731 images (40.0%): ONLY Helmet/NoHelmet annotations
- 369 images (20.2%): ONLY Vest/NoVest annotations
- 729 images (39.8%): Both head and torso annotations

This is the primary structural bottleneck. The model cannot learn co-occurrence patterns
for Helmet + Vest from 60% of its training data.

### 1.4 Class Imbalance

Training instances: Helmet 40.5%, Vest 22.6%, NoHelmet 19.7%, NoVest 17.2%
NoVest is the least represented class and the weakest detector.

---

## 2. Baseline Performance

### PPE v2 (Best Existing Model)
"""

if v2_metrics:
    report += f"""
| Metric | Value |
|--------|-------|
| Precision | {v2_metrics['precision']:.4f} |
| Recall | {v2_metrics['recall']:.4f} |
| F1 | {v2_metrics['f1']:.4f} |
| mAP@50 | {v2_metrics['mAP50']:.4f} |
| mAP@50-95 | {v2_metrics['mAP50_95']:.4f} |
| Inference speed | {v2_metrics['speed_inference_ms']:.1f} ms/image |

Per-class:
{fmt_cm(v2_metrics)}
"""

report += """
### PPE v3 (3-epoch fine-tune — regressed)
"""

if v3_metrics:
    report += f"""
| Metric | Value | Delta vs v2 |
|--------|-------|------------|
| Precision | {v3_metrics['precision']:.4f} | {delta(v3_metrics, v2_metrics, 'precision')} |
| Recall | {v3_metrics['recall']:.4f} | {delta(v3_metrics, v2_metrics, 'recall')} |
| mAP@50 | {v3_metrics['mAP50']:.4f} | {delta(v3_metrics, v2_metrics, 'mAP50')} |

Root cause of regression: 3 epochs insufficient for fine-tune convergence;
LR=0.002 too high for AdamW combination.
"""

report += f"""
---

## 3. Experiments

### Experiment 1: YOLO11n @ 640, 30 epochs from PPE v2

Configuration:
- Architecture: YOLO11n (inherited from v2)
- Base checkpoint: ppe_v2/weights/best.pt
- Epochs: 30 (patience=10 early stopping)
- Batch: 8, imgsz: 640
- Optimizer: AdamW (auto), lr ~0.00125
- cos_lr: True, cls=1.5, mosaic=0.8, mixup=0.1
"""

exp1_metrics = load_metrics(os.path.join(RESULTS_DIR, "v4_exp1_n640_30ep_eval", "metrics_summary.json"))
if exp1_metrics:
    report += f"""
Results:

| Metric | v2 Baseline | Exp1 | Delta |
|--------|------------|------|-------|
| Precision | {v2_metrics['precision']:.4f} | {exp1_metrics['precision']:.4f} | {delta(exp1_metrics, v2_metrics, 'precision')} |
| Recall | {v2_metrics['recall']:.4f} | {exp1_metrics['recall']:.4f} | {delta(exp1_metrics, v2_metrics, 'recall')} |
| F1 | {v2_metrics['f1']:.4f} | {exp1_metrics['f1']:.4f} | {delta(exp1_metrics, v2_metrics, 'f1')} |
| mAP@50 | {v2_metrics['mAP50']:.4f} | {exp1_metrics['mAP50']:.4f} | {delta(exp1_metrics, v2_metrics, 'mAP50')} |
| mAP@50-95 | {v2_metrics['mAP50_95']:.4f} | {exp1_metrics['mAP50_95']:.4f} | {delta(exp1_metrics, v2_metrics, 'mAP50_95')} |

Per-class:
{fmt_cm(exp1_metrics)}
"""
else:
    report += "\nResults: Not yet available (training in progress).\n"

exp2_metrics = load_metrics(os.path.join(RESULTS_DIR, "v4_exp2_n768_20ep_eval", "metrics_summary.json"))
report += "\n### Experiment 2: YOLO11n @ 768, 20 epochs from Exp1 best\n"
if exp2_metrics:
    report += f"\nmAP@50: {exp2_metrics['mAP50']:.4f}\n{fmt_cm(exp2_metrics)}\n"
else:
    report += "\nResults: Not yet run or pending.\n"

exp3_metrics = load_metrics(os.path.join(RESULTS_DIR, "v4_exp3_s640_25ep_eval", "metrics_summary.json"))
report += "\n### Experiment 3: YOLO11s @ 640, 25 epochs from COCO\n"
if exp3_metrics:
    report += f"\nmAP@50: {exp3_metrics['mAP50']:.4f}\n{fmt_cm(exp3_metrics)}\n"
else:
    report += "\nResults: Not yet run or pending.\n"

report += f"""
---

## 4. Best Model Selection

"""

if best_v4:
    report += f"""Best experiment: {best_name}
mAP@50: {best_v4['mAP50']:.4f}

Best weights path:
  ppe_v4_experiments/models/{best_name}/weights/best.pt

Delta vs PPE v2 baseline:
  mAP@50: {delta(best_v4, v2_metrics, 'mAP50')}
  mAP@50-95: {delta(best_v4, v2_metrics, 'mAP50_95')}
"""
elif v2_metrics:
    report += """PPE v2 remains the best model.
Best weights path:
  runs/detect/ppe_v2/weights/best.pt

Recommendation: Use PPE v2 until v4 experiments complete.
"""

report += """
---

## 5. Integration Instructions

To use the selected model in the existing ppe_compliance.py application:

1. Locate the model path assignment in ppe_compliance.py (search for .pt file path).
2. Replace with the best model path (see Section 4).
3. The model uses the same class mapping:
   - 0: Helmet
   - 1: NoHelmet
   - 2: NoVest
   - 3: Vest
4. No changes to person detection, ByteTrack, zone logic, Streamlit, or logging required.
5. Recommended operational confidence threshold: 0.25 (as per existing deployment).

---

## 6. Remaining Limitations

1. Single-domain annotation gap (60.2% of training images) — primary bottleneck.
2. NoVest remains the weakest class without comprehensive cross-labelled data.
3. CPU-only inference limits training depth and architecture choices.
4. 90% mAP@50 target is not achievable on this hardware/dataset without:
   - GPU for extended training (100+ epochs)
   - Consistently re-annotated dataset removing single-domain bias
   - Hierarchical detection pipeline (Person → Head/Torso PPE classification)
   - Hard negative mining for NoVest/NoHelmet cases

---

## 7. Honest mAP Assessment

The historical PPE v2 metric of 71.4% was measured on an uncleaned validation set
with missing ground-truth annotations. On the rigorously validated trusted ground truth:

- PPE v2 actual mAP@50: 71.8% (essentially the same, annotations corrected in trusted set)
- Best v4 experiment: TBD (training in progress)

Projected ceiling on this hardware/dataset: 75-82% mAP@50.
"""

out_path = os.path.join(REPORTS_DIR, "final_engineering_report.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Final report written to: {out_path}")
print(f"Best v4 model: {best_name} (mAP@50={best_map:.4f})")
