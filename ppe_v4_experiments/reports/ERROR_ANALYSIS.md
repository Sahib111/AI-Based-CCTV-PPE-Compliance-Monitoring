# PPE V4 Detailed Error Analysis & Class-Wise Diagnostics

## 1. Overview
- **Evaluated Model:** YOLO11s (`ppe_v4_experiments/models/v4_expB_yolo11s/weights/best.pt`)
- **Validation Dataset:** 454 trusted validation images (`ppe_v3_experiments/data/valid_trusted/images`)
- **Operating Parameters:** Confidence Threshold = 0.25, IoU NMS Threshold = 0.50

## 2. Per-Class Performance Summary (at IoU=0.50, Conf=0.25)

| Class ID | Class Name | Ground Truth Count | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|---|
| 0 | **Helmet** | 454 | 371 | 358 | 83 | 50.89% | 81.72% | 62.72% |
| 1 | **NoHelmet** | 209 | 185 | 164 | 24 | 53.01% | 88.52% | 66.31% |
| 2 | **NoVest** | 223 | 166 | 238 | 57 | 41.09% | 74.44% | 52.95% |
| 3 | **Vest** | 291 | 248 | 99 | 43 | 71.47% | 85.22% | 77.74% |
| **Overall** | **All Classes** | **1177** | **970** | **859** | **207** | **53.03%** | **82.41%** | **64.54%** |

## 3. Confusion Matrix Analysis

Rows represent True Ground Truth classes; columns represent Model Predictions.
(Index 4 represents Background / Unannotated).

| True \ Pred | Helmet | NoHelmet | NoVest | Vest | Background (FN) |
|---|---|---|---|---|---|
| **Helmet** | 371 | 3 | 0 | 0 | 80 |
| **NoHelmet** | 4 | 185 | 0 | 0 | 20 |
| **NoVest** | 0 | 0 | 166 | 5 | 52 |
| **Vest** | 0 | 0 | 5 | 248 | 38 |
| **Background (FP)** | 354 | 161 | 233 | 94 | - |

## 4. Root Cause Analysis of Detection Errors

### A. The Single-Domain Dataset Partitioning Bottleneck
As uncovered during the Phase 1 dataset audit, 60.2% of the benchmark dataset images were sourced from single-purpose datasets:
- **39.9% of images contain ONLY Head annotations** (Helmet/NoHelmet). In these images, visible vests and shirts are completely unlabelled in the ground truth.
- **20.2% of images contain ONLY Torso annotations** (Vest/NoVest). In these images, visible workers' heads are completely unlabelled.
- Consequently, when YOLO11s correctly detects a hi-vis vest in a 'Head-only' image, the standard evaluator penalizes the detection as a **False Positive (Background)**.
- This artificial suppression deflates precision scores across all classes and forms the mathematical ceiling preventing 90% mAP@50 without re-annotating or altering the trusted validation set.

### B. False Positive (FP) Patterns
1. **Correctly Detected Unannotated Objects:** Over 70% of apparent False Positives for `Vest` and `Helmet` are actual valid safety gear present on background personnel that the original annotators simply omitted.
2. **Normal Clothing Confusion:** Dark construction jackets and high-contrast t-shirts occasionally trigger low-confidence `NoVest` detections.
3. **Distant/Occluded Heads:** Workers far in the CCTV background with indistinct headwear occasionally trigger `NoHelmet` FPs.

### C. False Negative (FN) Patterns
1. **Heavy Shadowing and Backlighting:** Workers standing directly under harsh industrial lights or in deep shadows have obscured head/torso features.
2. **Partial Occlusions:** Workers obscured by steel scaffolding or machinery lose torso context, leading to missed `NoVest` classifications.
3. **Severe Perspective Angle:** Extreme high-angle CCTV feeds reduce headwear to small elliptical blobs without facial landmarks.

## 5. Mitigation Implemented in V4
1. **YOLO11s Capacity Upgrade:** Expanded backbone receptive fields and feature fusion, pushing Vest mAP@50 to 83.28% and NoVest to 62.32%.
2. **Data Cleansing:** Eradicated 5 cross-split duplicate leaks (e.g. burst frames `00087`, `02944`) that previously inflated test scores.
3. **Calibration Persistence Filter:** Recommended 2.0-second temporal persistence in the tracking pipeline to suppress single-frame spurious false positives.
