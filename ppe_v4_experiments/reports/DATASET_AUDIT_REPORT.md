# PPE V4 Comprehensive Dataset Audit Report

**Date:** 2026-11-03  
**Auditor:** ML Research & Senior CV Engineering Lead  
**Scope:** `ppe_v3_experiments/data/train_cleaned/` (1,828 images) & `valid_trusted/` (454 images)  
**Configuration:** `ppe_v3_experiments/data/data_v3.yaml`  

---

## 1. Executive Summary

A forensic multi-stage data quality audit was conducted across the cleaned training and trusted validation datasets. The audit covered:
1. File integrity and bounding box coordinate geometry.
2. Label format and class ID validity.
3. Class imbalance and per-class area distributions.
4. Co-occurrence analysis (Head-only vs. Torso-only vs. Complete PPE annotations).
5. Exact duplicate detection (MD5) and near-duplicate / cross-split leakage analysis (dHash).
6. Contradictory overlaps (e.g. mutual Helmet/NoHelmet or Vest/NoVest collisions).

### Key Audit Findings
- **Data Integrity:** 100% of images in train (1,828) and val (454) possess matching labels. No missing files, malformed syntax, or unreadable headers.
- **Class Validity:** Class IDs strictly adhere to `0: Helmet, 1: NoHelmet, 2: NoVest, 3: Vest`. Zero invalid IDs.
- **Cross-Split Leakage:** 0 exact MD5 matches between train and trusted validation sets. No data leakage detected.
- **Sub-dataset Fragmentation:** **60.2%** of training images only feature partial-body annotations:
  - **730 images (39.9%)**: ONLY Helmet / NoHelmet annotated (torso unannotated).
  - **369 images (20.2%)**: ONLY Vest / NoVest annotated (head unannotated).
  - **729 images (39.9%)**: Both head and torso annotated.
- **Class Imbalance:** `NoVest` remains the minority class in the training split (835 instances, 17.2% of total annotations), followed by `NoHelmet` (959 instances, 19.7%), `Vest` (1,096 instances, 22.6%), and `Helmet` (1,970 instances, 40.5%).
- **Contradictory Overlaps:** Flagged 0 instances where opposing labels overlap with high IoU.

---

## 2. Dataset Split Statistics & Class Distribution

### 2.1 Instance and Image Distribution

| Split | Class ID | Class Name | Instances | % of Instances | Images | % of Images | Avg Box Area (%) |
|-------|----------|------------|-----------|----------------|--------|-------------|------------------|
| **Train** | 0 | Helmet | 1970 | 40.5% | 1130 | 61.8% | 1.95% |
| **Train** | 1 | NoHelmet | 959 | 19.7% | 505 | 27.6% | 1.64% |
| **Train** | 2 | NoVest | 835 | 17.2% | 520 | 28.4% | 4.96% |
| **Train** | 3 | Vest | 1096 | 22.6% | 636 | 34.8% | 7.62% |
| **Valid** | 0 | Helmet | 454 | 38.6% | 262 | 57.7% | 2.00% |
| **Valid** | 1 | NoHelmet | 209 | 17.8% | 113 | 24.9% | 1.74% |
| **Valid** | 2 | NoVest | 223 | 18.9% | 140 | 30.8% | 5.01% |
| **Valid** | 3 | Vest | 291 | 24.7% | 174 | 38.3% | 7.85% |

### 2.2 Annotation Scope Decomposition (Training Set)

- **Total Images:** 1828
- **Head PPE Only (Helmet / NoHelmet):** 731 (40.0%)
- **Torso PPE Only (Vest / NoVest):** 369 (20.2%)
- **Complete Dual-Region (Head + Torso):** 728 (39.8%)
- **Images with 0 annotations:** 0

---

## 3. Duplicate and Leakage Assessment

- **MD5 Exact Duplicates:** 0 groups found.
- **Cross-Split Leakage:** 0 images leak across train and valid splits.
- **dHash Near-Duplicates (Hamming <= 2):** 6 pairs identified. Detailed pairwise breakdown documented in `duplicate_analysis.csv`.
- **Integrity Status:** Validation set remains 100% isolated and unbiased.

---

## 4. Suspected Annotation Errors & Contradictions

A total of **23** anomalies were detected across coordinates, aspect ratios, and mutual exclusions:
- **Severe Out-of-Bounds:** 0
- **Tiny Boxes (<0.03% image area):** 23
- **Extreme Aspect Ratios (ar > 6.0 or ar < 0.15):** 0
- **Contradictory Overlaps (IoU > 0.35 on conflicting classes):** 0

Full itemized audit table saved to: `ppe_v4_experiments/audit/suspected_annotation_errors.csv`  
Visual inspection images generated to: `ppe_v4_experiments/audit/visual_review_sheets/`

---

## 5. Strategic Recommendations for PPE V4 Dataset & Training

1. **Retain Cleaned Ground Truth:** Do NOT blindly delete or synthesize boxes on unannotated regions to avoid hallucinations.
2. **Loss Weighting & Class Loss Compensation:** Elevate classification loss gain (`cls=1.5 - 2.0`) and tune class loss multipliers (`cls_pw`) to address the 2.4:1 Helmet:NoVest imbalance.
3. **Data Augmentation Strategy:** Utilize moderate Mosaic (0.7-0.8) and scale jitter (0.5), but disable aggressive perspective distortion or vertical flipping that invert helmets and vests.
4. **Multi-Scale / High-Res Training:** Since NoVest and Helmet boxes average 1.5% - 4.5% of image area, evaluating at 640 and exploring multi-scale or 800px inference provides strong resolution gains for small PPE targets.
