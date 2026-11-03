# PPE v4 Dataset & Baseline Audit Report

**Project:** AI-Based CCTV PPE Compliance Monitoring System (YOLO11)
**Date:** 2026-11-03
**Status:** ACTIVE — Experiment 1 training in progress

---

## 1. Dataset Inventory

### 1.1 Validated Configuration (data_v4.yaml)

| Split | Directory | Images | Labels | Status |
|-------|-----------|--------|--------|--------|
| Train | ppe_v3_experiments/data/train_cleaned/images | 1,828 | 1,828 | Verified |
| Val   | ppe_v3_experiments/data/valid_trusted/images | 454 | 454 | Verified |

Both paths confirmed accessible. Zero missing or corrupt files per YOLO scan.

### 1.2 Class Distribution

Training Set: 4,860 total instances

| Class | Count | % of total | Ratio to Helmet |
|-------|-------|-----------|-----------------|
| 0 Helmet | 1,970 | 40.5% | 1.0x |
| 1 NoHelmet | 959 | 19.7% | 0.49x |
| 2 NoVest | 835 | 17.2% | 0.42x |
| 3 Vest | 1,096 | 22.6% | 0.56x |

Validation Set: 1,177 total instances

| Class | Count | % of total |
|-------|-------|-----------|
| 0 Helmet | 454 | 38.6% |
| 1 NoHelmet | 209 | 17.8% |
| 2 NoVest | 223 | 18.9% |
| 3 Vest | 291 | 24.7% |

Imbalance finding: NoVest is the least represented class (17.2% train, 18.9% val).
The Helmet:NoVest ratio of ~2.4:1 is not extreme but is meaningful.
Distribution is consistent between train and val — this is positive.

### 1.3 Annotation Quality Findings (from prior audit)

| Issue | Count | Resolution |
|-------|-------|-----------|
| Out-of-domain images excluded | 2 | Excluded from both splits |
| Policy-ambiguous images held out | 2 | Held out pending policy confirmation |
| Missing Helmet annotations corrected | 5 images, 5 boxes | Corrected in valid_trusted |
| Missing Vest annotations corrected | 1 image, 2 boxes | Corrected in valid_trusted |
| Missing NoVest annotations corrected | 1 image, 1 box | Corrected in valid_trusted |
| Dataset split annotation policy | Single-domain sub-datasets merged | Cannot fix without re-annotation |

Critical structural finding: 60.2% of training images originate from single-domain sub-datasets:
- 731 images (40.0%): ONLY Helmet/NoHelmet annotations
- 369 images (20.2%): ONLY Vest/NoVest annotations
- 729 images (39.8%): Both head and torso annotations

This is the primary structural bottleneck. The model rarely sees co-occurring Helmet + Vest
from a unified annotation policy.

---

## 2. Baseline Model Performance

All metrics on trusted validation set (454 images, 1,177 instances), conf=0.001.

### 2.1 PPE v2 — Primary Baseline

| Metric | Value |
|--------|-------|
| Precision | 60.5% |
| Recall | 76.7% |
| F1 | 67.6% |
| mAP@50 | 71.8% |
| mAP@50-95 | 50.8% |
| Inference speed | 29.8 ms/image |

Per-class:

| Class | mAP@50 | F1 | Notes |
|-------|--------|-----|-------|
| Helmet | 68.6% | 63.9% | Moderate — 17% FN-to-background |
| NoHelmet | 79.5% | 71.7% | Strong |
| NoVest | 58.8% | 57.6% | Weakest — 24% FN-to-background |
| Vest | 80.3% | 76.7% | Strong |

### 2.2 PPE v3 (3-epoch fine-tune from v2, cls=1.2) — REGRESSED

| Metric | Value | Delta vs v2 |
|--------|-------|------------|
| Precision | 61.0% | +0.5% |
| Recall | 69.5% | -7.2% |
| F1 | 65.0% | -2.6% |
| mAP@50 | 68.6% | -3.2% |
| mAP@50-95 | 47.2% | -3.6% |

Regression root causes:
1. Insufficient epochs (3) — model was still in early warmup phase
2. LR=0.002 too high — caused instability when combined with AdamW auto-tune
3. cls=1.2 too weak to shift loss landscape in only 3 epochs

Conclusion: PPE v2 is the current best. All v4 experiments start from v2.

---

## 3. Error Analysis

From confusion matrix of v2_standard_trusted:

- Background false negatives dominate all error modes
- NoVest: 24% FN-to-background (highest) — model confidence is below threshold
- Helmet: 17% FN-to-background — second worst
- Cross-class confusion (Vest/NoVest, Helmet/NoHelmet) is secondary

The primary lever is improving recall/confidence on NoVest and Helmet.

---

## 4. Experiment Plan

### Experiment 1 (RUNNING): YOLO11n @ 640, 30 epochs from v2

Hypothesis: 30 epochs with cosine LR, early stopping, tuned augmentation will improve v2.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base checkpoint | ppe_v2/weights/best.pt | Best known model |
| Architecture | YOLO11n (inherited) | Fast on CPU |
| Epochs | 30 (patience=10) | Convergence + early stop |
| Batch | 8 | CPU limit |
| imgsz | 640 | Baseline |
| LR | AdamW auto (~0.00125) | YOLO auto-tune |
| cos_lr | True | Smooth annealing |
| cls | 1.5 | Stronger NoVest signal |
| mosaic | 0.8 | Small-object co-occurrence |
| mixup | 0.1 | Light diversity |
| close_mosaic | 5 | Stable final epochs |
| Estimated time | ~60 min | |

Status: RUNNING — task-419

### Experiment 2 (Planned): YOLO11n @ 768 imgsz

Hypothesis: Higher resolution helps small hardhat and vest detection in CCTV footage.
Decision gate: Run only if Exp 1 achieves >75% mAP@50.
Base checkpoint: Best result from Exp 1.
Estimated time: ~90 min.

### Experiment 3 (Planned): YOLO11s Architecture

Hypothesis: YOLO11s has ~4x more parameters and higher capacity ceiling.
Base checkpoint: yolo11s.pt (COCO pretrained).
CPU inference note: ~90 ms/image vs 30 ms for YOLO11n.
Estimated time: ~120 min.

---

## 5. Why 90% mAP@50 Is Difficult — Structural Bottleneck Analysis

| Factor | Impact | Addressable? |
|--------|--------|-------------|
| Single-domain annotation gap (60.2% images) | High | Only partially with more epochs |
| NoVest scarcity (17.2% of instances) | Medium | cls weighting helps marginally |
| Small objects in distant CCTV footage | High | Higher resolution input helps |
| CPU-only training | High | Hardware constraint |
| Heterogeneous annotation policies | High | Cannot fix without re-annotation |
| NoVest visual ambiguity | Medium | Hard negative mining needed |

Realistic achievable target on this hardware/dataset:
- 75-80% mAP@50 with 30-epoch fine-tuning
- 80-85% with YOLO11s + extended training on GPU
- 90%+ would require GPU, 100+ epochs, consistent re-annotation, and hard negative mining

---

## 6. Status Tracker

| Experiment | Status | mAP@50 |
|------------|--------|--------|
| PPE v2 Baseline (conf=0.001) | Done | 71.8% (BEST CURRENT) |
| PPE v3 Finetuned 3ep | Done | 68.6% (regressed) |
| v4 Exp1: n640 30ep from v2 | RUNNING | TBD |
| v4 Exp2: n768 from best | Planned | TBD |
| v4 Exp3: YOLO11s | Planned | TBD |
