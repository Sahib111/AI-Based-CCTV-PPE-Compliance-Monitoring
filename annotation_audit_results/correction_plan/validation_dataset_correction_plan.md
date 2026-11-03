# Validation Dataset Correction Plan

**Project:** AI-Based CCTV PPE Compliance Monitoring System (YOLO11)  
**Dataset:** `20250731-PPE2286` (Roboflow Export)  
**Date:** 2026-11-02  
**Status:** PROPOSED — PENDING HUMAN APPROVAL  
**Output Directory:** `annotation_audit_results/correction_plan/`  

---

## 1. Executive Summary

Following the second-pass audit of the validation set, this correction plan provides exact, normalized coordinates and visual comparison sheets for all confirmed annotation defects, while holding ambiguous cases for formal human policy approval.

**Key Guardrails Followed:**
- **Zero modification of original files:** No image or label file in `valid/` or `train/` has been altered.
- **Human-in-the-loop review:** All proposed additions are rendered side-by-side with original labels in visual review sheets for verification before applying any dataset updates.
- **Policy isolation:** Non-worker / pedestrian cases (`02673`, `12_jpg`) are isolated for human policy sign-off rather than automatically modified.
- **Verified accuracy:** Image `00718_jpg` is preserved unchanged following confirmation that its white headwear is a genuine industrial hard hat.

---

## 2. Proposed Changes & Normalized YOLO Coordinates

The table below details every proposed label correction. All coordinates follow standard YOLO format:  
$$\text{Class ID} \quad x_{\text{center}} \quad y_{\text{center}} \quad \text{width} \quad \text{height} \quad (\text{normalized } 0.0 - 1.0)$$

| Image Name | Action | Class | Normalized YOLO ($c_x, c_y, w, h$) | Bounding Box Pixel Extent ($x_1, y_1, x_2, y_2$) | Confidence | Approval Status |
|---|---|---|---|---|---|---|
| **`66_jpg.rf…7c.jpg`** | **EXCLUDE_IMAGE** | N/A | Entire image and label removed | Image resolution 640 × 640 | HIGH | Ready to exclude (restaurant scene) |
| **`00838_jpg.rf…4a.jpg`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.295802 0.232143 0.083969 0.094156` | `[133, 57, 177, 86]` | HIGH | Worker 1 (left) white hard hat |
| **`00838_jpg.rf…4a.jpg`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.532443 0.232143 0.083969 0.100649` | `[257, 56, 301, 87]` | HIGH | Worker 2 (center) white hard hat |
| **`00838_jpg.rf…4a.jpg`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.855916 0.225649 0.089695 0.113636` | `[425, 52, 472, 87]` | HIGH | Worker 3 (right) white hard hat |
| **`00838_jpg.rf…4a.jpg`** | **ADD_BOX** | 2 (`NoVest`) | `2 0.842557 0.487013 0.166031 0.311688` | `[398, 102, 485, 198]` | MEDIUM | **NEEDS HUMAN APPROVAL** (Worker 3 torso) |
| **`00587_jpg.rf…10.jpg`** | **ADD_BOX** | 3 (`Vest`) | `3 0.288000 0.654655 0.276000 0.642643` | `[75, 111, 213, 325]` | HIGH | Technician 1 (left) yellow reflective vest |
| **`00587_jpg.rf…10.jpg`** | **ADD_BOX** | 3 (`Vest`) | `3 0.606000 0.650150 0.228000 0.612613` | `[246, 114, 360, 318]` | HIGH | Technician 2 (right) yellow reflective vest |
| **`03128_jpg.rf…f9.jpg`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.495833 0.511719 0.075000 0.101562` | `[440, 295, 512, 360]` | HIGH | Foreground worker halo hard hat |
| **`helmet_jacket_00831…`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.514062 0.111719 0.259375 0.145312` | `[246, 25, 412, 118]` | HIGH | Stock photo white safety hard hat |
| **`helmet_jacket_00885…`** | **ADD_BOX** | 0 (`Helmet`) | `0 0.602344 0.136719 0.251562 0.157812` | `[305, 37, 466, 138]` | HIGH | Catalog photo blue safety helmet |
| **`02673_jpg.rf…48.jpg`** | **HOLD** | N/A | None proposed automatically | Image resolution 500 × 333 | HIGH | **HELD FOR POLICY REVIEW** |
| **`12_jpg.rf…1a.jpg`** | **HOLD** | N/A | None proposed automatically | Image resolution 640 × 640 | HIGH | **HELD FOR POLICY REVIEW** |
| **`00718_jpg.rf…030.jpg`** | **RETAIN** | N/A | Existing 4 boxes retained as-is | Image resolution 1280 × 720 | HIGH | Verified correct (no change) |

---

## 3. Visual Review Sheets Catalogue

Side-by-side visual comparison review sheets have been generated and saved in `annotation_audit_results/correction_plan/review_sheets/`:

1. [`review_00838_jpg.rf.9df07.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_00838_jpg.rf.9df07.jpg): Shows 3 added `Helmet` boxes (magenta) and 1 proposed `NoVest` box (yellow, marked `NEEDS APPROVAL`).
2. [`review_00587_jpg.rf.2d89a.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_00587_jpg.rf.2d89a.jpg): Shows the 2 added `Vest` boxes (magenta) covering the prominent yellow vests.
3. [`review_03128_jpg.rf.1e6d9.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_03128_jpg.rf.1e6d9.jpg): Shows the added `Helmet` box (magenta) on the illuminated halo hard hat.
4. [`review_helmet_jacket_0083.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_helmet_jacket_0083.jpg): Shows the added `Helmet` box on the white hard hat.
5. [`review_helmet_jacket_0088.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_helmet_jacket_0088.jpg): Shows the added `Helmet` box on the blue hard hat.
6. [`review_66_jpg.rf.59fa725d.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_66_jpg.rf.59fa725d.jpg): Shows the proposed exclusion banner for the restaurant dining image.
7. [`review_02673_jpg.rf.e2f14.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_02673_jpg.rf.e2f14.jpg): Shows original annotations flagged for human policy decision.
8. [`review_12_jpg.rf.b835e726.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_12_jpg.rf.b835e726.jpg): Shows pedestrian straw hat annotation flagged for human policy decision.
9. [`review_00718_jpg.rf.5f4eb.jpg`](file:///C:/Users/Sahib%20Singh/Downloads/20250731-PPE2286y/20250731-ppe2286y/annotation_audit_results/correction_plan/review_sheets/review_00718_jpg.rf.5f4eb.jpg): Shows all original annotations preserved as verified.

---

## 4. Inspection of Training Labels (`train/labels`)

To verify whether the issues identified in the validation set exist across the broader dataset, we performed a structural audit of all **1,829 training label files**:

### Quantitative Findings:
- **Total Training Images/Labels:** 1,829
- **Overall Class Instances:**
  - `Helmet` (class 0): 1,970 annotations
  - `NoHelmet` (class 1): 960 annotations
  - `NoVest` (class 2): 836 annotations
  - `Vest` (class 3): 1,096 annotations
- **Annotation Domain Segregation:**
  - Labels containing **ONLY Vest/NoVest** (zero Helmet/NoHelmet): **369 images (20.2%)**
  - Labels containing **ONLY Helmet/NoHelmet** (zero Vest/NoVest): **731 images (40.0%)**
  - Labels containing **BOTH head and torso PPE classes**: **729 images (39.8%)**

### Key Takeaway for the Training Set:
> Over **60.2% of the training dataset (1,100 images)** contains single-domain annotations (annotating only vests or only helmets). This confirms that the dataset was compiled from disparate source collections without completing cross-class annotations.
> 
> *Note: Per instructions, no training images or label files were modified.*

---

## 5. Issues Requiring Human Decisions

Before generating the corrected validation dataset, the following three specific decisions require human sign-off:

### Decision 1: Worker 3 `NoVest` in `00838_jpg`
- **Situation:** Three workers enter the turnstile gate. Worker 1 and Worker 2 have `NoVest` annotated on their dark jackets. Worker 3 is wearing a dark green quilted jacket with no reflective elements.
- **Question:** Should Worker 3 receive a `NoVest` annotation (`2 0.842557 0.487013 0.166031 0.311688`) in addition to his hard hat?
- **Recommendation:** **YES**, approve the `NoVest` box. He is entering an active construction site alongside the other two workers under identical conditions.

### Decision 2: Visitor Delegation `NoHelmet` in `02673_jpg`
- **Situation:** An infrastructure inspection delegation is standing near a van. Box #1 and Box #2 label two men in button-down collared shirts as `NoHelmet`, but other members of the same group (including the speaker in a suit and adjacent officials) are unannotated.
- **Question:** Should `NoHelmet` be retained on site visitors / officials standing outside vehicles, or should `NoHelmet` be restricted strictly to active manual workers?
- **Recommendation:** If the CCTV system monitors worker compliance inside active work zones, delete the `NoHelmet` annotations on this image or exclude the image from validation, as it teaches the model to flag business visitors as PPE violators.

### Decision 3: Pedestrian Straw Sunhat in `12_jpg`
- **Situation:** A female civilian walking along the public sidewalk carrying a shopping bag is labeled `NoHelmet` because she wears a decorative straw sunhat.
- **Question:** Should passing pedestrians on public sidewalks be labeled `NoHelmet`?
- **Recommendation:** **DELETE Box #1** (`NoHelmet`). A CCTV safety compliance system should not penalize public pedestrians outside site boundaries.

---

## 6. Next Steps Upon Approval

Once you review and approve these proposals:
1. We will create a clean copy of the validation set in a dedicated folder (e.g., `valid_corrected/`) leaving original `valid/` untouched.
2. The 6 accepted modifications will be applied to the new label files.
3. Image `66_jpg` will be excluded from the corrected set and an updated `data_corrected.yaml` will be created for reliable YOLO11 evaluation.
