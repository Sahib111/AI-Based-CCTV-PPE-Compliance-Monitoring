# Second-Pass PPE Annotation Quality Verification Report

**Dataset:** 20250731-PPE2286 (Roboflow YOLOv11 Format)  
**Verification Date:** 2026-11-02  
**Auditor:** Computer Vision & Dataset Quality Assurance Engineer  
**Output Directory:** `annotation_audit_results/second_pass/`  

---

## 1. Verification of the Four Suspected Errors

Each of the four suspected error images was investigated individually at the bounding box level using high-resolution crop extractions and direct pixel inspection.

---

### Image 1: `00718_jpg.rf.5f4eb3952834c71021471805ba96f030.jpg`
- **Resolution:** 1280 × 720 (CCTV camera view, rooftop with AC compressor units, timestamp `2018/05/16 16:46:08`).
- **Total Existing Bounding Boxes:** 4

#### Per-Box Investigation:
1. **Box #1 (`NoVest`):**
   - **Actual Visible Object:** Upper torso of a person wearing a plain white short-sleeved collared shirt. No high-visibility or reflective vest.
   - **Assigned Class:** `NoVest` (class 2)
   - **Visually Defensible:** Yes. The person is clearly not wearing a high-visibility safety vest.
   - **Domain Fit:** Plausible industrial surveillance (rooftop CCTV).
   - **Confidence:** HIGH
2. **Box #2 (`Helmet`):**
   - **Actual Visible Object:** A white industrial safety hard hat with rigid dome curvature, a front visor/brim, and a printed logo/emblem on the front. *(Note: The initial first-pass suspicion that this was an ordinary "baseball cap" was disproven upon pixel-level cropped inspection).*
   - **Assigned Class:** `Helmet` (class 0)
   - **Visually Defensible:** Yes. The object is a genuine industrial hard hat.
   - **Domain Fit:** Yes (PPE safety equipment).
   - **Confidence:** HIGH
3. **Box #3 (`NoVest`):**
   - **Actual Visible Object:** Upper torso of an adjacent person wearing a plain greyish-green t-shirt and blue shorts. No safety vest.
   - **Assigned Class:** `NoVest` (class 2)
   - **Visually Defensible:** Yes. No safety vest present.
   - **Domain Fit:** Plausible industrial surveillance.
   - **Confidence:** HIGH
4. **Box #4 (`NoHelmet`):**
   - **Actual Visible Object:** Head of the person next to the helmeted individual, showing natural dark hair and glasses with no protective headwear.
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** Yes. The head is bare with no helmet.
   - **Domain Fit:** Plausible industrial surveillance.
   - **Confidence:** HIGH

#### Image-Level Finding:
- **Box-level accuracy:** The 4 existing bounding boxes are visually defensible (Box #2 is genuinely a hard hat, not a baseball cap).
- **Scene-level incompleteness:** There are **6 visible individuals** traversing the rooftop in the frame. Four individuals on the left side of the frame (all with bare heads and regular shirts) are **completely unannotated**. This demonstrates severe partial annotation of multi-person CCTV scenes.

---

### Image 2: `02673_jpg.rf.e2f1414b99d3c6a630cd1f83d41cf348.jpg`
- **Resolution:** 500 × 333 (Outdoor road/infrastructure delegation beside a passenger van/minibus).
- **Total Existing Bounding Boxes:** 4

#### Per-Box Investigation:
1. **Box #1 (`NoHelmet`):**
   - **Actual Visible Object:** Head of an adult male wearing a light blue button-down shirt. Bare head with dark hair; no headwear.
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** Yes, purely as "head lacking a helmet".
   - **Domain Fit:** Questionable. The setting is an on-site delegation/visit alongside a vehicle and roadway. The person is dressed in business casual attire. Without documented project rules regarding whether site visitors or delegations must wear helmets, classifying this individual as a non-compliant worker is policy-dependent.
   - **Confidence:** HIGH on visual feature; MEDIUM on domain applicability.
2. **Box #2 (`NoHelmet`):**
   - **Actual Visible Object:** Head of another adult male in a blue collared shirt with bare head and dark hair.
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** Yes, physically bare head.
   - **Domain Fit:** Same visitor/delegation ambiguity as Box #1.
   - **Confidence:** HIGH on visual feature; MEDIUM on domain applicability.
3. **Box #3 (`Helmet`):**
   - **Actual Visible Object:** A blue industrial safety helmet with harness/chin strap worn by a person holding rolled blueprints/drawings.
   - **Assigned Class:** `Helmet` (class 0)
   - **Visually Defensible:** Yes. Clearly an industrial hard hat.
   - **Domain Fit:** Yes (technical site personnel).
   - **Confidence:** HIGH
4. **Box #4 (`Helmet`):**
   - **Actual Visible Object:** A red industrial safety helmet with brim and suspension harness.
   - **Assigned Class:** `Helmet` (class 0)
   - **Visually Defensible:** Yes. Clearly an industrial hard hat.
   - **Domain Fit:** Yes.
   - **Confidence:** HIGH

#### Image-Level Finding:
- Helmets (Boxes #3 & #4) are confirmed valid.
- The `NoHelmet` annotations (Boxes #1 & #2) highlight a core policy dilemma: the dataset selectively tags some individuals in business clothes as `NoHelmet` while ignoring other un-helmeted individuals standing right next to them in the same delegation (e.g., the speaker in the blue suit, the woman on the far left, and officials in white shirts).

---

### Image 3: `12_jpg.rf.b835e726e69aac2a501c377d8efe151a.jpg`
- **Resolution:** 640 × 640 (Urban roadside utility work zone).
- **Total Existing Bounding Boxes:** 2

#### Per-Box Investigation:
1. **Box #1 (`NoHelmet`):**
   - **Actual Visible Object:** The head of a pedestrian seen from the rear, carrying a shopping bag and walking along the public sidewalk/street. She is wearing a **woven straw sunhat with a decorative bow** and long curly hair.
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** She is not wearing an industrial hard hat. However, labeling an ordinary pedestrian's straw sunhat on a public street as `NoHelmet` treats any civilian headwear as a safety compliance violation.
   - **Domain Fit:** NO. A CCTV industrial PPE compliance monitoring system should detect workers within a monitored perimeter, not pedestrians walking past on public sidewalks.
   - **Confidence:** HIGH
   - **Assessment:** `LIKELY_ERROR` (Domain mismatch and inappropriate target selection).
2. **Box #2 (`Helmet`):**
   - **Actual Visible Object:** A blue industrial hard hat with reflective striping worn by a utility worker wearing a high-visibility vest and work boots, actively working in the excavation area.
   - **Assigned Class:** `Helmet` (class 0)
   - **Visually Defensible:** Yes, clearly an active worker wearing a safety hard hat.
   - **Domain Fit:** Yes, prototypical industrial PPE target.
   - **Confidence:** HIGH
   - **Assessment:** `NO_ERROR_FOUND`.

---

### Image 4: `66_jpg.rf.59fa725d703fa7afc7f45d05ac92f27c.jpg`
- **Resolution:** 640 × 640 (Indoor commercial dining room / hotpot restaurant).
- **Total Existing Bounding Boxes:** 2

#### Per-Box Investigation:
1. **Box #1 (`NoHelmet`):**
   - **Actual Visible Object:** Head of a young customer looking down at a mobile phone while seated at a restaurant dining table (hotpot burner, food bowl, drinking glasses visible).
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** Strictly speaking, the person is not wearing a helmet. But the context is completely devoid of any workplace, construction, industrial, or PPE requirement.
   - **Domain Fit:** NO. Completely out of domain.
   - **Confidence:** HIGH
   - **Assessment:** `CONFIRMED_ERROR` (Gross domain error).
2. **Box #2 (`NoHelmet`):**
   - **Actual Visible Object:** Back of the head of another seated diner at the same table.
   - **Assigned Class:** `NoHelmet` (class 1)
   - **Visually Defensible:** Bare head.
   - **Domain Fit:** NO. Completely out of domain.
   - **Confidence:** HIGH
   - **Assessment:** `CONFIRMED_ERROR` (Gross domain error).

#### Image-Level Finding:
- This image represents an unintended, out-of-domain image inclusion in the dataset. Including indoor restaurant diners in an industrial PPE compliance dataset introduces false-positive negative-class samples that would cause a trained CCTV model to trigger compliance alarms on ordinary indoor civilian populations.

---

## 2. Investigation of Suspected Missing Annotations

We investigated the 5 suspected images to determine whether missing annotations represent genuine labeling errors or class requirements omitted by the original annotators.

---

### 1. `00838_jpg.rf.9df07396df1ffd66cfeb46e4cdc5a44a.jpg`
- **Scene:** Three workers passing through security turnstiles at a construction site entrance ("欢迎员工").
- **Visible PPE Items:**
  1. Worker 1 (Left turnstile): Clearly wearing a **white safety hard hat** and a dark zip jacket over a t-shirt (no reflective vest).
  2. Worker 2 (Center turnstile): Clearly wearing a **white safety hard hat** and a grey zip fleece without reflective bands (no reflective vest).
  3. Worker 3 (Right turnstile): Clearly wearing a **white safety hard hat** and a green quilted jacket (no reflective vest).
- **Existing Annotations:**
  - Box #1: `NoVest` on Worker 1 torso.
  - Box #2: `NoVest` on Worker 2 torso.
  - Helmet Annotations: **ZERO**.
  - Worker 3: **ZERO annotations** of any kind.
- **Genuine Error vs. Policy:**
  - This is a **GENUINE MISSING ANNOTATION ERROR**.
  - The dataset defines class 0 (`Helmet`). Three workers are located directly in the camera foreground wearing unmistakable white hard hats.
  - If a model correctly detects these hard hats, evaluation against this ground truth will falsely penalize the model with **3 False Positives for Helmet**, and will fail to award a true positive for Worker 3's absence of vest.

---

### 2. `00587_jpg.rf.2d89a904ea1815c04f6835aa6516d210.jpg`
- **Scene:** Two railway survey technicians standing on railway tracks operating survey theodolite equipment.
- **Visible PPE Items:**
  1. Technician 1 (Left): Wearing a white hard hat with chin strap and a **bright yellow high-visibility safety vest with silver reflective horizontal stripes**.
  2. Technician 2 (Right): Wearing a white hard hat with chin strap and a **bright yellow high-visibility safety vest with silver reflective horizontal stripes**.
- **Existing Annotations:**
  - Box #1: `Helmet` on Technician 1 head.
  - Box #2: `Helmet` on Technician 2 head.
  - Vest Annotations: **ZERO**.
- **Genuine Error vs. Policy:**
  - This is a **GENUINE MISSING ANNOTATION ERROR**.
  - Both high-visibility vests are unoccluded, occupy large proportions of the foreground, and display standard reflective safety markings.
  - The likely cause is dataset aggregation: this image originated from a helmet-only source collection and was imported into this 4-class dataset without completing the vest annotations.
  - An evaluation against this image will penalize any accurate model that detects the two visible vests with **2 False Positives for Vest**.

---

### 3. `03128_jpg.rf.1e6d990894e8aef4b65b19dc5f0990f9.jpg`
- **Scene:** Night-time road paving operations under artificial construction floodlighting.
- **Visible PPE Items:**
  1. Foreground Worker: Wearing an **illuminated halo safety helmet** (hard hat with 360-degree light band) and an orange reflective safety vest.
  2. Background Machine Operator: Seated on the paving machinery wearing a white hard hat and yellow reflective vest.
  3. Background Pedestrian Worker: Walking behind a barrier wearing a hard hat and vest.
- **Existing Annotations:**
  - Box #1: `Vest` on foreground worker torso.
  - All other items: **ZERO annotations**.
- **Distinction (Clear vs. Uncertain/Occluded):**
  - **Foreground Worker Helmet:** Unambiguously present directly on top of the worker whose vest was labeled. The absence of a `Helmet` label here is a **GENUINE MISSING ANNOTATION**.
  - **Background Personnel:** The machine operator and background worker are partially occluded, distant, and subject to intense lighting glare/smoke. In standard labeling guidelines, annotators often establish a distance/resolution threshold below which background objects are omitted. Therefore, omission of the background individuals is classified as **POLICY_UNCLEAR**, while the foreground helmet omission is a **CONFIRMED_ERROR**.

---

### 4. `helmet_jacket_00831_jpg.rf.c6eb0b134e84d9ae9605e972d5b7a894.jpg`
- **Scene:** Studio stock photo (isolated on white background, watermark `123RF`) of a construction worker holding the brim of a hard hat.
- **Visible PPE Items:**
  1. White industrial safety hard hat (head, completely unoccluded).
  2. High-visibility lime-green safety vest with double silver reflective bands (torso, completely unoccluded).
- **Existing Annotations:**
  - Box #1: `Vest` (covers the vest).
  - Helmet: **ZERO annotations**.
- **Genuine Error vs. Policy:**
  - This is a **GENUINE MISSING ANNOTATION ERROR**.
  - The source file name itself is `helmet_jacket_00831`, indicating the image depicts both helmet and jacket. The hard hat is high-resolution, centered, and unmistakable. Omission of the helmet in a 4-class PPE dataset is an objective defect.

---

### 5. `helmet_jacket_00885_jpg.rf.a0162cb17401caa7503e16d7d3553f03.jpg`
- **Scene:** Studio catalog portrait against a neutral grey studio background of a worker wearing PPE.
- **Visible PPE Items:**
  1. Blue industrial safety helmet (head, completely unoccluded, high sharpness).
  2. High-visibility two-tone yellow/navy safety vest with dual reflective stripes (torso, completely unoccluded).
- **Existing Annotations:**
  - Box #1: `Vest` (covers the vest).
  - Helmet: **ZERO annotations**.
- **Genuine Error vs. Policy:**
  - This is a **GENUINE MISSING ANNOTATION ERROR**.
  - Like image 00831, the hard hat is in plain sight on a studio model. Omitting `Helmet` while annotating `Vest` corrupts benchmark evaluation by generating false negative penalties for accurate detectors.

---

## 3. Investigation of the Original Annotation Policy

### Analysis of Documentation
We carefully inspected:
1. `README.dataset.txt`: Contains only the dataset URL (`https://universe.roboflow.com/ying2001/20250731-ppe2286-m0ega`), user name (`ying2001`), and license (`CC BY 4.0`).
2. `README.roboflow.txt`: Contains generic export metadata ("Helmet-LXNu-GDfm-C1kO-RxmU-AnTu are annotated in YOLOv11 format", total 2,286 images, no pre-processing or augmentation applied).
3. `data.yaml`: Lists classes `['Helmet', 'NoHelmet', 'NoVest', 'Vest']` and paths.

**Critical Finding on Documentation:**
> **The dataset documentation contains ZERO official guidelines, annotation protocols, compliance definitions, or worker boundary criteria.** There is no documented policy provided by the dataset author.

### Nature of the Classes: Objects vs. Compliance Status
Empirical examination of the actual bounding boxes across the images proves that the dataset uses a **contradictory mixture of both object detection and compliance state classification**:

1. **`Helmet` and `Vest` are OBJECT classes:**
   - They tightly bound tangible physical safety gear (hard hats on the head, reflective vests on the torso).
2. **`NoHelmet` and `NoVest` are COMPLIANCE STATUS (absence) classes:**
   - Because "absence of PPE" is not a physical object, annotators place bounding boxes over **body parts** (bare human heads or plain shirts/jackets).
   - This creates an inherent semantic flaw: bounding box detectors (such as YOLO) are designed to learn feature representations of physical objects, not the contextual absence of an object on a specific human role.
3. **Absence of a "Person" or "Worker" Anchor:**
   - In modern PPE architectures, compliance is typically modeled hierarchically:
     $$\text{Person Detection} \rightarrow \text{Head/Torso Crop} \rightarrow \text{PPE Classification (Helmet / No Helmet)}$$
   - By attempting to detect `NoHelmet` and `NoVest` as independent 2D bounding boxes without a parent `Person` or `Worker` class, the dataset creates severe boundary failures:
     - Is any bare human head in the frame a `NoHelmet`? (As seen in the restaurant and sidewalk pedestrian examples).
     - Or only workers inside a construction zone? (Which cannot be parsed without scene context).
4. **Composite Dataset Assembly:**
   - Filename prefixes (`00XXX_jpg`, `helmet_jacket_XXXXX_jpg`, `reflectivevest_XXXXX_jpg`) prove the dataset is a collage of multiple pre-existing sub-datasets merged together without cross-annotation:
     - Sub-dataset A: only annotated helmets.
     - Sub-dataset B: only annotated vests.
     - Sub-dataset C: annotated both.
   - Merging these without harmonizing annotations resulted in valid helmets and vests being left unannotated whenever an image from a single-class dataset was imported.

---

## 4. Second-Pass Audit Summary Table

The complete line-by-line verification table is saved in `annotation_audit_results/second_pass/second_pass_audit.csv`. Below is the consolidated summary:

| Image Name | Existing Class | Suspected Issue | Evidence | Confidence | Final Assessment | Recommended Action |
|---|---|---|---|---|---|---|
| `00718_jpg…030.jpg` | Helmet (Box #2) | Suspected baseball cap | Cropped inspection confirms white industrial safety hard hat with rigid dome, brim, and logo. | HIGH | **NO_ERROR_FOUND** | Retain Box #2 as Helmet |
| `00718_jpg…030.jpg` | NoHelmet (Box #4) | Selective person labeling | Bare head with dark hair; no helmet present. | HIGH | **NO_ERROR_FOUND** | Retain Box #4 as NoHelmet |
| `00718_jpg…030.jpg` | NoVest (Boxes #1, #3) | Incomplete scene coverage | Torsos have plain shirts; 4 other visible rooftop individuals unannotated. | HIGH | **POLICY_UNCLEAR** | Formalize rooftop annotation policy |
| `02673_jpg…48.jpg` | NoHelmet (Boxes #1, #2) | Labeled on business visitors | Bare heads in delegation beside van; visitor PPE mandate undocumented. | HIGH | **POLICY_UNCLEAR** | Formalize policy for visitor delegations |
| `02673_jpg…48.jpg` | Helmet (Boxes #3, #4) | Questioned in mixed scene | Box #4 is red hard hat; Box #3 is blue hard hat with chin strap and blueprints. | HIGH | **NO_ERROR_FOUND** | Retain Boxes #3 and #4 as Helmet |
| `12_jpg…1a.jpg` | NoHelmet (Box #1) | Pedestrian in straw hat | Pedestrian in woven straw sunhat with bow on public street. | HIGH | **LIKELY_ERROR** | Remove Box #1; limit targets to site personnel |
| `12_jpg…1a.jpg` | Helmet (Box #2) | Active worker verification | Utility worker in excavation wearing blue hard hat and high-vis vest. | HIGH | **NO_ERROR_FOUND** | Retain Box #2 as Helmet |
| `66_jpg…7c.jpg` | NoHelmet (Boxes #1, #2) | Diners in restaurant | Patrons seated at indoor hotpot restaurant table with phones/tableware. | HIGH | **CONFIRMED_ERROR** | Remove image `66_jpg` from dataset |
| `00838_jpg…44a.jpg` | NoVest (Boxes #1, #2) | Missing Helmets at turnstile | 3 workers with white hard hats; 0 Helmet boxes; 3rd worker unannotated. | HIGH | **CONFIRMED_ERROR** | Add 3 Helmet boxes; add 1 NoVest box |
| `00587_jpg…210.jpg` | Helmet (Boxes #1, #2) | Missing Vests on surveyors | 2 railway technicians prominently wearing bright yellow reflective vests. | HIGH | **CONFIRMED_ERROR** | Add 2 Vest boxes |
| `03128_jpg…0f9.jpg` | Vest (Box #1) | Missing foreground helmet | Worker wears illuminated halo hard hat with 0 Helmet label. | HIGH | **CONFIRMED_ERROR** | Add Helmet box on foreground worker |
| `helmet_jacket_00831…` | Vest (Box #1) | Missing helmet on stock photo | Worker model wears and tips white hard hat; 0 Helmet label. | HIGH | **CONFIRMED_ERROR** | Add Helmet box on white hard hat |
| `helmet_jacket_00885…` | Vest (Box #1) | Missing helmet on catalog photo | Model wears blue industrial safety helmet; 0 Helmet label. | HIGH | **CONFIRMED_ERROR** | Add Helmet box on blue helmet |

---

## 5. Key Engineering Recommendations for YOLO11 Compliance System

1. **Do not use the raw validation set for model evaluation:**
   - Evaluating a YOLO11 model against this validation set will generate misleading metrics.
   - For example, high-precision detections on prominent hard hats in `00838_jpg`, `00587_jpg`, `helmet_jacket_00831`, and `helmet_jacket_00885` will be scored as **false positives**, artificially depressing precision.
   - Similarly, correct detections of high-vis vests in `00587_jpg` will be counted as false positives.
2. **Transition from direct 4-class bounding boxes to a hierarchical compliance pipeline:**
   - Detecting `NoHelmet` and `NoVest` as standalone object boxes is conceptually flawed for CCTV monitoring.
   - Recommended architecture:
     1. Stage 1: Standard Person/Worker Detector (detect `Person`).
     2. Stage 2: Head & Torso crop extraction.
     3. Stage 3: Binary PPE Classifiers (`Head` $\rightarrow$ `Helmet` vs. `NoHelmet`; `Torso` $\rightarrow$ `Vest` vs. `NoVest`).
3. **Dataset Sanitization Checklist (Prior to Benchmarking):**
   - **Step 1:** Purge out-of-domain images (`66_jpg`).
   - **Step 2:** Complete multi-class annotations for merged single-source images (add missing helmets to `helmet_jacket_` and `00838`, add missing vests to `00587`).
   - **Step 3:** Establish geofencing/ROI rules in the CCTV analytics logic to restrict PPE evaluation exclusively to active work zones, filtering out street pedestrians and casual bystanders.
