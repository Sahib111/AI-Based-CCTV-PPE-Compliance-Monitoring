import os
import shutil
import glob
import csv

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
SRC_IMG_DIR = os.path.join(BASE_DIR, "ppe_v3_experiments", "data", "train_cleaned", "images")
SRC_LBL_DIR = os.path.join(BASE_DIR, "ppe_v3_experiments", "data", "train_cleaned", "labels")

DST_DATA_DIR = os.path.join(BASE_DIR, "ppe_v4_experiments", "data")
DST_TRAIN_IMG = os.path.join(DST_DATA_DIR, "train", "images")
DST_TRAIN_LBL = os.path.join(DST_DATA_DIR, "train", "labels")

os.makedirs(DST_TRAIN_IMG, exist_ok=True)
os.makedirs(DST_TRAIN_LBL, exist_ok=True)

# Documented exclusions: Cross-split leakage / near-duplicate frames with trusted validation set
EXCLUDED_PATTERNS = {
    "00087_jpg.rf.6367621f76b66bc6ebb4381eaddddaf7": "Cross-split exact near-duplicate with val image 00472 (dHash dist 0, mean pixel diff 5.2)",
    "02944_jpg.rf.66db8b6bd52e58d40eba6cdb8c621baa": "Cross-split exact near-duplicate with val image 02476 (dHash dist 0, mean pixel diff 2.6)",
    "02951_jpg.rf.cf4438800d97dc6ae06e489bb1970bf9": "Temporal burst duplicate of val image 02969 (dHash dist 2)",
    "02955_jpg.rf.69cc0d44c0ff3b7d1629667e3d92f8bf": "Temporal burst duplicate of val image 02969 (dHash dist 1)",
    "02968_jpg.rf.8c129993ba2ee87732ccd8017052ba74": "Temporal burst duplicate of val image 02969 (dHash dist 1)"
}

manifest_path = os.path.join(DST_DATA_DIR, "dataset_v4_manifest.csv")

src_imgs = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "*.*")))
print(f"Total source cleaned training images: {len(src_imgs)}")

kept_count = 0
excluded_count = 0

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Source_Dir", "Action", "Reason"])
    
    for img_path in src_imgs:
        base_name = os.path.basename(img_path)
        stem = os.path.splitext(base_name)[0]
        
        # Check exclusion
        reason = None
        for pattern, expl in EXCLUDED_PATTERNS.items():
            if pattern in stem:
                reason = expl
                break
                
        if reason:
            excluded_count += 1
            writer.writerow([base_name, "train_cleaned", "EXCLUDED", reason])
        else:
            kept_count += 1
            dst_img_path = os.path.join(DST_TRAIN_IMG, base_name)
            shutil.copy2(img_path, dst_img_path)
            
            lbl_name = stem + ".txt"
            src_lbl = os.path.join(SRC_LBL_DIR, lbl_name)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(DST_TRAIN_LBL, lbl_name)
                shutil.copy2(src_lbl, dst_lbl)
            writer.writerow([base_name, "train_cleaned", "KEPT", "Cleaned verified industrial training sample"])

print(f"Dataset V4 Train Preparation Complete:")
print(f"  Kept: {kept_count}")
print(f"  Excluded: {excluded_count}")
print(f"  Manifest written to: {manifest_path}")

# Write data_v4.yaml
data_yaml_path = os.path.join(DST_DATA_DIR, "data_v4.yaml")
yaml_content = f"""path: {BASE_DIR.replace('\\', '/')}
train: ppe_v4_experiments/data/train/images
val: ppe_v3_experiments/data/valid_trusted/images
test:

nc: 4
names: ['Helmet', 'NoHelmet', 'NoVest', 'Vest']
"""
with open(data_yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"data_v4.yaml written to: {data_yaml_path}")

