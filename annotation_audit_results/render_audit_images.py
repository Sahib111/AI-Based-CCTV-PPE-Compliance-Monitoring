"""
Script to render audit images with YOLO bounding boxes overlaid.
Reads from annotation_audit folder and valid/labels.
Saves rendered images to annotation_audit_results/rendered/
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Paths
BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
AUDIT_IMG_DIR = os.path.join(BASE_DIR, "annotation_audit")
LABEL_DIR = os.path.join(BASE_DIR, "valid", "labels")
OUT_DIR = os.path.join(BASE_DIR, "annotation_audit_results", "rendered")
os.makedirs(OUT_DIR, exist_ok=True)

# Class info: id -> (name, color)
CLASS_INFO = {
    0: ("Helmet",   "#00FF00"),   # Green
    1: ("NoHelmet", "#FF0000"),   # Red
    2: ("NoVest",   "#FF8C00"),   # Orange
    3: ("Vest",     "#00BFFF"),   # Blue
}

def parse_label_file(label_path):
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((cls, cx, cy, w, h))
    return annotations

def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

def render_image(img_path, annotations, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for cls, cx, cy, w, h in annotations:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
        cls_name, color = CLASS_INFO.get(cls, (f"cls{cls}", "#FFFFFF"))
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw label background
        label_text = cls_name
        text_bbox = draw.textbbox((x1, y1), label_text)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        lx1 = x1
        ly1 = max(0, y1 - th - 4)
        draw.rectangle([lx1, ly1, lx1 + tw + 4, ly1 + th + 4], fill=color)
        draw.text((lx1 + 2, ly1 + 2), label_text, fill="black")

    img.save(out_path, "JPEG", quality=90)
    return img_w, img_h

# Process each audit image
audit_images = sorted(os.listdir(AUDIT_IMG_DIR))
stats = []

for fname in audit_images:
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(AUDIT_IMG_DIR, fname)
    # Build label filename (same base, .txt extension)
    base = os.path.splitext(fname)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    out_path = os.path.join(OUT_DIR, base + "_annotated.jpg")

    annotations = parse_label_file(label_path)
    has_label = os.path.exists(label_path)

    img_w, img_h = render_image(img_path, annotations, out_path)

    # Count class annotations
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for cls, *_ in annotations:
        if cls in counts:
            counts[cls] += 1

    stats.append({
        "fname": fname,
        "label_path": label_path,
        "has_label": has_label,
        "total": len(annotations),
        "helmet": counts[0],
        "nohelmet": counts[1],
        "novest": counts[2],
        "vest": counts[3],
        "img_w": img_w,
        "img_h": img_h,
        "out_path": out_path,
    })
    print(f"[OK] {fname}: {len(annotations)} annotations -> {out_path}")

print("\n=== LABEL SUMMARY ===")
for s in stats:
    label_status = "FOUND" if s["has_label"] else "MISSING"
    print(f"{s['fname']}: label={label_status}, total={s['total']}, "
          f"Helmet={s['helmet']}, NoHelmet={s['nohelmet']}, NoVest={s['novest']}, Vest={s['vest']}, "
          f"img={s['img_w']}x{s['img_h']}")
