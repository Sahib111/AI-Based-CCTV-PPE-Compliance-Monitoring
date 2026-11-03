import os
from PIL import Image, ImageDraw

BASE = r'C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y'
IMG_DIR = os.path.join(BASE, 'valid', 'images')
LBL_DIR = os.path.join(BASE, 'valid', 'labels')
PLAN_DIR = os.path.join(BASE, 'annotation_audit_results', 'correction_plan')
SHEETS_DIR = os.path.join(PLAN_DIR, 'review_sheets')
os.makedirs(SHEETS_DIR, exist_ok=True)

CLASS_INFO = {
    0: ('Helmet', (0, 255, 0)),        # Green
    1: ('NoHelmet', (255, 0, 0)),      # Red
    2: ('NoVest', (255, 140, 0)),      # Orange
    3: ('Vest', (0, 191, 255))         # Cyan
}

plan_data = {
    '66_jpg.rf.59fa725d703fa7afc7f45d05ac92f27c.jpg': {
        'action': 'EXCLUDE_IMAGE',
        'proposed_boxes': [],
        'reason': 'Image depicts patrons seated at an indoor restaurant/hotpot dining table. Complete out-of-domain error.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    },
    '00838_jpg.rf.9df07396df1ffd66cfeb46e4cdc5a44a.jpg': {
        'action': 'ADD_BOXES',
        'proposed_boxes': [
            (0, 'Helmet', 0.2958, 0.2321, 0.0840, 0.0942, 'Worker 1 (Left turnstile) white hard hat', 'HIGH', 'NO'),
            (0, 'Helmet', 0.5324, 0.2321, 0.0840, 0.1006, 'Worker 2 (Center turnstile) white hard hat', 'HIGH', 'NO'),
            (0, 'Helmet', 0.8559, 0.2256, 0.0897, 0.1136, 'Worker 3 (Right turnstile) white hard hat', 'HIGH', 'NO'),
            (2, 'NoVest', 0.8426, 0.4870, 0.1660, 0.3117, 'Worker 3 (Right turnstile) green jacket / torso without reflective vest', 'MEDIUM', 'YES')
        ],
        'reason': 'Add 3 missing Helmet annotations for workers visibly wearing white hard hats at access turnstile. Review whether Worker 3 also receives NoVest.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'YES (Worker 3 NoVest)'
    },
    '00587_jpg.rf.2d89a904ea1815c04f6835aa6516d210.jpg': {
        'action': 'ADD_BOXES',
        'proposed_boxes': [
            (3, 'Vest', 0.2880, 0.6547, 0.2760, 0.6426, 'Technician 1 (Left) bright yellow reflective safety vest', 'HIGH', 'NO'),
            (3, 'Vest', 0.6060, 0.6502, 0.2280, 0.6126, 'Technician 2 (Right) bright yellow reflective safety vest', 'HIGH', 'NO')
        ],
        'reason': 'Add 2 Vest annotations for two railway survey technicians prominently wearing fluorescent reflective safety vests.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    },
    '03128_jpg.rf.1e6d990894e8aef4b65b19dc5f0990f9.jpg': {
        'action': 'ADD_BOXES',
        'proposed_boxes': [
            (0, 'Helmet', 0.4958, 0.5117, 0.0750, 0.1016, 'Foreground worker illuminated halo hard hat', 'HIGH', 'NO')
        ],
        'reason': 'Add missing Helmet annotation for foreground worker wearing illuminated safety hard hat. Note: background equipment operators left unannotated per resolution/distance threshold.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    },
    'helmet_jacket_00831_jpg.rf.c6eb0b134e84d9ae9605e972d5b7a894.jpg': {
        'action': 'ADD_BOXES',
        'proposed_boxes': [
            (0, 'Helmet', 0.5141, 0.1117, 0.2594, 0.1453, 'White industrial safety hard hat worn and held by model', 'HIGH', 'NO')
        ],
        'reason': 'Add missing Helmet annotation on clear stock photo model where only the vest was previously annotated.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    },
    'helmet_jacket_00885_jpg.rf.a0162cb17401caa7503e16d7d3553f03.jpg': {
        'action': 'ADD_BOXES',
        'proposed_boxes': [
            (0, 'Helmet', 0.6023, 0.1367, 0.2516, 0.1578, 'Blue industrial safety helmet worn by catalog model', 'HIGH', 'NO')
        ],
        'reason': 'Add missing Helmet annotation on catalog model where only the vest was previously annotated.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    },
    '02673_jpg.rf.e2f1414b99d3c6a630cd1f83d41cf348.jpg': {
        'action': 'POLICY_REVIEW',
        'proposed_boxes': [],
        'reason': 'Contains 2 NoHelmet annotations on visitor delegation members in business attire. Do NOT automatically remove; hold for human policy decision on whether site visitors require PPE.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'YES'
    },
    '12_jpg.rf.b835e726e69aac2a501c377d8efe151a.jpg': {
        'action': 'POLICY_REVIEW',
        'proposed_boxes': [],
        'reason': 'Contains NoHelmet annotation on a pedestrian walking past street works in a straw sunhat. Do NOT automatically remove; hold for human policy decision on pedestrian exclusion vs strict full-frame tagging.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'YES'
    },
    '00718_jpg.rf.5f4eb3952834c71021471805ba96f030.jpg': {
        'action': 'VERIFIED_NO_CHANGE',
        'proposed_boxes': [],
        'reason': 'Second-pass cropped pixel inspection confirmed Box #2 is genuinely a white safety hard hat, not a baseball cap. All 4 existing boxes verified correct; no changes proposed.',
        'confidence': 'HIGH',
        'requires_manual_approval': 'NO'
    }
}

for fname, pdata in plan_data.items():
    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0]+'.txt')
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    panel_orig = img.copy()
    panel_prop = img.copy()
    draw_orig = ImageDraw.Draw(panel_orig)
    draw_prop = ImageDraw.Draw(panel_prop)
    
    orig_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    orig_boxes.append((cls, cx, cy, bw, bh))
    
    # Draw original on panel_orig
    for cls, cx, cy, bw, bh in orig_boxes:
        cname, col = CLASS_INFO[cls]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        draw_orig.rectangle([x1, y1, x2, y2], outline=col, width=3)
        lbl_str = f'{cname}'
        draw_orig.rectangle([x1, max(0, y1-16), x1+len(lbl_str)*9+4, max(0, y1-16)+16], fill=col)
        draw_orig.text((x1+2, max(0, y1-16)+1), lbl_str, fill=(0,0,0))

    # Draw proposed on panel_prop
    if pdata['action'] == 'EXCLUDE_IMAGE':
        draw_prop.rectangle([0, int(h*0.4), w, int(h*0.6)], fill=(200, 0, 0))
        draw_prop.text((int(w*0.15), int(h*0.48)), 'ACTION: PROPOSED FOR EXCLUSION (OUT OF DOMAIN)', fill=(255, 255, 255))
    else:
        for cls, cx, cy, bw, bh in orig_boxes:
            cname, col = CLASS_INFO[cls]
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            draw_prop.rectangle([x1, y1, x2, y2], outline=col, width=2)
            draw_prop.text((x1+2, max(0, y1-16)+1), cname, fill=col)
        
        for pcls, pcname, pcx, pcy, pbw, pbh, pnote, pconf, papp in pdata['proposed_boxes']:
            px1 = int((pcx - pbw/2) * w)
            py1 = int((pcy - pbh/2) * h)
            px2 = int((pcx + pbw/2) * w)
            py2 = int((pcy + pbh/2) * h)
            p_col = (255, 0, 255) if papp == 'NO' else (255, 215, 0)
            draw_prop.rectangle([px1, py1, px2, py2], outline=p_col, width=4)
            tag = f'+ {pcname} (NEEDS APPROVAL)' if papp == 'YES' else f'+ {pcname}'
            draw_prop.rectangle([px1, max(0, py1-18), px1+len(tag)*8+6, max(0, py1-18)+18], fill=p_col)
            draw_prop.text((px1+3, max(0, py1-18)+2), tag, fill=(0,0,0))
            
        if pdata['action'] == 'POLICY_REVIEW':
            draw_prop.rectangle([0, 0, w, int(h*0.12)], fill=(255, 140, 0))
            draw_prop.text((15, int(h*0.04)), 'HELD FOR HUMAN POLICY REVIEW (NO AUTO REMOVAL)', fill=(0,0,0))
        elif pdata['action'] == 'VERIFIED_NO_CHANGE':
            draw_prop.rectangle([0, 0, w, int(h*0.12)], fill=(0, 180, 0))
            draw_prop.text((15, int(h*0.04)), 'VERIFIED CORRECT - NO CHANGES PROPOSED', fill=(255,255,255))

    banner_h = 40
    sheet = Image.new('RGB', (w*2 + 20, h + banner_h), (25, 25, 25))
    draw_sheet = ImageDraw.Draw(sheet)
    
    act = pdata['action']
    draw_sheet.text((20, 12), f'ORIGINAL: {fname[:30]}... ({len(orig_boxes)} boxes)', fill=(220, 220, 220))
    draw_sheet.text((w + 30, 12), f'PROPOSED: {act} (+{len(pdata["proposed_boxes"])} new boxes)', fill=(255, 215, 0))
    
    sheet.paste(panel_orig, (10, banner_h))
    sheet.paste(panel_prop, (w + 20, banner_h))
    
    out_file = os.path.join(SHEETS_DIR, f'review_{fname[:18]}.jpg')
    sheet.save(out_file, quality=92)
    print(f'Generated review sheet: {out_file}')

print('All review sheets generated successfully.')
