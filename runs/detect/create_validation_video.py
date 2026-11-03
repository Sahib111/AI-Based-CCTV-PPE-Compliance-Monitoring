import cv2
from ultralytics import YOLO

INPUT_VIDEO = r"C:\Users\Sahib Singh\Downloads\ppe_test.mp4.mp4"
OUTPUT_VIDEO = r"C:\Users\Sahib Singh\Downloads\ppe_validation.mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)

print("Opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError("Could not open source video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)

cap.release()

print("Video:", width, "x", height)
print("FPS:", fps)
print("Frames:", len(frames))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Could not create output video")

conditions = [
    "COMPLIANT",
    "HELMET_MISSING",
    "VEST_MISSING",
    "BOTH_MISSING",
    "COMPLIANT"
]

for segment_number, condition in enumerate(conditions, 1):

    print(
        f"Creating segment {segment_number}/5: {condition}"
    )

    for frame in frames:

        output_frame = frame.copy()

        results = model(
            frame,
            classes=[0],
            conf=0.35,
            verbose=False
        )

        if results[0].boxes is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                person_height = y2 - y1

                if person_height <= 0:
                    continue

                helmet_y1 = y1
                helmet_y2 = y1 + int(person_height * 0.30)

                vest_y1 = y1 + int(person_height * 0.25)
                vest_y2 = y1 + int(person_height * 0.65)

                if condition in [
                    "HELMET_MISSING",
                    "BOTH_MISSING"
                ]:

                    cv2.rectangle(
                        output_frame,
                        (x1, helmet_y1),
                        (x2, helmet_y2),
                        (80, 80, 80),
                        -1
                    )

                if condition in [
                    "VEST_MISSING",
                    "BOTH_MISSING"
                ]:

                    cv2.rectangle(
                        output_frame,
                        (x1, vest_y1),
                        (x2, vest_y2),
                        (80, 80, 80),
                        -1
                    )

        cv2.rectangle(
            output_frame,
            (20, 20),
            (750, 90),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output_frame,
            f"CONTROLLED VALIDATION: {condition}",
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        out.write(output_frame)

out.release()

duration = len(frames) * len(conditions) / fps

print()
print("======================================")
print("VALIDATION VIDEO CREATED")
print("======================================")
print(OUTPUT_VIDEO)
print(f"Duration: {duration:.1f} seconds")