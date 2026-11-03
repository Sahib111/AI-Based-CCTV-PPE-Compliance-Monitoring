from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
import json
import argparse
from pathlib import Path


# ============================================================
# CONFIGURATION & PARAMETERS
# ============================================================

parser = argparse.ArgumentParser(description="AI-Based CCTV PPE Compliance Monitoring")
parser.add_argument("--config", type=str, default="ppe_pipeline_config.json", help="Path to config JSON")
parser.add_argument("--ppe-model", type=str, default=None, help="Override PPE model weights path")
parser.add_argument("--video-path", type=str, default=None, help="Override input video path")
parser.add_argument("--output-path", type=str, default=None, help="Override output video path")
parser.add_argument("--incident-log", type=str, default=None, help="Override incident log CSV path")
parser.add_argument("--snapshot-folder", type=str, default=None, help="Override snapshot folder")
parser.add_argument("--conf", type=float, default=None, help="Confidence threshold for PPE model")
parser.add_argument("--iou", type=float, default=None, help="IoU NMS threshold for PPE model")
parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
args, unknown = parser.parse_known_args()

# Search for config file in current directory or project root
config_data = {}
config_file = Path(args.config)
if not config_file.exists():
    root_candidate = Path(__file__).resolve().parent.parent.parent / "ppe_pipeline_config.json"
    if root_candidate.exists():
        config_file = root_candidate

if config_file.exists():
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        print(f"Loaded pipeline configuration from: {config_file}")
    except Exception as e:
        print(f"Warning: Failed to parse {config_file}: {e}")

PERSON_MODEL = config_data.get("person_model", "yolo11n.pt")
PPE_MODEL = args.ppe_model or config_data.get("ppe_model", "runs/detect/ppe_v2/weights/best.pt")

default_vid = r"C:\Users\Sahib Singh\Downloads\ppe_validation.mp4"
if not os.path.exists(default_vid) and os.path.exists("ppe_validation_slow.mp4"):
    default_vid = "ppe_validation_slow.mp4"

VIDEO_PATH = args.video_path or config_data.get("video_path", default_vid)
if not os.path.exists(VIDEO_PATH) and os.path.exists(config_data.get("fallback_video_path", "")):
    VIDEO_PATH = config_data.get("fallback_video_path")

OUTPUT_PATH = args.output_path or config_data.get("output_path", "ppe_compliance_output.avi")
INCIDENT_LOG = args.incident_log or config_data.get("incident_log", "incident_log.csv")
SNAPSHOT_FOLDER = args.snapshot_folder or config_data.get("snapshot_folder", "incident_snapshots")
PERSISTENCE_SECONDS = float(config_data.get("persistence_seconds", 2.0))
CONF_THRESHOLD = args.conf if args.conf is not None else float(config_data.get("conf_threshold", 0.25))
IOU_THRESHOLD = args.iou if args.iou is not None else float(config_data.get("iou_threshold", 0.50))
MAX_FRAMES = args.max_frames

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

raw_zone = config_data.get("safety_zone", [
    [500, 1650],
    [1650, 1650],
    [1950, 3400],
    [250, 3400]
])
ZONE = np.array(raw_zone, np.int32)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading person model...")
person_model = YOLO(PERSON_MODEL)

print("Loading PPE model...")
ppe_model = YOLO(PPE_MODEL)

print("Models loaded.")


# ============================================================
# PPE CLASS IDs
# ============================================================

HELMET = 0
NO_HELMET = 1
NO_VEST = 2
VEST = 3


# ============================================================
# VIDEO SETUP
# ============================================================

print("Opening video:")
print(VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Frames: {total_frames}")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# CREATE INCIDENT CSV
# ============================================================

if not os.path.exists(INCIDENT_LOG):

    with open(
        INCIDENT_LOG,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Incident_ID",
            "Timestamp",
            "Worker_ID",
            "Violation_Type",
            "Violation_Duration_Seconds",
            "Zone",
            "Snapshot"
        ])


# ============================================================
# VIOLATION TRACKING
# ============================================================

# Stores when a violation started
violation_start_time = {}

# Stores whether the violation
# has already been logged
violation_logged = {}

# Incident counter
incident_counter = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    if MAX_FRAMES is not None and frame_number > MAX_FRAMES:
        print(f"Reached MAX_FRAMES limit ({MAX_FRAMES}). Stopping video processing.")
        break

    current_time = time.time()


    # ========================================================
    # DRAW SAFETY ZONE
    # ========================================================

    cv2.polylines(
        frame,
        [ZONE],
        isClosed=True,
        color=(255, 255, 0),
        thickness=8
    )

    cv2.putText(
        frame,
        "PPE SAFETY ZONE",
        (
            ZONE[0][0],
            ZONE[0][1] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 0),
        3
    )


    # ========================================================
    # TRACK PEOPLE
    # ========================================================

    person_results = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.35,
        verbose=False
    )


    # ========================================================
    # DETECT PPE
    # ========================================================

    ppe_results = ppe_model(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )


    # ========================================================
    # EXTRACT TRACKED PEOPLE
    # ========================================================

    people = []

    result = person_results[0]

    if (
        result.boxes is not None
        and result.boxes.is_track
    ):

        boxes = result.boxes

        for i in range(len(boxes)):

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i]
            )

            confidence = float(
                boxes.conf[i]
            )

            track_id = int(
                boxes.id[i]
            )

            people.append({
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence": confidence,
                "track_id": track_id
            })


    # ========================================================
    # EXTRACT PPE
    # ========================================================

    ppe_objects = []

    for box in ppe_results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        ppe_objects.append({
            "box": (
                x1,
                y1,
                x2,
                y2
            ),
            "confidence": confidence,
            "class_id": class_id
        })


    # ========================================================
    # PROCESS EACH WORKER
    # ========================================================

    for person in people:

        px1, py1, px2, py2 = person["box"]

        track_id = person["track_id"]


        # ====================================================
        # PERSON POSITION
        # ====================================================

        person_x = (
            px1 + px2
        ) // 2

        person_y = py2


        # ====================================================
        # CHECK SAFETY ZONE
        # ====================================================

        inside_zone = (
            cv2.pointPolygonTest(
                ZONE,
                (
                    person_x,
                    person_y
                ),
                False
            ) >= 0
        )


        # ====================================================
        # OUTSIDE SAFETY ZONE
        # ====================================================

        if not inside_zone:

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (255, 255, 255),
                4
            )

            cv2.putText(
                frame,
                f"Worker ID: {track_id}",
                (
                    px1,
                    max(
                        py1 - 55,
                        40
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "OUTSIDE SAFETY ZONE",
                (
                    px1,
                    max(
                        py1 - 20,
                        40
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3
            )

            # Reset violation state
            violation_start_time.pop(
                track_id,
                None
            )

            violation_logged.pop(
                track_id,
                None
            )

            continue


        # ====================================================
        # PPE STATUS
        # ====================================================

        helmet_found = False
        vest_found = False

        helmet_confidence = 0.0
        vest_confidence = 0.0


        # ====================================================
        # ASSOCIATE PPE WITH WORKER
        # ====================================================

        for ppe in ppe_objects:

            x1, y1, x2, y2 = ppe["box"]

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            inside_person = (
                px1 <= center_x <= px2
                and
                py1 <= center_y <= py2
            )

            if not inside_person:
                continue


            # Helmet
            if ppe["class_id"] == HELMET:

                helmet_found = True

                helmet_confidence = max(
                    helmet_confidence,
                    ppe["confidence"]
                )


            # Vest
            elif ppe["class_id"] == VEST:

                vest_found = True

                vest_confidence = max(
                    vest_confidence,
                    ppe["confidence"]
                )


        # ====================================================
        # COMPLIANCE STATUS
        # ====================================================

        if helmet_found and vest_found:

            status = "COMPLIANT"

            status_color = (
                0,
                255,
                0
            )

            # Worker is compliant again
            violation_start_time.pop(
                track_id,
                None
            )

            violation_logged.pop(
                track_id,
                None
            )


        else:

            # =================================================
            # IDENTIFY VIOLATION
            # =================================================

            if (
                not helmet_found
                and
                not vest_found
            ):

                violation_type = (
                    "Helmet Missing + Vest Missing"
                )

            elif not helmet_found:

                violation_type = (
                    "Helmet Missing"
                )

            else:

                violation_type = (
                    "Vest Missing"
                )


            status = violation_type


            # =================================================
            # START VIOLATION TIMER
            # =================================================

            if (
                track_id
                not in violation_start_time
            ):

                violation_start_time[
                    track_id
                ] = current_time


            # =================================================
            # CALCULATE DURATION
            # =================================================

            violation_duration = (
                current_time
                -
                violation_start_time[
                    track_id
                ]
            )


            # =================================================
            # CHECK PERSISTENCE
            # =================================================

            if (
                violation_duration
                >=
                PERSISTENCE_SECONDS
            ):

                status = (
                    "CONFIRMED: "
                    +
                    violation_type
                )

                status_color = (
                    0,
                    0,
                    255
                )


                # =============================================
                # LOG ONLY ONCE
                # =============================================

                if not violation_logged.get(
                    track_id,
                    False
                ):

                    incident_counter += 1

                    timestamp = (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )


                    # =========================================
                    # SNAPSHOT NAME
                    # =========================================

                    snapshot_name = (
                        f"incident_"
                        f"{incident_counter}_"
                        f"worker_{track_id}.jpg"
                    )


                    snapshot_path = os.path.join(
                        SNAPSHOT_FOLDER,
                        snapshot_name
                    )


                    # =========================================
                    # DRAW CONFIRMED INCIDENT
                    # =========================================

                    cv2.rectangle(
                        frame,
                        (px1, py1),
                        (px2, py2),
                        (0, 0, 255),
                        6
                    )

                    cv2.putText(
                        frame,
                        "CONFIRMED INCIDENT",
                        (
                            px1,
                            max(
                                py1 - 90,
                                40
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )


                    # =========================================
                    # SAVE SNAPSHOT
                    # =========================================

                    snapshot_saved = cv2.imwrite(
                        snapshot_path,
                        frame
                    )


                    # =========================================
                    # WRITE CSV RECORD
                    # =========================================

                    if snapshot_saved:

                        with open(
                            INCIDENT_LOG,
                            "a",
                            newline="",
                            encoding="utf-8"
                        ) as file:

                            writer = csv.writer(file)

                            writer.writerow([
                                incident_counter,
                                timestamp,
                                track_id,
                                violation_type,
                                round(
                                    violation_duration,
                                    2
                                ),
                                "PPE Safety Zone",
                                snapshot_path
                            ])

                            file.flush()


                        # Mark as logged
                        violation_logged[
                            track_id
                        ] = True

                        print(
                            f"INCIDENT {incident_counter}: "
                            f"Worker {track_id} - "
                            f"{violation_type} - "
                            f"{violation_duration:.2f}s"
                        )

                    else:

                        print(
                            "WARNING: "
                            "Could not save incident snapshot."
                        )


            else:

                status = (
                    f"CHECKING: "
                    f"{violation_type}"
                )

                status_color = (
                    0,
                    165,
                    255
                )


        # ====================================================
        # DRAW WORKER BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            status_color,
            4
        )


        # ====================================================
        # WORKER ID
        # ====================================================

        cv2.putText(
            frame,
            f"Worker ID: {track_id}",
            (
                px1,
                max(
                    py1 - 55,
                    40
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3
        )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            frame,
            status,
            (
                px1,
                max(
                    py1 - 20,
                    40
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3
        )


        # ====================================================
        # ZONE POINT
        # ====================================================

        cv2.circle(
            frame,
            (
                person_x,
                person_y
            ),
            10,
            status_color,
            -1
        )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    out.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print()
print("======================================")
print("PROCESSING COMPLETE")
print("======================================")
print(f"Output video: {OUTPUT_PATH}")
print(f"Incident log: {INCIDENT_LOG}")
print(f"Snapshots: {SNAPSHOT_FOLDER}")
print(f"Total incidents: {incident_counter}")