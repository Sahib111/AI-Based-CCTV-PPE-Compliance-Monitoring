import streamlit as st
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PPE Safety Monitoring",
    page_icon="🦺",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# dashboard.py is inside runs/detect/
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

INCIDENT_LOG = os.path.join(
    BASE_DIR,
    "incident_log.csv"
)

SNAPSHOT_DIR = os.path.join(
    BASE_DIR,
    "incident_snapshots"
)

OUTPUT_VIDEO = os.path.join(
    BASE_DIR,
    "ppe_compliance_output.avi"
)


# ============================================================
# HEADER
# ============================================================

st.title("🦺 AI-Based CCTV PPE Safety Monitoring")

st.caption(
    "AI-Based CCTV PPE Compliance Monitoring for Industrial Safety Zones"
)

st.divider()


# ============================================================
# LOAD INCIDENT DATA
# ============================================================

if os.path.exists(INCIDENT_LOG):

    try:
        df = pd.read_csv(INCIDENT_LOG)

    except Exception:
        df = pd.DataFrame()

else:

    df = pd.DataFrame()


required_columns = [
    "Incident_ID",
    "Timestamp",
    "Worker_ID",
    "Violation_Type",
    "Violation_Duration_Seconds",
    "Zone",
    "Snapshot"
]

for column in required_columns:

    if column not in df.columns:

        df[column] = pd.Series(
            dtype="object"
        )


if not df.empty:

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_incidents = len(df)

unique_workers = (
    df["Worker_ID"].nunique()
    if not df.empty
    else 0
)

helmet_violations = (
    df["Violation_Type"]
    .str.contains(
        "Helmet",
        case=False,
        na=False
    )
    .sum()
    if not df.empty
    else 0
)

vest_violations = (
    df["Violation_Type"]
    .str.contains(
        "Vest",
        case=False,
        na=False
    )
    .sum()
    if not df.empty
    else 0
)

average_duration = (
    df["Violation_Duration_Seconds"]
    .mean()
    if not df.empty
    else 0
)


# ============================================================
# SAFETY OVERVIEW
# ============================================================

st.subheader("Safety Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Confirmed Incidents",
    total_incidents
)

col2.metric(
    "Workers Involved",
    unique_workers
)

col3.metric(
    "Helmet Violations",
    helmet_violations
)

col4.metric(
    "Vest Violations",
    vest_violations
)

st.divider()


# ============================================================
# MONITORING STATUS
# ============================================================

st.subheader("Monitoring Status")

status1, status2, status3, status4 = st.columns(4)

with status1:

    st.success(
        "🟢 AI Monitoring Active"
    )

with status2:

    st.success(
        "🟢 Safety Zone Detection Active"
    )

with status3:

    st.success(
        "🟢 2-Second Persistence Active"
    )

with status4:

    st.success(
        "🟢 Incident Logging Active"
    )

st.divider()


# ============================================================
# CCTV MONITORING
# ============================================================

st.subheader("🎥 CCTV Monitoring")

if os.path.exists(OUTPUT_VIDEO):

    try:

        st.video(
            OUTPUT_VIDEO,
            format="video/avi"
        )

    except Exception:

        st.warning(
            "The processed AVI video could not be "
            "previewed directly by the browser."
        )

        st.code(
            OUTPUT_VIDEO
        )

else:

    st.warning(
        "Processed CCTV output video was not found."
    )

st.caption(
    "Processed CCTV output showing worker detection, "
    "PPE detection, worker tracking, safety-zone "
    "evaluation and confirmed incident detection."
)

st.divider()


# ============================================================
# INCIDENT ANALYTICS
# ============================================================

st.subheader("📊 Incident Analytics")

if df.empty:

    st.info(
        "No confirmed PPE incidents were recorded."
    )

else:

    left, right = st.columns(2)

    # --------------------------------------------------------
    # VIOLATIONS BY TYPE
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Violations by Type"
        )

        violation_counts = (
            df["Violation_Type"]
            .value_counts()
        )

        st.bar_chart(
            violation_counts
        )

    # --------------------------------------------------------
    # INCIDENTS BY WORKER
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Incidents by Worker"
        )

        worker_counts = (
            df["Worker_ID"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            worker_counts
        )


    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    st.markdown(
        "### Incident Duration"
    )

    duration_data = df[
        [
            "Incident_ID",
            "Violation_Duration_Seconds"
        ]
    ].set_index("Incident_ID")

    st.bar_chart(
        duration_data
    )


st.divider()


# ============================================================
# INCIDENT LOG
# ============================================================

st.subheader("🚨 Incident Log")

if df.empty:

    st.info(
        "No confirmed incidents are currently recorded."
    )

else:

    display_df = df.copy()

    display_df["Timestamp"] = (
        display_df["Timestamp"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )


st.divider()


# ============================================================
# INCIDENT EVIDENCE
# ============================================================

st.subheader("📸 Incident Evidence")

if df.empty:

    st.info(
        "Incident snapshots will appear here when "
        "a PPE violation persists beyond the configured "
        "2-second threshold."
    )

else:

    valid_snapshots = []

    for snapshot in df[
        "Snapshot"
    ].dropna():

        snapshot_path = os.path.join(
            BASE_DIR,
            snapshot
        )

        if os.path.exists(
            snapshot_path
        ):

            valid_snapshots.append(
                snapshot_path
            )


    if valid_snapshots:

        cols = st.columns(
            min(
                3,
                len(valid_snapshots)
            )
        )

        for i, snapshot_path in enumerate(
            valid_snapshots
        ):

            with cols[
                i % len(cols)
            ]:

                st.image(
                    snapshot_path,
                    caption=os.path.basename(
                        snapshot_path
                    ),
                    width="stretch"
                )

    else:

        st.warning(
            "Incident records exist, but the "
            "corresponding snapshot files "
            "could not be found."
        )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📈 Model Performance")

st.caption(
    "Final PPE detector validation results — "
    "20-epoch YOLO11n model"
)


performance_data = pd.DataFrame(
    {
        "Class": [
            "Helmet",
            "No Helmet",
            "No Vest",
            "Vest"
        ],

        "Precision": [
            0.541,
            0.629,
            0.501,
            0.724
        ],

        "Recall": [
            0.767,
            0.832,
            0.646,
            0.813
        ],

        "mAP@50": [
            0.679,
            0.793,
            0.584,
            0.802
        ],

        "mAP@50-95": [
            0.478,
            0.593,
            0.394,
            0.561
        ]
    }
)

st.dataframe(
    performance_data.style.format(
        {
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "mAP@50": "{:.1%}",
            "mAP@50-95": "{:.1%}"
        }
    ),
    width="stretch",
    hide_index=True
)


# ============================================================
# OVERALL MODEL METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Precision",
        "59.9%"
    )

with metric2:

    st.metric(
        "Recall",
        "76.5%"
    )

with metric3:

    st.metric(
        "mAP@50",
        "71.4%"
    )

with metric4:

    st.metric(
        "mAP@50-95",
        "50.6%"
    )


# ============================================================
# MODEL INFERENCE PERFORMANCE
# ============================================================

st.markdown(
    "### Inference Performance"
)

speed1, speed2, speed3 = st.columns(3)

with speed1:

    st.metric(
        "Preprocessing",
        "0.6 ms/image"
    )

with speed2:

    st.metric(
        "Inference",
        "28.2 ms/image"
    )

with speed3:

    st.metric(
        "Postprocessing",
        "0.7 ms/image"
    )

st.caption(
    "Validation performed on 457 images containing "
    "1,176 annotated PPE instances."
)

st.divider()


# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

st.subheader("⚙️ System Configuration")

config1, config2, config3, config4 = st.columns(4)

with config1:

    st.markdown(
        "**Detection Model**"
    )

    st.write(
        "YOLO11n + PPE Detector"
    )

with config2:

    st.markdown(
        "**Tracking**"
    )

    st.write(
        "ByteTrack"
    )

with config3:

    st.markdown(
        "**Safety Zone**"
    )

    st.write(
        "Polygon-based PPE Zone"
    )

with config4:

    st.markdown(
        "**Persistence**"
    )

    st.write(
        "2 seconds"
    )


config5, config6, config7, config8 = st.columns(4)

with config5:

    st.markdown(
        "**PPE Classes**"
    )

    st.write(
        "Helmet + Vest"
    )

with config6:

    st.markdown(
        "**Input**"
    )

    st.write(
        "Controlled Validation CCTV"
    )

with config7:

    st.markdown(
        "**Processing**"
    )

    st.write(
        "CPU"
    )

with config8:

    st.markdown(
        "**Alert Output**"
    )

    st.write(
        "CSV + Snapshot"
    )

st.divider()


# ============================================================
# VALIDATION SUMMARY
# ============================================================

st.subheader("🧪 System Validation Summary")

validation1, validation2, validation3, validation4 = st.columns(4)

with validation1:

    st.markdown(
        "**Validation Video**"
    )

    st.write(
        "Controlled PPE evaluation"
    )

with validation2:

    st.markdown(
        "**Confirmed Incidents**"
    )

    st.write(
        f"{total_incidents}"
    )

with validation3:

    st.markdown(
        "**Persistence Threshold**"
    )

    st.write(
        "2 seconds"
    )

with validation4:

    st.markdown(
        "**Incident Snapshots**"
    )

    st.write(
        f"{len(valid_snapshots) if not df.empty else 0}"
    )


if not df.empty:

    st.success(
        f"System validation completed successfully: "
        f"{total_incidents} confirmed PPE incidents were "
        f"logged after applying the 2-second persistence rule."
    )

else:

    st.info(
        "No confirmed PPE incidents were recorded "
        "during this validation run."
    )


st.caption(
    "Note: the controlled validation video is used to "
    "demonstrate end-to-end incident detection. "
    "Model performance metrics are reported separately "
    "from the independent PPE dataset validation."
)


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI-Based CCTV PPE Compliance Monitoring Prototype | "
    "Industrial Safety Zone"
)