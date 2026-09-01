import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tensorflow.keras.models import load_model
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentinel AI | Activity & Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f111a;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #aab2c5;
        font-size: 17px;
        margin-top: 0;
    }

    .metric-card {
        background: linear-gradient(
            135deg,
            #1e2130 0%,
            #292d3f 100%
        );

        padding: 20px;
        border-radius: 15px;

        border-left: 5px solid #00f2fe;

        box-shadow:
            0 4px 15px rgba(0,0,0,0.30);
    }

    .metric-title {
        color: #aab2c5;
        font-size: 15px;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
    }

    .alert-card {
        background: linear-gradient(
            135deg,
            #ff4b4b 0%,
            #b00000 100%
        );

        padding: 22px;
        border-radius: 15px;

        color: white;
        text-align: center;

        font-size: 22px;
        font-weight: 800;

        box-shadow:
            0 5px 20px rgba(255, 0, 0, 0.30);
    }

    .safe-card {
        background: linear-gradient(
            135deg,
            #123c32 0%,
            #17594a 100%
        );

        padding: 18px;
        border-radius: 15px;

        color: white;
        text-align: center;

        font-size: 19px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = "yolov8n-pose.pt"
CLASSIFIER_PATH = "models/fall_detection_model.h5"

# IMPORTANT:
# This order MUST match the class order used while training
CLASS_NAMES = [
    "Falling",
    "Not Falling"
]

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_ai_system():

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    # Ultralytics will auto-download this if it's not already
    # cached locally — no need to store it in the repo.
    pose_model = YOLO(YOLO_MODEL_PATH)

    classifier = load_model(
        CLASSIFIER_PATH,
        compile=False
    )

    return pose_model, classifier


# ============================================================
# INITIALIZE MODELS
# ============================================================

try:

    yolo, classifier = load_ai_system()

except Exception as error:

    st.error("⚠️ AI system could not be loaded.")

    st.code(str(error))

    st.info(
        "Check that both model files exist inside the "
        "'models' folder."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

if "fall_events" not in st.session_state:
    st.session_state.fall_events = 0

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0


# ============================================================
# HELPER — RESET SESSION
# ============================================================

def reset_session():

    st.session_state.activity_log = []
    st.session_state.fall_events = 0
    st.session_state.frame_count = 0


# ============================================================
# POSE FEATURE EXTRACTION
# ============================================================
def extract_pose_features(result, frame_width, frame_height):

    if result.keypoints is None:
        return None

    if len(result.keypoints.data) == 0:
        return None

    keypoints = (
        result.keypoints.data[0]
        .cpu()
        .numpy()
    )

    if keypoints.shape[0] != 17:
        return None

    # Normalize x, y to 0-1 range using frame dimensions
    # Keep the 3rd column (confidence) unchanged
    keypoints[:, 0] = keypoints[:, 0] / frame_width
    keypoints[:, 1] = keypoints[:, 1] / frame_height

    flattened = keypoints.flatten()

    if flattened.shape[0] != 51:
        return None

    return flattened

# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(frame, threshold):

    results = yolo(
        frame,
        verbose=False
    )

    annotated = frame.copy()

    label = "No Person Detected"
    confidence = 0.0

    for result in results:

        features = extract_pose_features(
            result,
            frame.shape[1],  # width
            frame.shape[0]   # height
        )

        if features is None:
            continue

        # Draw YOLO pose visualization
        annotated = result.plot()

        prediction = classifier.predict(
            features.reshape(1, -1),
            verbose=0
        )

        prediction = np.asarray(prediction)

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[0][class_index]
        )

        if class_index < len(CLASS_NAMES):
            label = CLASS_NAMES[class_index]
        else:
            label = "Unknown"

        # --------------------------------------------------------
        # NEW: BOUNDING BOX OVERRIDE FOR SITTING FALSE POSITIVES
        # --------------------------------------------------------
        # Double-check that a bounding box actually exists in this frame
        i# --------------------------------------------------------
        # NEW: SMART KEYPOINT OVERRIDE
        # --------------------------------------------------------
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kp = result.keypoints.xy[0].cpu().numpy()
            
            # YOLO keypoint indices: 0 is Nose, 11 is Left Hip, 12 is Right Hip
            # Ensure the keypoints actually exist in the frame (Y > 0)
            if len(kp) >= 13 and kp[0][1] > 0 and kp[11][1] > 0:
                nose_y = kp[0][1]
                avg_hip_y = (kp[11][1] + kp[12][1]) / 2.0
                
                # If the nose is above the hips (smaller Y value), they are upright
                if label == "Falling" and (nose_y < avg_hip_y):
                    label = "Not Falling"
                    confidence = 0.99  # Force correction


    # --------------------------------------------------------
    # STATUS OVERLAY
    # --------------------------------------------------------

    if (
        label == "Falling"
        and confidence >= threshold
    ):

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 85),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated,
            f"FALL DETECTED | {confidence * 100:.1f}%",
            (20, 55),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (255, 255, 255),
            3
        )

    else:

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 65),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated,
            f"{label} | {confidence * 100:.1f}%",
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

    return annotated, label, confidence


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Sentinel AI")

    st.caption(
        "AI-powered activity and fall detection"
    )

    st.markdown("---")

    input_mode = st.radio(
        "Monitoring Source",
        [
            "Static Image",
            "Video Stream"
        ]
    )

    alert_threshold = st.slider(
        "Fall Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.05
    )

    st.markdown("---")

    st.markdown("### System Status")

    st.success("🟢 AI Engine Online")

    st.markdown(
        "**Pose Model:** YOLOv8 Pose"
    )

    st.markdown(
        "**Classifier:** Keras CNN"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Session Data",
        use_container_width=True
    ):

        reset_session()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🛡️ Sentinel AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Elderly Activity & Fall Detection Network'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Processed Frames
            </div>
            <div class="metric-value">
                {st.session_state.frame_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Logged Activities
            </div>
            <div class="metric-value">
                {len(st.session_state.activity_log)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Confirmed Fall Events
            </div>
            <div class="metric-value">
                {st.session_state.fall_events}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Alert Threshold
            </div>
            <div class="metric-value">
                {alert_threshold * 100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📷 Monitoring",
        "📈 Analytics",
        "📋 Logs & Export"
    ]
)


# ============================================================
# TAB 1 — MONITORING
# ============================================================

with tab1:

    st.subheader(
        f"Current Feed: {input_mode}"
    )

    # --------------------------------------------------------
    # STATIC IMAGE
    # --------------------------------------------------------

    if input_mode == "Static Image":

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        if uploaded_file:

            file_bytes = np.asarray(
                bytearray(
                    uploaded_file.read()
                ),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            processed, label, confidence = analyze_frame(
                image,
                alert_threshold
            )

            st.session_state.frame_count += 1

            st.session_state.activity_log.append(
                {
                    "Activity": label,
                    "Confidence": confidence,
                    "Source": "Image"
                }
            )

            if (
                label == "Falling"
                and confidence >= alert_threshold
            ):

                st.session_state.fall_events += 1

                st.markdown(
                    """
                    <div class="alert-card">
                        🚨 FALL DETECTED<br>
                        Immediate attention recommended
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif label != "No Person Detected":

                st.markdown(
                    f"""
                    <div class="safe-card">
                        ✓ {label}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Confidence: {confidence * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:

                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Original Input",
                    use_container_width=True
                )

            with c2:

                st.image(
                    cv2.cvtColor(
                        processed,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="AI Pose Analysis",
                    use_container_width=True
                )


    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    elif input_mode == "Video Stream":

        uploaded_video = st.file_uploader(
            "Upload a surveillance video",
            type=[
                "mp4",
                "avi",
                "mov"
            ]
        )

        if uploaded_video:

            temporary_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temporary_file.write(
                uploaded_video.read()
            )

            temporary_file.close()

            cap = cv2.VideoCapture(
                temporary_file.name
            )

            placeholder = st.empty()

            progress = st.progress(0)

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            video_fall_detected = False

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                processed, label, confidence = analyze_frame(
                    frame,
                    alert_threshold
                )

                st.session_state.frame_count += 1

                if label != "No Person Detected":

                    st.session_state.activity_log.append(
                        {
                            "Activity": label,
                            "Confidence": confidence,
                            "Source": "Video"
                        }
                    )

                if (
                    label == "Falling"
                    and confidence >= alert_threshold
                ):

                    video_fall_detected = True

                placeholder.image(
                    cv2.cvtColor(
                        processed,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB",
                    use_container_width=True
                )

                if total_frames > 0:

                    current_frame = int(
                        cap.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )

                    progress.progress(
                        min(
                            current_frame / total_frames,
                            1.0
                        )
                    )

            cap.release()

            progress.empty()

            if video_fall_detected:

                st.session_state.fall_events += 1

                st.markdown(
                    """
                    <div class="alert-card">
                        🚨 FALL EVENT DETECTED
                        <br>
                        Potential fall identified in uploaded video
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "✅ Video analysis completed. "
                    "No high-confidence fall event detected."
                )

            os.unlink(
                temporary_file.name
            )


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab2:

    st.subheader(
        "Behavioral Analytics & Trends"
    )

    if st.session_state.activity_log:

        df = pd.DataFrame(
            st.session_state.activity_log
        )

        c1, c2 = st.columns(2)

        with c1:

            fig_pie = px.pie(
                df,
                names="Activity",
                hole=0.45,
                title="Activity Distribution"
            )

            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        with c2:

            counts = (
                df["Activity"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Activity",
                "Count"
            ]

            fig_bar = px.bar(
                counts,
                x="Activity",
                y="Count",
                title="Detected Activity Counts",
                text="Count"
            )

            fig_bar.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        st.markdown("### Confidence Statistics")

        avg_confidence = df[
            "Confidence"
        ].mean()

        max_confidence = df[
            "Confidence"
        ].max()

        c1, c2 = st.columns(2)

        c1.metric(
            "Average Confidence",
            f"{avg_confidence * 100:.2f}%"
        )

        c2.metric(
            "Maximum Confidence",
            f"{max_confidence * 100:.2f}%"
        )

    else:

        st.info(
            "Upload an image or video to generate analytics."
        )


# ============================================================
# TAB 3 — LOGS & EXPORT
# ============================================================

with tab3:

    st.subheader(
        "System Logs & Incident Reports"
    )

    if st.session_state.activity_log:

        log_df = pd.DataFrame(
            st.session_state.activity_log
        )

        display_df = log_df.copy()

        display_df["Confidence"] = (
            display_df["Confidence"] * 100
        ).round(2)

        display_df.rename(
            columns={
                "Confidence": "Confidence (%)"
            },
            inplace=True
        )

        st.dataframe(
            display_df.tail(30),
            use_container_width=True
        )

        csv_data = log_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Activity Report",
            data=csv_data,
            file_name="sentinel_activity_log.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No activity has been recorded yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Sentinel AI | Educational AI Healthcare Monitoring Prototype"
)
