import os
import tempfile
import time
import math

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
        background: linear-gradient(135deg, #1e2130 0%, #292d3f 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00f2fe;
        box-shadow: 0 4px 15px rgba(0,0,0,0.30);
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
        background: linear-gradient(135deg, #ff4b4b 0%, #b00000 100%);
        padding: 22px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 5px 20px rgba(255, 0, 0, 0.30);
    }
    .safe-card {
        background: linear-gradient(135deg, #123c32 0%, #17594a 100%);
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
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")

    pose_model = YOLO(YOLO_MODEL_PATH)
    classifier = load_model(CLASSIFIER_PATH, compile=False)
    return pose_model, classifier


try:
    yolo, classifier = load_ai_system()
except Exception as error:
    st.error("⚠️ AI system could not be loaded.")
    st.code(str(error))
    st.info("Check that both model files exist inside the 'models' folder.")
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


def reset_session():
    st.session_state.activity_log = []
    st.session_state.fall_events = 0
    st.session_state.frame_count = 0


def extract_pose_features(result, frame_width, frame_height):
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    keypoints = result.keypoints.data[0].cpu().numpy()
    if keypoints.shape[0] != 17:
        return None

    keypoints[:, 0] = keypoints[:, 0] / frame_width
    keypoints[:, 1] = keypoints[:, 1] / frame_height

    flattened = keypoints.flatten()
    if flattened.shape[0] != 51:
        return None
    return flattened


# ============================================================
# FRAME ANALYSIS (WITH VECTOR TRIGONOMETRY)
# ============================================================

def analyze_frame(frame, threshold, show_diagnostics=False):
    results = yolo(frame, verbose=False)
    annotated = frame.copy()
    label = "No Person Detected"
    confidence = 0.0
    spine_angle = None

    for result in results:
        features = extract_pose_features(result, frame.shape[1], frame.shape[0])
        if features is None:
            continue

        annotated = result.plot()
        
        # Base AI Prediction
        prediction = classifier.predict(features.reshape(1, -1), verbose=0)
        prediction = np.asarray(prediction)
        class_index = int(np.argmax(prediction))
        confidence = float(prediction[0][class_index])

        if class_index < len(CLASS_NAMES):
            label = CLASS_NAMES[class_index]
        else:
            label = "Unknown"

        # --------------------------------------------------------
        # ADVANCED VECTOR POSTURE CALIBRATION
        # --------------------------------------------------------
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kp = result.keypoints.xy[0].cpu().numpy()
            
            # Ensure we have shoulders (5, 6) and hips (11, 12)
            if len(kp) > 12 and kp[5][1] > 0 and kp[6][1] > 0 and kp[11][1] > 0 and kp[12][1] > 0:
                mid_shoulder_x = (kp[5][0] + kp[6][0]) / 2.0
                mid_shoulder_y = (kp[5][1] + kp[6][1]) / 2.0
                
                mid_hip_x = (kp[11][0] + kp[12][0]) / 2.0
                mid_hip_y = (kp[11][1] + kp[12][1]) / 2.0
                
                # Calculate vector trajectory of the spine
                dx = mid_hip_x - mid_shoulder_x
                dy = mid_hip_y - mid_shoulder_y
                
                # Convert to degrees (90 = vertical, 0/180 = horizontal)
                spine_angle = abs(math.degrees(math.atan2(dy, dx)))
                
                # RULE 1: Standing/Sitting Verticality Check (Between 45 and 135 degrees)
                if 45 <= spine_angle <= 135:
                    label = "Not Falling"
                    confidence = 0.99  # Absolute certainty based on math
                
                # RULE 2: Fallen Horizontal Check (Under 35 or over 145 degrees)
                elif spine_angle < 35 or spine_angle > 145:
                    label = "Falling"
                    confidence = 0.98

        # Diagnostic Overlay for Presentation
        if show_diagnostics and spine_angle is not None:
            cv2.putText(annotated, f"Spine Vector: {spine_angle:.1f} deg", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        break

    # --------------------------------------------------------
    # STATUS OVERLAY
    # --------------------------------------------------------
    if label == "Falling" and confidence >= threshold:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 85), (0, 0, 255), -1)
        cv2.putText(annotated, f"FALL DETECTED | {confidence * 100:.1f}%", (20, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)
    else:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 65), (20, 20, 20), -1)
        cv2.putText(annotated, f"{label} | {confidence * 100:.1f}%", (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return annotated, label, confidence


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ Sentinel AI")
    st.caption("AI-powered activity and fall detection")
    st.markdown("---")

    input_mode = st.radio("Monitoring Source", ["Static Image", "Video Stream"])
    alert_threshold = st.slider("Fall Confidence Threshold", 0.50, 0.99, 0.75, 0.05)
    
    st.markdown("---")
    st.markdown("### Advanced Settings")
    dev_mode = st.checkbox("⚙️ Developer Diagnostic Overlay", value=False, help="Displays raw mathematical vectors on the feed.")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("🟢 AI Engine Online")
    st.markdown("**Pose Model:** YOLOv8 Pose")
    st.markdown("**Classifier:** Keras CNN + Vector Alg.")
    st.markdown("---")

    if st.button("🗑️ Clear Session Data", use_container_width=True):
        reset_session()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="main-title">🛡️ Sentinel AI</p>', unsafe_allow_html=True)

with st.expander("ℹ️ Project Abstract & System Architecture", expanded=False):
    st.markdown("""
    **Objective:** A real-time, non-intrusive monitoring system designed to detect fall events in elderly care environments using computer vision.
    
    **Architecture:** 
    * **Feature Extraction:** YOLOv8 Pose Estimation (51-point skeletal mapping).
    * **Classification:** Custom Keras Convolutional Neural Network.
    * **Edge Logic (Algorithmic Calibration):** Dynamic mathematical calculation of spine trajectories using vector trigonometry to eliminate false positives in diverse postures.
    """)

st.markdown("---")


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Processed Frames</div><div class="metric-value">{st.session_state.frame_count}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Logged Activities</div><div class="metric-value">{len(st.session_state.activity_log)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Confirmed Fall Events</div><div class="metric-value">{st.session_state.fall_events}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Alert Threshold</div><div class="metric-value">{alert_threshold * 100:.0f}%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["📷 Monitoring", "📈 Analytics", "📋 Logs & Export"])


# ============================================================
# TAB 1 — MONITORING
# ============================================================

with tab1:
    st.subheader(f"Current Feed: {input_mode}")

    if input_mode == "Static Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.toast("Image loaded. Initializing neural network...", icon="🧠") 
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            processed, label, confidence = analyze_frame(image, alert_threshold, show_diagnostics=dev_mode)
            st.session_state.frame_count += 1
            
            st.session_state.activity_log.append({
                "Activity": label,
                "Confidence": confidence,
                "Source": "Image"
            })

            if label == "Falling" and confidence >= alert_threshold:
                st.session_state.fall_events += 1
                
                st.markdown("""
                    <audio autoplay>
                        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
                    </audio>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="alert-card">
                        🚨 FALL DETECTED<br>
                        Immediate attention recommended
                    </div>
                """, unsafe_allow_html=True)
            elif label != "No Person Detected":
                st.markdown(f"""
                    <div class="safe-card">
                        ✓ {label} &nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {confidence * 100:.1f}%
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Input", use_container_width=True)
            with c2:
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="AI Pose Analysis", use_container_width=True)

    elif input_mode == "Video Stream":
        uploaded_video = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov"])

        if uploaded_video:
            st.toast("Video stream loaded. Analyzing frames...", icon="🧠") 
            
            temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temporary_file.write(uploaded_video.read())
            temporary_file.close()

            cap = cv2.VideoCapture(temporary_file.name)
            placeholder = st.empty()
            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fall_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed, label, confidence = analyze_frame(frame, alert_threshold, show_diagnostics=dev_mode)
                st.session_state.frame_count += 1

                if label != "No Person Detected":
                    st.session_state.activity_log.append({
                        "Activity": label,
                        "Confidence": confidence,
                        "Source": "Video"
                    })

                if label == "Falling" and confidence >= alert_threshold:
                    video_fall_detected = True

                placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if total_frames > 0:
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress.progress(min(current_frame / total_frames, 1.0))

            cap.release()
            progress.empty()

            if video_fall_detected:
                st.session_state.fall_events += 1
                
                st.markdown("""
                    <audio autoplay>
                        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
                    </audio>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="alert-card">
                        🚨 FALL EVENT DETECTED<br>
                        Potential fall identified in uploaded video
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Video analysis completed. No high-confidence fall event detected.")

            os.unlink(temporary_file.name)


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab2:
    st.subheader("Behavioral Analytics & Trends")

    if st.session_state.activity_log:
        df = pd.DataFrame(st.session_state.activity_log)
        c1, c2 = st.columns(2)

        with c1:
            fig_pie = px.pie(df, names="Activity", hole=0.45, title="Activity Distribution")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            counts = df["Activity"].value_counts().reset_index()
            counts.columns = ["Activity", "Count"]
            fig_bar = px.bar(counts, x="Activity", y="Count", title="Detected Activity Counts", text="Count")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("### Confidence Timeline (Live Session)")
        fig_line = px.line(df, y="Confidence", x=df.index, title="Detection Confidence Over Time", markers=True)
        fig_line.add_hline(y=alert_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### Confidence Statistics")
        avg_confidence = df["Confidence"].mean()
        max_confidence = df["Confidence"].max()

        c3, c4 = st.columns(2)
        c3.metric("Average Confidence", f"{avg_confidence * 100:.2f}%")
        c4.metric("Maximum Confidence", f"{max_confidence * 100:.2f}%")

    else:
        st.info("Upload an image or video to generate analytics.")


# ============================================================
# TAB 3 — LOGS & EXPORT
# ============================================================

with tab3:
    st.subheader("System Logs & Incident Reports")

    if st.session_state.activity_log:
        log_df = pd.DataFrame(st.session_state.activity_log)
        display_df = log_df.copy()
        display_df["Confidence"] = (display_df["Confidence"] * 100).round(2)
        display_df.rename(columns={"Confidence": "Confidence (%)"}, inplace=True)

        st.dataframe(display_df.tail(30), use_container_width=True)

        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Activity Report",
            data=csv_data,
            file_name="sentinel_activity_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No activity has been recorded yet.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Sentinel AI | Educational AI Healthcare Monitoring Prototype")
