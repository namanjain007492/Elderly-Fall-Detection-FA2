import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import requests
from streamlit_lottie import st_lottie
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Sentinel AI | Fall Detection", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0f111a;}
    h1, h2, h3 {color: #00f2fe;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #1e2130; border-radius: 10px 10px 0px 0px; padding: 10px 20px;}
    .stTabs [aria-selected="true"] {background-color: #00f2fe; color: black; font-weight: bold;}
    .metric-card {background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%); padding: 20px; border-radius: 15px; border-left: 5px solid #00f2fe; box-shadow: 0 4px 15px rgba(0,0,0,0.3);}
    .alert-card {background: linear-gradient(135deg, #ff4b4b 0%, #ff0000 100%); padding: 20px; border-radius: 15px; color: white; text-align: center; font-weight: bold; animation: pulse 1.5s infinite; box-shadow: 0 4px 15px rgba(255,75,75,0.4);}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.02);} 100% {transform: scale(1);}}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_ai_system():
    yolo_model = YOLO('yolov8n-pose.pt')
    classifier = load_model('models/fall_detection_model.h5')
    return yolo_model, classifier

# --- INITIALIZE AI & STATE ---
try:
    yolo, clf = load_ai_system()
    # 🚨 Ensure this matches your model's exact LabelEncoder order!
    classes = ['Falling', 'Normal Activity', 'Sitting', 'Standing', 'Walking']
except Exception as e:
    st.error("⚠️ AI Models not found. Check 'yolov8n-pose.pt' and 'models/fall_detection_model.h5'")
    st.stop()

if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []
if 'fall_count' not in st.session_state:
    st.session_state.fall_count = 0

# --- SIDEBAR: MASCOT & CONTROLS ---
with st.sidebar:
    # 3D-style animated AI Mascot
    lottie_robot = load_lottieurl("https://lottie.host/8040a4e7-6a4a-4c22-9e2e-2a2979219e27/0O1K8UoQnZ.json")
    st_lottie(lottie_robot, height=200, key="robot")
    
    st.title("⚙️ System Controls")
    input_mode = st.radio("Monitoring Source", ["Static Image", "Video Stream", "Live Webcam"])
    alert_threshold = st.slider("Fall Confidence Threshold", 0.50, 0.99, 0.75, 0.05)
    
    st.markdown("---")
    st.markdown("**System Status:** 🟢 Online")
    st.markdown("**AI Engine:** YOLOv8 + Keras")

# --- CORE AI PIPELINE ---
def analyze_frame(frame):
    results = yolo(frame, verbose=False)
    annotated = frame.copy()
    label = "Scanning..."
    confidence = 0.0
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            annotated = r.plot()
            kp = r.keypoints.data[0].cpu().numpy().flatten()
            
            if len(kp) == 51:
                prediction = clf.predict(kp.reshape(1, -1), verbose=0)
                class_idx = np.argmax(prediction)
                label = classes[class_idx]
                confidence = float(np.max(prediction))
                
                # Visual overlay for emergencies
                if label == 'Falling' and confidence >= alert_threshold:
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 90), (0, 0, 255), -1)
                    cv2.putText(annotated, f"🚨 EMERGENCY: FALL DETECTED ({confidence*100:.1f}%)", 
                                (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
                else:
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (0, 0, 0), -1)
                    cv2.putText(annotated, f"Status: {label} | Conf: {confidence*100:.1f}%", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated, label, confidence

# --- MAIN DASHBOARD LAYOUT ---
st.title("🛡️ Sentinel: Elderly Fall Detection Network")

# Real-time Top Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h3>Active Scans</h3><h2>{len(st.session_state.activity_log)}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>Current Input</h3><h2>{input_mode}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card' style='border-left: 5px solid #ff4b4b;'><h3>Critical Alerts</h3><h2>{st.session_state.fall_count}</h2></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TABS FOR CLEAN INTERFACE
tab1, tab2, tab3 = st.tabs(["📷 Live Monitoring Hub", "📈 Advanced Analytics", "📋 Data Export & Logs"])

with tab1:
    st.subheader(f"Current Feed: {input_mode}")
    
    if input_mode == "Static Image":
        uploaded_file = st.file_uploader("Upload Room Footage", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            processed_img, final_label, final_conf = analyze_frame(img)
            
            if final_label == 'Falling' and final_conf >= alert_threshold:
                st.markdown("<div class='alert-card'>🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED: FALL DETECTED 🚨</div><br>", unsafe_allow_html=True)
                st.session_state.fall_count += 1
            
            st.session_state.activity_log.append({"Activity": final_label, "Confidence": final_conf})
            
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Raw Camera Feed", use_container_width=True)
            c2.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="AI Spatial Analysis", use_container_width=True)

    elif input_mode == "Video Stream":
        uploaded_video = st.file_uploader("Upload Surveillance Clip", type=['mp4', 'avi'])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            video_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                p_frame, final_label, final_conf = analyze_frame(frame)
                video_placeholder.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                if final_label != "Scanning...":
                    st.session_state.activity_log.append({"Activity": final_label, "Confidence": final_conf})
                    if final_label == 'Falling' and final_conf >= alert_threshold:
                        st.session_state.fall_count += 1

    elif input_mode == "Live Webcam":
        st.warning("Webcam processing requires high local CPU usage.")
        run_cam = st.checkbox("Turn on Camera")
        cam_placeholder = st.empty()
        
        if run_cam:
            cap = cv2.VideoCapture(0)
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera access failed.")
                    break
                p_frame, final_label, final_conf = analyze_frame(frame)
                cam_placeholder.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            cap.release()

with tab2:
    st.subheader("Behavioral Analytics & Trends")
    if len(st.session_state.activity_log) > 0:
        df = pd.DataFrame(st.session_state.activity_log)
        
        c1, c2 = st.columns(2)
        with c1:
            # Enhanced 3D-style Donut Chart
            fig_pie = px.pie(df, names='Activity', hole=0.4, title="Activity Distribution",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            # Gauge chart for system stress/alerts
            alert_ratio = (st.session_state.fall_count / len(df)) * 100 if len(df) > 0 else 0
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = alert_ratio,
                title = {'text': "Critical Incident Ratio (%)"},
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "red"},
                         'steps': [{'range': [0, 10], 'color': "lightgreen"},
                                   {'range': [10, 30], 'color': "orange"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("Awaiting data stream to generate analytics...")

with tab3:
    st.subheader("System Logs & Incident Reports")
    if len(st.session_state.activity_log) > 0:
        log_df = pd.DataFrame(st.session_state.activity_log)
        st.dataframe(log_df.tail(20), use_container_width=True)
        
        csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Incident Report (CSV)",
            data=csv,
            file_name='healthcare_monitoring_log.csv',
            mime='text/csv',
        )
    else:
        st.info("No recorded events yet.")
