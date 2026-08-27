import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Elderly Safety AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .st-emotion-cache-1wivap2 {border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);}
    .alert-danger {background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; animation: pulse 1.5s infinite;}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.02);} 100% {transform: scale(1);}}
    .metric-card {background: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #00f2fe;}
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED FOR SPEED) ---
@st.cache_resource
def load_ai_system():
    yolo = YOLO('yolov8n-pose.pt')
    model_path = 'models/fall_detection_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Missing model file at {model_path}. Please upload it to GitHub.")
        st.stop()
    classifier = load_model(model_path)
    return yolo, classifier

yolo, clf = load_ai_system()
classes = ['Fall', 'Normal_Activity']

# --- SIDEBAR & STATE ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100)
st.sidebar.title("Caregiver Controls")
input_mode = st.sidebar.radio("Observation Mode", ["Live Camera", "Upload Video", "Static Frame"])
alert_threshold = st.sidebar.slider("Sensitivity Threshold", 0.50, 0.99, 0.75, 0.05)

if "total_events" not in st.session_state:
    st.session_state.total_events = 0
if "fall_events" not in st.session_state:
    st.session_state.fall_events = 0

# --- CORE PROCESSING PIPELINE ---
def analyze_frame(frame):
    results = yolo(frame, verbose=False)
    output_frame = frame.copy()
    current_label = "Scanning..."
    highest_conf = 0.0
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            output_frame = r.plot()
            kp = r.keypoints.data[0].cpu().numpy().flatten()
            
            if len(kp) == 51:
                pred = clf.predict(kp.reshape(1, -1), verbose=0)
                current_label = classes[np.argmax(pred)]
                highest_conf = np.max(pred)
                
                if current_label == 'Fall' and highest_conf >= alert_threshold:
                    cv2.rectangle(output_frame, (0, 0), (output_frame.shape[1], 80), (0, 0, 255), -1)
                    cv2.putText(output_frame, f"CRITICAL: FALL DETECTED ({highest_conf*100:.1f}%)", 
                                (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    
    return output_frame, current_label, highest_conf

# --- MAIN DASHBOARD ---
st.title("🛡️ Sentinel: AI Healthcare Monitor")
st.markdown("Real-time posture estimation and activity classification dashboard.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h3>Active Streams</h3><h2>1</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>Total Scans</h3><h2>{st.session_state.total_events}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card' style='border-left: 5px solid #ff4b4b;'><h3>Critical Alerts</h3><h2>{st.session_state.fall_events}</h2></div>", unsafe_allow_html=True)

st.divider()

if input_mode == "Static Frame":
    uploaded_file = st.file_uploader("Upload Room Footage", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.session_state.total_events += 1
        processed_img, label, conf = analyze_frame(img)
        
        if label == 'Fall' and conf >= alert_threshold:
            st.session_state.fall_events += 1
            st.markdown("<div class='alert-danger'>🚨 EMERGENCY RESPONSE REQUIRED: FALL DETECTED 🚨</div><br>", unsafe_allow_html=True)
        else:
            st.success(f"Status: Normal ({label} - {conf*100:.1f}%)")
            
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Raw Camera Feed", use_container_width=True)
        c2.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="AI Spatial Analysis", use_container_width=True)

elif input_mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload Surveillance Clip", type=['mp4', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        video_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            p_frame, label, conf = analyze_frame(frame)
            video_placeholder.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            if label == 'Fall' and conf >= alert_threshold:
                st.session_state.fall_events += 1
                
elif input_mode == "Live Camera":
    st.info("Ensure your browser has camera permissions enabled.")
    run = st.checkbox("Start Live Stream")
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video feed.")
                break
            
            p_frame, label, conf = analyze_frame(frame)
            FRAME_WINDOW.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB))
        cap.release()
