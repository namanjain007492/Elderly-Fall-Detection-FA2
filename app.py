import os
import sys
import subprocess

# --- 1. EMERGENCY OPENCV FIX (MUST BE LINES 1-10) ---
# Wipes the broken desktop OpenCV and installs the headless server version before booting.
if not os.path.exists('/tmp/cv2_fixed.txt'):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80"])
    with open('/tmp/cv2_fixed.txt', 'w') as f:
        f.write("fixed")

# --- 2. STANDARD IMPORTS ---
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# --- 3. APP CONFIG & MODEL LOADING ---
st.set_page_config(page_title="FA-2 Elderly Fall Detection", layout="wide")
st.title("🛡️ Healthcare Monitoring: Elderly Fall Detection")

@st.cache_resource
def load_ai_models():
    yolo = YOLO('yolov8n-pose.pt')
    # Use relative path for Streamlit Cloud
    model = load_model('models/fall_detection_model.h5') 
    return yolo, model

try:
    yolo, clf = load_ai_models()
except Exception as e:
    st.error("⚠️ Model not found! Please ensure 'models/fall_detection_model.h5' is uploaded to GitHub.")
    st.stop()

classes = ['Fall', 'Normal_Activity']

# --- 4. UI AND CORE LOGIC ---
st.sidebar.header("Monitoring Settings")
input_type = st.sidebar.radio("Select Input Source", ["Upload Image", "Upload Video"])
confidence_threshold = st.sidebar.slider("Alert Confidence Threshold", 0.5, 0.99, 0.75)

if "fall_count" not in st.session_state:
    st.session_state.fall_count = 0
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

def process_frame(img):
    results = yolo(img, verbose=False)
    annotated_img = img.copy()
    detected_label = "No Person"
    conf = 0.0
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            annotated_img = r.plot()
            kp = r.keypoints.data[0].cpu().numpy().flatten()
            
            # Ensure the array size matches the 51 features you trained on
            if len(kp) == 51:
                pred = clf.predict(kp.reshape(1, -1), verbose=0)
                detected_label = classes[np.argmax(pred)]
                conf = np.max(pred)
                
                if detected_label == 'Fall' and conf >= confidence_threshold:
                    cv2.putText(annotated_img, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    return annotated_img, detected_label, conf

if input_type == "Upload Image":
    uploaded = st.file_uploader("Upload Frame", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        annotated_img, label, conf = process_frame(img)
        
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original View")
        col2.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="AI Pose Estimation")
        
        if label == 'Fall' and conf >= confidence_threshold:
            st.error(f"🚨 EMERGENCY ALERT: Fall Detected ({conf*100:.1f}% Confidence)")
            st.session_state.fall_count += 1
            st.session_state.activity_log.append({"Activity": "Fall", "Confidence": conf})
        elif label != "No Person":
            st.success(f"Activity Status: {label} ({conf*100:.1f}%)")
            st.session_state.activity_log.append({"Activity": label, "Confidence": conf})

elif input_type == "Upload Video":
    uploaded = st.file_uploader("Upload Surveillance Video", type=['mp4', 'avi'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame, label, conf = process_frame(frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            if label == 'Fall' and conf >= confidence_threshold:
                st.error(f"🚨 FALL DETECTED IN VIDEO STREAM ({conf*100:.1f}%)")
                st.session_state.fall_count += 1

st.divider()
st.subheader("📊 Healthcare Monitoring Analytics")
m1, m2 = st.columns(2)
m1.metric("Total Fall Alerts", st.session_state.fall_count)
m2.metric("Total Activities Tracked", len(st.session_state.activity_log))

if st.session_state.activity_log:
    st.bar_chart(pd.DataFrame(st.session_state.activity_log)["Activity"].value_counts())
