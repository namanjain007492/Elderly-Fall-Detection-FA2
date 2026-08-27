import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
from ultralytics import YOLO
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Elderly Fall Detection", layout="wide")
st.title("🛡️ Healthcare Monitoring: Elderly Fall Detection")

@st.cache_resource
def load_ai_models():
    yolo = YOLO('yolov8n-pose.pt')
    model = load_model('/content/FA2_Elderly_Activity_Fall_Detection/models/fall_detection_model.h5')
    return yolo, model

yolo, clf = load_ai_models()
classes = ['Fall', 'Normal_Activity']

# Sidebar controls
st.sidebar.header("Monitoring Settings")
input_type = st.sidebar.radio("Select Input Source", ["Upload Image", "Upload Video"])
confidence_threshold = st.sidebar.slider("Alert Confidence Threshold", 0.5, 0.99, 0.75)

st.sidebar.header("✨ Advanced Features")
caregiver_mode = st.sidebar.checkbox("Caregiver Focus Mode (Simple UI)")
low_light_mode = st.sidebar.checkbox("Enable Low-Light Enhancement")

if "fall_count" not in st.session_state:
    st.session_state.fall_count = 0
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "false_alarms" not in st.session_state:
    st.session_state.false_alarms = 0

def enhance_low_light(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def process_frame(img):
    if low_light_mode:
        img = enhance_low_light(img)
        
    results = yolo(img, verbose=False)
    annotated_img = img.copy()
    detected_label = "No Person"
    conf = 0.0
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            if not caregiver_mode: # Hide complex bounding boxes in simple mode
                annotated_img = r.plot()
            kp = r.keypoints.data[0].cpu().numpy().flatten()
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
        
        if caregiver_mode:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Live Feed")
        else:
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original View")
            col2.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="AI Pose Estimation")
        
        if label == 'Fall' and conf >= confidence_threshold:
            st.error(f"🚨 EMERGENCY ALERT: Fall Detected ({conf*100:.1f}% Confidence)")
            st.session_state.fall_count += 1
            st.session_state.activity_log.append({"Activity": "Fall", "Confidence": conf})
            if st.button("🚩 Flag as False Alarm (Retrain Model)"):
                st.session_state.false_alarms += 1
                st.session_state.fall_count -= 1
                st.success("False alarm logged for future AI retraining.")
        elif label != "No Person":
            st.success(f"Activity Status: {label} ({conf*100:.1f}%)")
            st.session_state.activity_log.append({"Activity": label, "Confidence": conf})

# ... (Video Logic remains the same, just call process_frame) ...

if not caregiver_mode:
    st.divider()
    st.subheader("📊 Healthcare Monitoring Analytics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Fall Alerts", st.session_state.fall_count)
    m2.metric("Total Activities Tracked", len(st.session_state.activity_log))
    m3.metric("Logged False Alarms", st.session_state.false_alarms)

    if st.session_state.activity_log:
        st.bar_chart(pd.DataFrame(st.session_state.activity_log)["Activity"].value_counts())
