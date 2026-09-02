#🛡️ Sentinel AI
## Intelligent Elderly Activity & Fall Detection System

<p align="center">

<img src="https://img.shields.io/badge/AI-Computer%20Vision-00f2fe?style=for-the-badge&logo=opencv&logoColor=white">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">

<img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">

<img src="https://img.shields.io/badge/YOLOv8-Pose-111111?style=for-the-badge">

<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">

<img src="https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

</p>

<p align="center">


<img src="https://img.shields.io/badge/Project-FA--2-blue?style=flat-square">

<img src="https://img.shields.io/badge/Domain-Artificial%20Intelligence-purple?style=flat-square">

<img src="https://img.shields.io/badge/Application-Healthcare%20Monitoring-green?style=flat-square">

</p>

---

## 📌 Project Overview

**Sentinel AI** is an Artificial Intelligence and Computer Vision project designed to recognize human activities and identify potential fall events from visual input.

The system combines:

- Human pose estimation
- Deep learning
- Computer vision
- Activity classification
- Confidence-based decision making
- Data visualization
- Interactive monitoring
- Event logging
- Streamlit deployment

The primary objective is to develop an intelligent monitoring prototype capable of recognizing activities such as:

- 🚨 Falling
- 🚶 Walking
- 🪑 Sitting
- 🧍 Standing
- 🏠 Normal Activity

The project explores how human body-pose information can be transformed into numerical features and used by a machine-learning model to distinguish between different activities.

---

# 🎯 Project Objective

The main objective of Sentinel AI is to investigate whether **human pose landmarks extracted from visual data can be used to classify everyday activities and detect potential falls**.

Instead of relying only on raw image appearance, the system extracts the spatial positions and confidence values of human body keypoints.

These pose features are then supplied to a trained deep-learning classifier.

### Core concept

```text
Camera / Image / Video
          ↓
   Human Detection
          ↓
   Pose Estimation
          ↓
  Body Keypoint Extraction
          ↓
 Numerical Pose Features
          ↓
 Deep Learning Classifier
          ↓
 Activity Prediction
          ↓
 Fall Detection Decision
          ↓
 Monitoring Dashboard
