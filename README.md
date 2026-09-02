🛡️ Sentinel AI
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

<img src="https://img.shields.io/badge/Application-Sentinel%20AI-green?style=flat-square">

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

🧠 Artificial Intelligence Pipeline

Sentinel AI follows a multi-stage computer-vision pipeline.

Stage 1 — Visual Input

The system accepts visual information through:

Static images
Uploaded videos
Future live-camera integration
Stage 2 — Human Pose Estimation

The system uses YOLOv8 Pose to identify human body keypoints.

The pose model provides information about body landmarks such as:

Nose
Shoulders
Elbows
Wrists
Hips
Knees
Ankles

The standard human pose representation contains:

17 keypoints
×
3 values per keypoint

= 51 numerical features

The three values represent:

X coordinate
Y coordinate
Confidence

Stage 3 — Feature Extraction

The detected pose landmarks are converted into a numerical feature vector.

Example:

[x1, y1, c1,
 x2, y2, c2,
 x3, y3, c3,
 ...
 x17, y17, c17]

This produces a 51-dimensional representation of the detected person's pose.

Stage 4 — Activity Classification

The extracted pose representation is passed to the trained Keras/TensorFlow classifier.

The classifier produces probabilities for each activity class.

Example:

Falling          0.82
Normal Activity  0.04-0.05

The class with the highest probability becomes the predicted activity.

Stage 5 — Fall Decision

The application applies a configurable confidence threshold.

For example:

Fall confidence ≥ threshold
             ↓
      Potential fall
             ↓
      Alert displayed

The threshold can be adjusted through the Streamlit interface.

🏷️ Activity Classes

The FA-2 system is designed around two activity categories.

| Class              | Description                           |
| ------------------ | ------------------------------------- |
| 🚨 Falling         | Person is detected in a falling state |
| 🏠 Normal Activity | General non-specific daily activity   |


The exact class ordering used by the trained model must match the classifier configuration in the application.

🏗️ System Architecture

                    ┌──────────────────────┐
                    │    Visual Input      │
                    │ Image / Video / Cam  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    YOLOv8 Pose       │
                    │  Pose Estimation     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 17 Body Keypoints    │
                    │ X / Y / Confidence   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Vector       │
                    │ 51 Numerical Values  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ TensorFlow / Keras   │
                    │ Activity Classifier  │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Activity Prediction      │
              └───────────────┬────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Normal Activity            Falling
                 │                         │
                 ▼                         ▼
          Status Display             Alert System
                                           │
                                           ▼
                                  Incident Logging
                                           │
                                           ▼
                                  Analytics Dashboard

🔬 FA-2 Development Workflow

The FA-2 development process is organized into several stages.

Phase 1 — Environment Setup
Google Colab environment
Python
TensorFlow
OpenCV
MediaPipe / pose-processing tools where applicable
Ultralytics
Scikit-learn
Phase 2 — Dataset Preparation
Dataset acquisition
Activity verification
Label verification
Frame extraction
Dataset organization
Phase 3 — Pose Processing
Human detection
Keypoint extraction
Feature preparation
Pose-data validation
Phase 4 — Model Development
Training dataset preparation
Validation dataset preparation
Classifier development
Model training
Hyperparameter experimentation
Phase 5 — Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Classification report
Phase 6 — Application Development

The trained model is integrated into a Streamlit dashboard.

Phase 7 — Testing

The system is tested using:

Individual images
Previously unseen videos
Different activity classes
Confidence thresholds
Phase 8 — Deployment

The final application can be deployed using Streamlit-compatible hosting.

📥 Data Export

The activity log can be exported as a CSV file.

Example:

sentinel_activity_log.csv

This allows further analysis using:

Microsoft Excel
Google Sheets
Python
Pandas
Statistical software

👨‍💻 Project Information

Project: Sentinel AI — Elderly Activity & Fall Detection

Assessment: FA-2

Program: IBCP — Artificial Intelligence

Academic Year: 2026–2027

Student: Naman Jain

School: Jain Vidyalaya IB World School, Madurai

⚖️ Educational Disclaimer

Sentinel AI is an educational Artificial Intelligence project developed for learning and experimentation in computer vision and machine learning.

The system should not be used as a substitute for:

Medical professionals
Emergency services
Certified fall-detection systems
Professional healthcare monitoring

Predictions generated by the model should be considered experimental outputs.
