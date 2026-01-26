# KinovaARM
A modular control system for the Kinova Gen3 Robotic Arm, featuring voice-activated commands and dual-camera computer vision for autonomous pick-and-place operations.
# Kinova Gen3 Voice & Vision Controller 🦾👁️🎤

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hardware](https://img.shields.io/badge/Hardware-Kinova%20Gen3-orange)

## 📖 Project Overview
This repository documents the development of an autonomous control system for the **Kinova Gen3 Robotic Arm**. The project evolves from basic manual control to a sophisticated AI-driven assistant capable of understanding voice commands and using computer vision to identify, locate, and manipulate objects in 3D space.

The system utilizes a **dual-camera setup**:
1.  **Wrist-Mounted Camera:** For close-range alignment and gripping.
2.  **External Camera:** For global scene perception and object recognition.

## 🎯 Key Objectives
* **Manual Control:** Establish communication via Kortex API.
* **Voice Integration:** Parse natural language commands (e.g., *"Robot, hand me the red screwdriver"*).
* **Computer Vision:** Implement YOLO/OpenCV to detect object coordinates.
* **Automated Manipulation:** Inverse kinematics for safe pick-and-place trajectories.

## 🛠️ Hardware & Tech Stack

### Hardware
* **Robot:** Kinova Gen3 (7 DOF)
* **Vision:** Integrated Wrist Camera + External Depth Camera (e.g., Intel RealSense or Webcam)
* **Compute:** [Your PC Specs, e.g., Ubuntu 20.04 Workstation]

### Software
* **Language:** Python 3.x
* **Robot API:** Kinova Kortex API
* **Vision:** OpenCV, YOLO (Object Detection)
* **Audio:** SpeechRecognition, PyAudio
* **Version Control:** Git

## 📂 Repository Structure
```text
├── docs/                 # Documentation and Setup Guides
│   ├── 01_Setup.md       # Initial hardware setup
│   ├── 02_Dev_Env.md     # Coding environment setup
│   └── 03_Roadmap.md     # Project milestones
├── scripts/              # Executable Python scripts
│   ├── basic_movement/   # Simple joint/cartesian tests
│   ├── vision/           # Camera streaming and detection logic
│   └── voice/            # Audio processing scripts
├── src/                  # Core library code (classes and functions)
└── README.md             # Project Overview
