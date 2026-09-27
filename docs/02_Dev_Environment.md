# Kinova Arm - Phase 2: Development Environment

## 1. Prerequisites
* **OS:** Ubuntu (Linux) is highly recommended for robotics, but Windows works for the Kortex API.
* **Python:** Install Python 3.6 or higher.

## 2. Installing the API (Kortex)
Kinova provides the `kortex_api`. This allows your code to talk to the robot.

**Installation Steps (Python):**
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Kinovarobotics/kortex.git](https://github.com/Kinovarobotics/kortex.git)
    ```
2.  Install the API python package (usually found in the `api_python` folder):
    ```bash
    cd kortex/api_python
    pip install .
    ```

## 3. Your First Script: "Hello Robot"
Create a script to verify connection:

```python
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

# Connection details
IP_ADDRESS = "192.168.1.10" # Replace with your robot's IP
PORT = 10000

# Setup connection
transport = TCPTransport()
router = RouterClient(transport, RouterClient.basicErrorCallback)
transport.connect(IP_ADDRESS, PORT)

# Create session
session_info = SessionManager(router).CreateSession(username="admin", password="admin")
print("Session created successfully! Robot is connected.")

# Close connection
session_info = SessionManager(router).CloseSession()
transport.disconnect()

---

### Phase 3: The Final Project (Voice & Vision)
**Goal:** Integrate Audio, Vision, and Motion.



#### 📄 README 03: Voice & Vision Project Roadmap
Copy the text below to save as `03_Project_Roadmap.md`.

```markdown
# Final Project: Voice-Controlled Pick & Place

## System Architecture
1.  **Input:** Microphone (User Voice).
2.  **Processing:** PC (AI Models).
3.  **Output:** Kinova Arm (Movement).

## Step 1: Voice Recognition (Audio)
**Tools:** `SpeechRecognition` library, OpenAI Whisper, or Google Speech API.
* **Logic:** 1.  Listen for wake word (e.g., "Hey Robot").
    2.  Parse command: "Pick up the [Red Cup]."
    3.  Extract target keyword: `Red Cup`.

## Step 2: Computer Vision (The Eyes)
**Tools:** OpenCV, YOLO (You Only Look Once), RealSense SDK (if using Intel RealSense), or Kinova Vision Module.
* **Task:** 1.  Access the **Wrist Camera** (Color stream).
    2.  Use Object Detection (YOLO) to find the bounding box of `Red Cup`.
    3.  **Depth Calculation:** Convert the 2D pixel (x,y) of the cup into 3D coordinates (x,y,z) relative to the robot base. This is crucial.

## Step 3: Motion Planning
**Tools:** Kortex API (Cartesian Actions).
* **Sequence:**
    1.  **Pre-grasp:** Move 10cm *above* the target coordinates.
    2.  **Approach:** Lower the arm to the target Z-height.
    3.  **Grasp:** Close gripper (monitor current/force to ensure grip).
    4.  **Retract:** Move back up.

## Step 4: Integration Loop
```python
while True:
    command = listen_audio()
    if "pick up" in command:
        target = extract_object_name(command)
        coordinates = vision_system.find(target)
        if coordinates:
            robot.move_to(coordinates)
            robot.grip()
        else:
            print("Object not seen.")
