# Kinova Arm - Phase 1: Setup & Operations

## 1. Safety First (Crucial)
* **The E-Stop:** Always keep the Emergency Stop (red button) within reach. If the robot moves unpredictably, hit this immediately.
* **Workspace:** Ensure the arm has a full range of motion without hitting walls, monitors, or you. 
* **Pinch Points:** Keep fingers away from the joints (actuators) during movement.

## 2. Powering Up
1.  Connect the power supply to the robot base.
2.  Connect the Ethernet cable from the robot to your PC (or router).
3.  Turn on the robot. Wait for the light on the base (Status LED) to turn **Green** (Ready). 
    * *Note: If it is blinking orange/red, check the manual for error codes.*

## 3. The Web App (Kortex Web App)
Modern Kinova arms (Gen3/Lite) host a website inside their own brain.
1.  Open a browser (Chrome recommended).
2.  Type the robot's IP address (usually `192.168.1.10` if connected via USB/Ethernet directly, but check your specific network manual).
3.  **Log in:** Default is often `admin` / `admin`.
4.  **Control:** Go to the "Operations" tab. Try to move the robot using the virtual joystick. 
    * **Cartesian Mode:** Moves the hand (XYZ) in straight lines.
    * **Angular Mode:** Moves individual joints.

## 4. Packing
* Always return the robot to a "Retract" or "Home" position before turning it off so it doesn't slump over under gravity (though Kinova brakes are good).
