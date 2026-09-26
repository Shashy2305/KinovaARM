# Hand-Eye Calibration Pipeline
## Kinova Gen3 + OAK-D Pro Wide → Eye-in-Hand → Pick & Place

---

## Your Setup
```
[OAK-D Pro Wide] ← FIXED to stand/ceiling
       ↓ sees
[7×10 Checkerboard] ← MOUNTED on Kinova Gen3 EE
```
**Goal:** Find T_base_cam (where camera is in robot base frame)

---

## Checkerboard Specs (from calib.io)
| Parameter | Value |
|-----------|-------|
| Pattern   | 8×11 squares → **7×10 inner corners** |
| Square size | **15 mm** |
| Board size | 200×150 mm |

## OAK-D Intrinsics (640×400)
| FX | FY | CX | CY |
|----|----|----|-----|
| 426.117 | 425.930 | 315.225 | 205.309 |

---

## Step-by-Step Instructions

### 0. Hardware Setup
- [ ] Print checkerboard at **exact 1:1 scale** (no scaling!)  
- [ ] Mount rigidly on Kinova Gen3 EE (bracelet_link)
- [ ] Fix OAK-D so it **cannot move** during calibration
- [ ] Start your ROS2 stack: `ros2 launch ... robot_launch.py`
- [ ] Start OAK-D: `ros2 run your_pkg oak_camera_node`

### 1. Data Collection (Python/ROS2)
```bash
ros2 run your_pkg collect_calib_data.py
```
- Move robot to **15–20 diverse poses** — vary:
  - Position (all quadrants of workspace)
  - Orientation (tilt EE left/right/forward/back)
  - Distance to camera (near + far)
- Press **ENTER** at each pose to capture
- Data saved to: `~/calib_data/`

**Good pose tips:**
- ✓ Large rotation differences between poses (>30° preferred)
- ✓ Checkerboard fully visible in every image
- ✗ Don't keep robot in same orientation just moving linearly

### 2. MATLAB — Detect Corners
```matlab
>> Step1_DetectCorners
```
- Detects 7×10 corners in each image
- Computes T_cam_board for each pose
- Checks reprojection error (should be < 1.5 px)

### 3. MATLAB — Solve AX=XB
```matlab
>> Step2_HandEyeCalib
```
- Runs Tsai-Lenz + Park-Martin + MATLAB estimateHandEye
- Picks best result by residual
- Outputs T_base_cam

### 4. MATLAB — Verify & Export
```matlab
>> Step3_VerifyAndExport
```
- 3D verification: < 5mm = excellent, < 15mm = good
- **Outputs new `calibration_tf` block for robot_launch.py**
- Copy the snippet into `robot_launch.py`!

### 5. MATLAB — Pick & Place Test
```matlab
>> Step4_PickAndPlace
```
- Edit the `example_detections` array with real OAK-D XYZ values
- Generates goal_poses.json for MoveIt2

### 6. Execute Pick & Place (ROS2)
```bash
ros2 run your_pkg publish_goal_pose.py
```

---

## Files
| File | Purpose |
|------|---------|
| `CALIB_CONFIG.m` | All shared configuration |
| `collect_calib_data.py` | ROS2 data collector |
| `Step1_DetectCorners.m` | Corner detection + PnP |
| `Step2_HandEyeCalib.m` | AX=XB solver (3 methods) |
| `Step3_VerifyAndExport.m` | Verification + launch.py snippet |
| `Step4_PickAndPlace.m` | Transform detection → base frame |
| `publish_goal_pose.py` | Send goals to MoveIt2 |

---

## Math: AX = XB
```
A_ij = T_base_ee_i  * inv(T_base_ee_j)   ← relative EE motion (from robot)
B_ij = T_cam_board_i * inv(T_cam_board_j) ← relative board motion (from camera)
X    = T_base_cam                          ← SOLUTION: camera in base frame
```

---

## Troubleshooting
| Problem | Fix |
|---------|-----|
| Board not detected | Check lighting, ensure full visibility |
| High reprojection error | Reprint board at exact scale |
| High 3D error (>15mm) | Add more poses with larger rotations |
| TF lookup failed | Check `bracelet_link` frame name in URDF |
| estimateHandEye missing | Use MATLAB R2022b+ or use Tsai/Park result |
