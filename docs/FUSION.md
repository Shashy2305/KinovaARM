# Multi-camera object fusion: what it is, what it runs on, and what to take from this repo

This is the perception half of the system: several cameras -> one list of objects in the robot's base frame (`/scene_snapshot`).
It is **late fusion of per-camera detections, not a learned fusion model.** Each camera detects on its own; a small node
turns every detection into base-frame coordinates through TF, matches it to the objects it already knows, blends the positions, and cleans up what
the cameras wrongly report.

## 1. Data flow

```
 RealSense 1 (D435i)  --RGB+aligned depth-->  realsense_detection   --/detections_side------\
 RealSense 2 (D435)   --RGB+aligned depth-->  realsense2_detection  --/detections_realsense2-+--> scene_graph_node --/scene_snapshot--> planner, arm controller, dashboard
 OAK-D (parked)       --RGB+depth--------->   object_detection      --/detections----------/        ^   ^
 Kinova wrist camera  --RGB+depth--------->   wrist_detection       --/detections_wrist----/        |   |
                                                                      /wrist_pixel_detections (pixels, orientation, shape; used for centring, not fusion)
 TF: base_link -> each camera optical frame  (camera_tf_broadcaster nodes, one yaml per camera: ~/.ros/<camera>_calibration.yaml)
 TF: arm links and finger pads (robot_state_publisher)  ->  used to REMOVE the robot's own body from the detections
 ~/.ros/table_geometry.yaml (recorded table top and footprint)  ->  used to keep only what is on the table
 depth only:  obstacle_guard_node (/unknown_obstacles): things on the table the detectors cannot name
```

## 2. Models

| Where | Model | Role |
|---|---|---|
| each RealSense detector, OAK detector | Ultralytics **YOLOv8m** (COCO weights `yolov8m.pt`, GPU) | class + 2D box per camera; target classes bottle, cup, bowl, cell phone, remote, book, scissors, vase, mouse |
| wrist detector | **YOLOv8m-seg** (`yolov8m-seg.pt`) | boxes AND masks; the mask gives orientation, elongation and (shape_analysis.py) the full outline, width and contact profiles |
| fallback | HSV colour blobs (red/blue/green/yellow objects) | when YOLO finds nothing |
| depth | the cameras' own stereo depth (RealSense) | 3D position from the box centre (median of a 16 x 16 patch) |
| fusion | **none learned** | geometry + confidence-weighted averaging + rules |
| planning | Qwen2.5-7B via Ollama (separate from perception) | turns a command + the scene into a plan |

Known weakness of the detectors, handled in the arm controller rather than by retraining: from straight above, a bottle is read as "sports ball"/"bowl" and a
mouse as "cup" (the wrist matching therefore uses POSITION, class is a tie-break). Fine-tuning YOLO on the six table objects is the biggest remaining gain.

## 3. The fusion node (`scene_graph_node.py`)

1. **Resolve to base_link** (`_resolve_base_xyz`): a detection carries either base-frame coordinates or a camera-frame point + `frame_id`; the latter is moved
   through TF. Anything else is dropped (camera numbers are never used as robot numbers).
2. **Data association**: detections are assigned to existing tracks with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), gated by label and
   distance (`SAME_OBJECT_DIST` 0.15 m).
3. **Position fusion** (`_fuse_position`): exponential blend `new = a*reading + (1-a)*estimate` with `a = clip(confidence, 0.15, 0.85)`, so a confident camera
   corrects a shaky one without ever discarding it.
4. **Hygiene filters** (these are what make the output usable; each cost a real failure to find):
   table-region filter (the footprint recorded in `table_geometry.yaml` + 5 cm: other desks, chairs and people are dropped);
   robot-body filter (within 10 cm of an arm link or 4.5 cm of a finger = the robot itself, via TF);
   held-object filter (while the gripper is partly closed, detections in the volume around the pads are the carried object's ghosts);
   cross-label duplicate suppression (6 cm: one physical object read under two labels);
   same-label merge (10 cm; bottle 14, cup 12: transparent objects differ most between cameras);
   staleness (unseen 20 s -> `stale`, 60 s -> dropped).
5. **Snapshot** at 5 Hz on `/scene_snapshot` (JSON `{id: {label, x, y, z, confidence, reachable, stale, ...}}`), with predicates (`reachable`, near/left_of/right_of) for the planner.

## 4. Calibration (what makes the positions right)

The cameras' base-frame poses come from the arm itself (`calibration/pnp_extrinsics.py`): click known points of the arm in a still image (base, shoulder, elbow, wrist,
flange, two fingertips; their base-frame positions are exact from the joint angles), solve PnP, save a candidate yaml, install it. `calibration/check_extrinsics.py`
measures the result without moving anything: table-plane tilt/height per camera (should be level and agree) and the position spread of one object seen by two cameras
(< 5 cm). Typical accuracy here: 8-10 px RMS, objects within 2-4 cm between cameras. Re-run the check whenever anything looks off; a re-plugged camera is flagged by `camera_watchdog`.

## 5. Reusing it: what to take

Minimum for a new setup with your own cameras and arm:
- `scene_graph_node.py` (+ `safety_geometry.py` for the table region and robot-body filters; both are pure Python apart from the ROS plumbing)
- one detection node per camera (copy `realsense_detection.py`; it publishes `/detections*` JSON lists with `label`, `confidence`, camera-frame `cx_3d/cy_3d/cz_3d`, `frame_id`)
- `static_tf_broadcaster.py` + a calibration yaml per camera (`parent_frame`, `child_frame`, `translation`, `rotation_quat`)
- `calibration/pnp_extrinsics.py`, `calibration/check_extrinsics.py` (arm-based calibration and its check)
- a table geometry file (`~/.ros/table_geometry.yaml`: `table_top_z`, `x`, `y`) - the dashboard's Table safety tab records it
Optional: `camera_watchdog.py` (blocks a camera after a drop until it is re-verified), `obstacle_guard*.py` (unknown obstacles from depth),
`wrist_servo.py` + `shape_analysis.py` (close-range centring and shape from a wrist camera).

Dependencies: ROS 2 Humble, `realsense2_camera`, `ultralytics` (+ CUDA for speed), `cv_bridge`, `tf2_ros`, `scipy`, `numpy`, `opencv`.
Everything is tuned to this table, these cameras and these objects: the merge radii, filters and thresholds are constants at the top of `scene_graph_node.py`
(and `OBJECT_*` tables in `wrist_servo.py`); expect to retune them.

## 6. Checking that it works (no arm needed)
`python3 calibration/check_extrinsics.py --label cup` (cameras agree?), `ros2 topic echo /scene_snapshot --once` (objects where they are?),
`python3 scripts/outcome_report.py` (how picks and places have actually gone), the dashboard's "3D Fusion" tab (per-camera point clouds and the fused objects).
