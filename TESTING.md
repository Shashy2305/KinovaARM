# Testing the core pipeline on the real arm

Staged checks for the core pipeline changes (frame handling, single OAK-D TF source,
`pick` moving to the object, planner validation). Each stage isolates one thing, and every
stage that moves the arm comes after the ones that can catch a problem without motion.
**Stop at the first stage that fails** and note what you saw — the later stages depend on it.

What was already verified off-hardware (ROS 2 Jazzy, fake YOLO / MoveIt / Ollama): message
formats, TF math, the order and targets of the pick commands, failure handling. What only
the arm can tell us: gripper values, gripper orientation, real depth, and how far off the
calibration is.

**Safety for every stage that moves:** one person on the E-stop, workspace clear, start with
`-p speed:=0.10`. `arm_controller` clamps targets to x ∈ [0.10, 0.55], y ∈ [-0.35, 0.35],
z ∈ [0.08, 0.50] m, but that is not a collision check against the table or objects.

---

## Stage 0 — Build and preflight (no hardware)

Check out the `main` branch in the copy of this repo the lab workspace builds
`thesis_robot` from. `oak_camera_node.py` now lives at `drivers/oak_camera_node.py`
inside this repo — it is no longer a separate copy outside it. If the lab PC's
`cameras.launch.py` still points at an external
`~/workspace/ros2_kortex_ws/oak_camera_node.py`, delete that copy once
`drivers/oak_camera_node.py` (via `drivers/oak_launch.py`) is confirmed working, so
there is only ever one copy.

```bash
cd ~/workspace/ros2_kortex_ws
colcon build --packages-select thesis_robot && source install/setup.bash
ros2 pkg executables thesis_robot | grep -E "camera_tf_broadcaster|arm_controller|scene_graph_node"
ls ~/.ros/handeye_calibration_corrected.yaml             # may be missing — that is fine
```

**Pass:** build succeeds and the three executables are listed.
**Record:** whether the calibration file exists (if it does, its values will be used instead
of the old hard-coded OAK-D ones — mention it).

---

## Stage 1 — Calibration TF (cameras on, no motion)

Start robot bringup and cameras as in the README (step 1, step 2 including
`camera_tf_broadcaster`). Then:

```bash
ros2 run tf2_ros tf2_echo base_link global_camera_link
ros2 run tf2_ros tf2_echo base_link global_camera_color_optical_frame

# exactly one publisher of the OAK-D transform:
timeout -s INT 4 ros2 topic echo /tf_static > /tmp/tfs.txt
grep -c "child_frame_id: global_camera_link" /tmp/tfs.txt
```

**Pass:**
- OAK-D: translation `[0.480, 0.720, 1.000]` (when no calibration file exists) and the
  broadcaster log says `No calibration found … using last known OAK-D calibration`.
- RealSense: translation `[0.990, -0.130, 0.770]` (from `cameras.launch.py`).
- The count is **1**. A count of 2 means an old `oak_camera_node.py` that still broadcasts
  is being launched — check which copy `cameras.launch.py` points to.

---

## Stage 2 — Perception → scene, one camera at a time (no motion)

This measures how far off each camera is. **The numbers from this stage are the most useful
input for the calibrator.**

**2a. Ground truth.** Put one cup in the middle of the workspace. Jog the arm (web app or
joystick) until the gripper's tool point is directly above the centre of the cup's top rim,
almost touching. First confirm in RViz (TF display) where `end_effector_link` actually sits
on the Robotiq 2F-140 — use whichever frame is at the fingertip centre. Then:

```bash
ros2 run tf2_ros tf2_echo base_link end_effector_link    # note x, y, z
```

Move the arm back out of the cameras' view.

**2b. OAK-D only.**

```bash
ros2 run thesis_robot object_detection --ros-args -p desired_object:=cup
ros2 run thesis_robot scene_graph_node
ros2 topic echo /scene_snapshot --once
```

**Pass:**
- `object_detection` logs `Intrinsics: … for 416x256` and **no**
  `camera_info is … but image is …` warning.
- `scene_graph_node` shows **no** `Dropping … no TF` warnings.
- The cup appears with a numeric `z` (not `null`) and `reachable: true`.

**Record:** the cup's `x, y, z` from the snapshot, and the error = snapshot − ground truth
(x and y should match the rim centre; z is the visible surface point, so expect it to be
somewhat below the rim).

**2c. RealSense only.** Stop `object_detection`, restart `scene_graph_node` (clears the scene), and run:

```bash
ros2 run thesis_robot realsense_detection
ros2 topic echo /scene_snapshot --once
```

Same pass criteria and the same record. Repeat 2b/2c with the cup at 2–3 other spots
(left, right, far) if time allows — if the error changes direction between spots, the
calibration is rotated, not just shifted.

---

## Stage 3 — Gripper open/close values (gripper only)

```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 50.0}}"
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.695, max_effort: 50.0}}"
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.14, max_effort: 50.0}}"
```

**Record:** which position is fully open and which is fully closed.
`arm_controller` currently uses **open = 0.14, closed = 0.02**. If 0.695 closes the
gripper, those values are reversed — **do not continue to Stage 7** until they are fixed.

---

## Stage 4 — Grasp orientation (manual jog, no commanded motion)

Jog the arm so the gripper points straight down with the fingers vertical, then:

```bash
ros2 run tf2_ros tf2_echo base_link end_effector_link    # note the quaternion (xyzw)
```

**Pass:** the quaternion matches `[0.0, 0.707, 0.0, 0.707]` — or its negative, which is
the same rotation. If it is different, record it: that value replaces the constant
`arm_controller` uses for every move.

---

## Stage 5 — Full pipeline in dry run (no motion)

Run everything from the README (perception, `scene_graph_node`, `llm_planner_node`,
`arm_controller` with the default `dry_run:=true`). Watch:

```bash
ros2 topic echo /planner_status
ros2 topic echo /pick_place_status
```

**5a. Text command, no microphone:**

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: 'pick up the cup'}"
```

**Pass:** `/planner_status` goes `PLANNING` → `EXECUTING`, and `arm_controller` logs:

```
pick cup_00 at (x, y, z) — pre-grasp z=…, grasp z=…
  [DRY RUN] open_gripper
  [DRY RUN] move_to robot(…, pre-grasp z)
  [DRY RUN] move_to robot(…, grasp z) cartesian
  [DRY RUN] close_gripper
  [DRY RUN] move_to robot(…, pre-grasp z) cartesian
Plan executed successfully
```

and the x/y in that log match the cup in `/scene_snapshot`.
**Record:** the `LLM response in X s` latency from the planner log.

**5b. Failure paths:**

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: 'pick up the purple elephant'}"
```

Expected: `REJECTED: … not in scene` or `REJECTED: Plan is empty…`. `ERROR` should now only
appear if the model returned invalid JSON — note it if you see it.

```bash
ros2 topic pub --once /action_plan std_msgs/msg/String '{data: "{\"command\": \"test\", \"plan\": [{\"action\": \"pick\", \"object_id\": \"mug_99\", \"approach_z\": 0.2}]}"}'
```

Expected in the `arm_controller` log: `pick: 'mug_99' not in scene`, `Step 1 FAILED`, then
`[DRY RUN] go_home`; `/arm_status` shows `FAILED at step 1`.

**5c. Voice:** run `audio_node`, press Enter, say "pick up the cup", press Enter. Same pass
criteria as 5a.

---

## Stage 6 — First live motion, no object

Restart the arm controller live and slow:

```bash
ros2 run thesis_robot arm_controller --ros-args -p dry_run:=false -p speed:=0.10
```

Wait for `MoveIt2 initialised successfully — LIVE mode active`. Then send one move to a
clear point (this bypasses the LLM, so **you** are the safety check — confirm the point is
clear first):

```bash
ros2 topic pub --once /action_plan std_msgs/msg/String '{data: "{\"command\": \"test\", \"plan\": [{\"action\": \"move_to\", \"x\": 0.35, \"y\": 0.0, \"z\": 0.30}]}"}'
ros2 run tf2_ros tf2_echo base_link end_effector_link
```

**Pass:** the arm moves there, `/arm_status` shows `COMPLETE`, and the tf2_echo translation is
within ~1 cm of (0.35, 0.00, 0.30). If MoveIt refuses the orientation, note the error text.

---

## Stage 7 — Live pick (only after Stages 3 and 4 pass)

**7a. Stop above the object first.** Restart with a raised grasp height so the descend ends
6 cm above the detected point:

```bash
ros2 run thesis_robot arm_controller --ros-args -p dry_run:=false -p speed:=0.10 -p grasp_z_offset:=0.06
ros2 topic echo /scene_snapshot --once            # note the cup's object_id, e.g. cup_00
ros2 topic pub --once /action_plan std_msgs/msg/String '{data: "{\"command\": \"test\", \"plan\": [{\"action\": \"pick\", \"object_id\": \"cup_00\", \"approach_z\": 0.25}]}"}'
```

**Record:** where the fingertips stop relative to the cup — how far off in x/y (and
which direction), and the height above the rim. This is the end-to-end calibration error.

**7b. Full pick.** If 7a was centred to within the gripper's opening, repeat with
`grasp_z_offset:=0.0` (raise or lower it in 1–2 cm steps to reach a good grasp height).

**Pass:** the gripper closes on the cup and lifts it.

---

## Stage 8 — End to end by voice

With everything live, say *"pick up the cup"*. Then try a place:
*"pick up the cup and put it on the right side"*.

**Record:** success or failure, LLM latency, and — if it failed — the last lines of
`/planner_status`, `/arm_status` and `/pick_place_status`.

---

## What to send back

| Stage | Result | Numbers |
|---|---|---|
| 0 | pass / fail | calibration file present? |
| 1 | pass / fail | publisher count |
| 2b OAK-D | pass / fail | snapshot xyz, ground truth xyz, error |
| 2c RealSense | pass / fail | snapshot xyz, ground truth xyz, error |
| 3 | open = ?, closed = ? | |
| 4 | pass / fail | quaternion if different |
| 5 | pass / fail | LLM latency (s) |
| 6 | pass / fail | reached xyz |
| 7a | | x/y offset + direction, height above rim |
| 7b | pass / fail | grasp_z_offset that worked |
| 8 | pass / fail | latency, failure messages |
