# START HERE — current state and how to (re)start everything

Written 2026-10-28 (last commit before this file: `9666294`). Newer than README.md / HANDOFF_REAL1.md where they disagree. Machine: REAL-1 (Ubuntu 22.04, ROS 2 Humble), Kinova Gen3 7-DOF +
Robotiq 2F-140, RealSense RS1 / RS2 / RS3 + Kinova wrist camera, Ollama `qwen2.5:7b`, dashboard backend :8000 / frontend :5173. Repo: https://github.com/Shashy2305/KinovaARM (branch `main`).

## 0. The rules that cost us the most

* **Ask the people around before anything that moves the arm; never kill another account's processes.** The `kinova` account shares the arm and cameras (its own RealSense / wrist sessions).
* **Everything started from your desktop session dies when that session logs out** (2026-10-27 13:41 -> backend, robot bringup, cameras, detectors, arm controller all stopped). After a logout or reboot run section 2.
* **Risky features default OFF** (`side_grasp: off`, `approach_tilt_deg: 0.0`, `use_unknown_obstacles: false`, `dry_run` unless "MAKE IT LIVE"). Turn them on one at a time, watch, keep the rollback in mind.
* **Go home first**, then pick; do not give a command while the gripper holds something unless it is "put it down" (the planner refuses a pick / go home while holding).
* No sudo for Claude. `ros2` CLI hangs ("rclpy.ok()" errors) -> `ros2 daemon stop` and retry. Do not `pkill -f <pattern>` (matches your own shell); kill by exact PID.

## 1. What exists now (short)

| Area | State |
|---|---|
| Voice -> plan -> motion | works end to end; typed commands through the dashboard "Command" box (`POST /api/command {"text": ...}`) |
| Pick / place | verified live on cup, mug, mouse, bottle; relational commands ("pick the cup and put it next to the mouse") 3/3 |
| Put-down after a leaned grip | FIXED 2026-10-26: the lean is now held through lift / carry / place (a leaned cup stood upright in the gripper dropped). 30 deg pick + put-back worked live |
| Side grasp + method chooser | BUILT, unit-tested (452 tests), reach-studied, **NOT yet tried on the arm** (section 4) |
| Dashboard | jog (hold-to-move, server-side limits), calibrate-from-arm, health banner with fix buttons, camera multi-view |
| Calibration | RS1, RS2 levelled to the table; RS3 calibrated but its depth is ~42 % valid so RS3 detections stay OFF; wrist pose verified (4.5 mm scatter) |
| Shadow detector (detector_b) | built, off by default; start from Node Control, review with `scripts/detector_b_report.py` |

## 2. Cold start / restart (copy-paste, in this order)

Check the arm is reachable and powered first: `ping -c1 192.168.1.10` (Ethernet cable `enp2s0` has dropped before: reseat it).

```bash
# 0. the workspace image must be mounted (fstab automount): 
df -h /mnt/ros_workspace          # must show /dev/loopN ~1 TB; if empty: see docs/RUNBOOK.md section 7 / memory "shared machine"
nvidia-smi -L                     # GPU must be listed (else detectors run on CPU and starve the backend)
systemctl is-active ollama        # must print "active"

# 1. dashboard only (no hardware): backend :8000 + frontend :5173
bash /mnt/ros_workspace/Shashproject/scripts/start_dashboard.sh

# 2. open http://localhost:5173 , then press in this order (or use the health banner buttons):
#    Robot bringup  ->  wait for "You can start planning now" in the log  ->  Cameras bringup  ->  Full bringup
#    (or all of it from a shell:  bash /mnt/ros_workspace/Shashproject/scripts/start_everything_real1.sh )
```

Make the arm controller LIVE (it starts in dry run): dashboard "Make it live" (exact phrase `MAKE IT LIVE`), or

```bash
curl -s -X POST localhost:8000/api/arm_controller/go_live -H 'Content-Type: application/json' -d '{"confirm_phrase":"MAKE IT LIVE"}'
```

Health check any time: `bash scripts/check_system.sh` (read-only, prints the fix for each FAIL) and `curl -s localhost:8000/api/health/issues`.
Shell environment for any ROS command: `source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh` (domain 42). Tests must NOT use domain 42:
`ROS_DOMAIN_ID=77 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/thesis_robot/test -q` (expect: all pass except test_flake8 / test_pep257, pre-existing).

Restart only the backend after editing `dashboard/backend`: `bash scripts/restart_dashboard_backend.sh` (ROS nodes keep running). Restart only the arm controller after editing
`arm_controller_node.py` (the install imports the source tree through `build/`; check `python3 -c "import thesis_robot.arm_controller_node as m; print(m.__file__)"`): run the go_live call above.

Logs: `~/.ros/dashboard_logs/<node>.log`, `/tmp/dashboard_backend.log`, `/tmp/dashboard_frontend.log`; outcome log `~/.ros/outcomes/` (`scripts/outcome_report.py`).

## 3. Switches and parameters (set on the running node, no restart)

```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
ros2 param set /arm_controller side_grasp only          # off (default) | auto | only
ros2 param set /arm_controller approach_tilt_deg 30.0   # lean; 0.0 = straight down (default). tilt_cap_deg caps it (30 now; step 35 then 45, one watched pick each)
ros2 param get /arm_controller dry_run                  # must be False for a live pick
```
Shipped defaults live in `config/arm_controller_params.yaml`; the node default must equal the yaml (test_param_files).

## 4. NEXT: first live test of the side grasp (not done yet)

Demonstration the geometry is copied from: `docs/demonstrations/cup_side_grasp_2026-10-26/` (tool horizontal 5.3 deg below, az 269, pads under the rim, cup upright).
Design and reach numbers: `docs/ANGLED_APPROACH.md` sections 8-9; code: `src/thesis_robot/thesis_robot/grasp_options.py`, `descend_side` / `choose_method` in `arm_controller_node.py`.

1. Bring everything up (section 2), arm controller LIVE, gripper empty, cup on the table (best near x 0.35-0.45, y -0.15..0), nothing near it, hand clear.
2. `ros2 param set /arm_controller side_grasp only` and `approach_tilt_deg 0.0`.
3. Dashboard command: `pick up the cup` — watch: swing to a point 12 cm above/behind the cup, straight down, straight in, close, straight up. Log: `grep -E "grasp methods|side grasp|IK" ~/.ros/dashboard_logs/arm_controller.log | tail`.
4. `put the cup back on the table` — it keeps the horizontal tool and lowers to the recorded height. If the cup is not gentle: STOP, `side_grasp off`.
5. Then `side_grasp auto` and a second object next to the cup: the ranking (side az / top / lean, with clearance) is printed as "pick: grasp methods ranked: ...".
Rollback: `ros2 param set /arm_controller side_grasp off`. If a pick fails after the arm moved, the cup may be held: use "put it down" or open the gripper by hand only when the cup is supported.
Reach study (read-only, arm does not move): `python3 scripts/side_grasp_study.py` — about 13-14 of 20 grid positions have a reachable azimuth; the others fall back to top-down.
Known limitations: the carry check only covers the held object's footprint plus height thresholds (a tall bottle near the horizontal gripper is avoided by height, not by exact shape); handle bearing needs the wrist shape analysis (none = no handle rule).

## 5. Open items

* Step the lean to 35 then 45 deg only after the side grasp / put-down is judged; the side grasp may make the lean unnecessary.
* RS3 depth (firmware 5.17.3.10 vs 5.17.0.10 on the others? projector dots are visible on a phone camera); RS3 detections stay off until fixed.
* RS1 / RS2 shared-object x/y (yaw) disagreement 1-4 cm; more shared objects would settle it.
* `arm_controller` uses ~50 % CPU (1 kHz joint states in Python); not touched.
* Scene may show ghost objects (red_obj / blue_obj); the planner only uses confident, non-stale ones.
* Mic / Whisper: see `docs/RUNBOOK.md` section 9.
* Do not enable systemd linger (declined); a workspace backup was declined (code is on GitHub, calibration in `~/.ros`).

## 6. Where things are

* Calibration files `~/.ros/<camera>_calibration.yaml`, table `~/.ros/table_geometry.yaml`; tools in `calibration/` (wizard_consensus, rescale_wizard_result, level_to_table, check_arm_depth, check_wrist_board).
* Docs: `docs/RUNBOOK.md` (full procedures), `docs/ANGLED_APPROACH.md`, `docs/FUSION.md`, `docs/DETECTOR_SECOND_OPINION.md`, `docs/TEST_PLAN_2026-10-20.md`, `TESTING.md`, `HANDOFF_REAL1.md` (older status).
* Claude's memory notes (context for a new session): `~/.claude/projects/-mnt-ros-workspace/memory/` (index MEMORY.md).
* Commit style: `git -c user.name="Shashwat Shah" -c user.email="sshah260@ucr.edu" commit --author="Shashwat Shah <sshah260@ucr.edu>"`, message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## 7. Added 2026-10-28 (after the first version of this file)

* **Gripper stuck half closed** (finger 0.31 on nothing, e.g. after the object was taken out by hand) used to refuse every pick ("already holding something") and place ("do not know how the object
  is held"). Now (arm_controller): a half-closed gripper with nothing held and no scene object within 12 cm of the pads is opened automatically before a pick/place (`auto_release_empty_gripper`,
  default true; needs a fresh, non-empty scene), and the held-object record is forgotten once the fingers have been open for 1 s (so the planner stops refusing). Manual way out:
  type `open the gripper`, or the Gripper buttons in the "Jog the arm" tab.
* **"Jog the arm" tab** now also has the multi-camera view and a Gripper panel (Open / Close / go to a position; needs the jog ON = the typed phrase `area clear, hands off`; gamepad: hold LB, A closes, Y opens).
  Backend: `POST /api/jog/gripper {"action": "open"|"close"|"set", "position": 0..0.695}`.
* **Mic**: the voice node uses the system default input (`-p input_device:=pulse`; the external USB mic "Jieli USB Composite Device" is the PulseAudio default). Check: `pactl info | grep "Default Source"`.
* **RS3 detection** publishes bad heights (z -0.02 .. -0.19) and creates ghost objects; the full bringup starts it anyway: stop it with `curl -X POST localhost:8000/api/nodes/realsense3_detection/stop` until RS3 depth is fixed.

## 8. Added 2026-10-29

* **Planner workspace widened** (`safety_geometry.PLAN_X_RANGE` 0.07..0.65, `PLAN_Y_RANGE` -0.50..0.50; was 0.10..0.60 / +-0.35; `~/.ros/workspace_bounds.yaml` x max 0.65). The horizontal reach circle (0.85 m) still applies. The planner prompt now takes the limits from there. Restart scene_graph_node, llm_planner_node and arm_controller after changing them. These limits are in the ROBOT frame: moving a camera does not move them (only the camera's calibration decides where it SEES an object).
* **Live 3D robot** (dashboard Overview and Jog tabs, `RobotView.tsx`): the arm and gripper move as the real arm does, over the table, the planner workspace rectangle and the scene objects. Backend: `GET /api/robot/pose`, websocket `/api/ws/robot` (read-only, forward kinematics of the live joints, ~15 Hz). The finger position comes from the arm controller on `/gripper_finger` (5 Hz); without the arm controller the gripper is drawn open.
* **Side grasp is live-tested** (see ANGLED_APPROACH.md 8-9 and the memory note): `side_grasp: only` picks and puts back the cup. Fixes found live: joint-space short moves with a horizontal tool, whole-motion IK check, place without lift and retreat along the tool axis, never turn the wrist of a side-held object, held record in `~/.ros/held_object.json`.

## 9. Camera recalibration after the lab moved the cameras (2026-10-29)

* **Moved:** all three static cameras. RS2 ~25 cm; RS1 turned 8.7 deg (position ~2 cm). RS3 moved twice (now ~0.55 m to the side of the table, 0.9 m high).
* **Recalibrated + installed:** RS2 (wizard captures 1,3,4,5,6, -4 cm along its view axis from the arm ruler, levelled to the table) and RS1 (captures 1,2, -4 cm axis, levelled: tilt 3.4 deg). Backups: `~/.ros/calibration_candidates/*before-moved-20261029.yaml`. Both verified by `calibration/check_arm_depth.py` (RS1 med 6.8 / spread 1.8, RS2 med 6.1 / 2.7) and drawing the arm into each image.
* **Method lessons:** run `check_arm_depth.py` with the arm in an OPEN pose clear of the board (a low pose gave 5-19 cm spreads); the wizard reads distance along the view ~4 cm short (apply the shift that centres the arm-ruler median at ~5.5 cm); always `level_to_table.py` too (the arm ruler cannot see tilt). New tools: `calibration/record_shared_objects.py` (two cameras' readings of the same objects) and `calibration/fit_camera_to_reference.py`.
* **Finding (important):** after both cameras passed every placement check, RS1 and RS2 still disagreed by ~9 cm on table objects. The wrist camera (calibrated to the arm) showed why: a depth camera reads the NEAR SURFACE, the arm needs the centre; each static camera's error was along its own view direction, toward itself (radius + ~2 cm). `scene_graph_node` now moves static-camera points horizontally away from the camera by the object's radius (`SURFACE_CENTRE_FACTOR`, 1.0; 0 = off). Fused scene vs wrist: cup 0.8 cm, mouse 3.4, bowl 3.5. A rigid object-based correction of RS1 to RS2 was REJECTED (it fitted the objects but failed the arm check by 47 cm).
* **RS3 recalibrated + installed** (new position t = 0.63, 0.53, 0.88 before the 6 cm shift): 5 wizard captures at the final position (spread 1.8 cm / 1.2 deg; the board must lie FLAT, aligned with the table and not too far or oblique, or RS3 cannot detect it), shifted 6 cm back along its view axis from the arm ruler (median 4.7 cm, spread 1.3, 78% within 5 cm, only ~18 depth points). NOT levelled: its depth is too noisy for `level_to_table.py` (tilt left unresolved). RS3 DETECTIONS STAY OFF. Verified by drawing the arm into its image (dots on every link). Backups `~/.ros/calibration_candidates/*before-moved-20261029.yaml`. The cameras are final: do not move them again without recalibrating.
* Restart `scene_graph_node` after changing calibrations.

## 10. Colour commands and "near the robot base" (2026-10-29)

* **Colour:** the RealSense detectors add `color` (white, black, grey, blue, red, ...) from the pixels of each box (`object_colour.py`: table brown removed, thresholds tuned against RS1's cyan cast); the scene graph keeps a confidence-weighted vote per object (a bad frame cannot flip it) and publishes `color` in `/scene_snapshot`. Commands: "pick up the white cup", "put the mouse next to the blue cup", "pick up the blue cup and put it next to the white cup" (blue/white/black/grey/red/green/yellow/orange/purple/pink/brown; teal/navy = blue, silver/gray = grey). A plan that picks the wrong colour is rejected and repaired; an unknown colour is refused with "the cups I see are blue (cup_00), white (cup_01)".
* **"near the robot base":** "pick up the white cup and put it near the robot base" (also "next to / to the base"): `{"place","near":"base"}`, resolved to a free spot 0.27-0.36 m from the base axis, the one nearest to where the object was picked up (`BASE_RINGS_M` in `llm_planner_node.py`).
* Restart `realsense_detection`, `realsense2_detection`, `scene_graph_node` and `llm_planner_node` after changing any of this. RS3 detections stay off.

## 11. Incident 2026-10-29 (blue cup, side grasp): what happened and the fixes

* **What happened:** a cup at x = 0.58 was given a side grasp (ranking only had preferences) whose IK needed the wrist turned 180 deg and a 28 s swing; the controller aborted it and the robot body touched the table. The path check only watched the gripper and wrist-end links.
* **Fixes:** (1) `safety_geometry.GUARDED_LINKS` now includes the forearm, half-arm-2 and both wrist links with their own table clearances (`LINK_CLEARANCE_M` 0.075-0.08 m on the frame heights; the verified poses keep them at 10-20 cm); (2) a side swing whose planned path takes > `SIDE_SWING_MAX_S` (14 s; verified ones take 7-8 s) is refused BEFORE it moves and the next option / straight down is used; (3) the side grasp is only ranked inside `grasp_options.SIDE_ZONE_X/Y` (x 0.30-0.52, y +-0.22, where it was proven); elsewhere the pick goes straight down; `side_grasp: only` still forces it for tests.
* **White cup (top-down, handle along the fingers):** a cup whose handle is detected is aligned from the handle (handle-aware ranking) instead of the outline orientation. `side_grasp` is now `auto` by default (code + yaml): no toggling; it chooses per object, side only inside the proven zone and with the swing gate, otherwise straight down. Placement spots are capped at 0.66 m from the base (`PLACE_MAX_RADIUS_M`).

## 12. Position-aware learning (2026-10-30)

`grasp_options.py` learns per PLACE: for each class and method it keeps a distance-weighted success rate (Gaussian kernel 10 cm) from the outcome log (`xy`, `method`, `ok` of each live pick), blended with the method's overall rate where there is little nearby data (`local_rate`, `method_factor(..., xy)`). The side grasp's allowed region is the seed zone plus a TRUST REGION that grows (within 12 cm of a place where a side grasp worked, local rate >= 0.5) and shrinks (a place whose local rate falls below 0.35 is closed, even inside the seed zone): `side_allowed()`. The pick log prints "learned here: ...". Dashboard Overview > "Learned grasp success on the table" draws the same numbers as a heat map per class and method (`GET /api/outcomes/learning?label=cup`). Still fixed rules, not learned: the whole-arm table guard, the 14 s swing limit, the 17 cm standoff. Not learned: calibration, reach.
