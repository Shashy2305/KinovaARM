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
