# STATUS 2026-10-09 (newest; read this, then the 2026-10-06 status below)

**Everything below this block was verified on the real arm on 2026-10-06..08; the overnight 2026-10-09 work was built and tested WITHOUT moving
the arm (Dry Run, unit tests, stub nodes, real camera frames, MoveIt IK/FK). Start with `docs/TEST_PLAN_2026-10-09.md`** (staged, with switches and rollbacks).

New since 2026-10-06 (all pushed; commits up to `e674ecd` and the docs commit after it):
- Relational commands work live: "pick up the X and put it next to the Y" (3/3 on 2026-10-08/09, after fixing: carry height from the
  held object's underside, carry detour around obstacles, set-down no lower than the closed pads allow, the lifted object's own scene
  entry never blocks the release, planner must return a pick/place, wrist centring tracks the object it locked onto).
- Minimum-jerk trajectory smoothing on every move (peak jerk 40.5 -> 7.4 rad/s^3 measured); `smooth_free_vel_scale` for faster big swings.
- Outcome log (`~/.ros/outcomes`, `scripts/outcome_report.py`), grasp policy learned from it (`grasp_policy.py`).
- Shape analysis (`shape_analysis.py`: contour, minAreaRect, approxPolyDP, hull/defects/holes, ellipse, Hough, width + contact profiles) on the wrist
  detections; grasp-angle planner (`grasp_planner.py`) with retry from a NEW angle (ON for retries); measured-width preshape (off).
- Planner: `command_grammar.py` verifies/repairs plans, `planner_memory.py` adds a worked example; `scripts/planner_benchmark.py` (84% -> 100%).
- Unknown-obstacle guard from depth (`obstacle_guard*.py`, `/unknown_obstacles`); consumed behind `use_unknown_obstacles` (off).
- Angled approach: feasibility study + geometry only (`docs/ANGLED_APPROACH.md`); not wired.

**Shared-machine facts learned the hard way (2026-10-08/09)**
- `/mnt/ros_workspace` is an ext4 image on an external exFAT USB drive; a USB-controller reset detached it (I/O errors everywhere). Recovery recipe and
  the fix suggestion (copy the image to the internal disk) are in the memory notes / docs; the host `/` filled to 99% from other users' data.
- Another account (`kinova`) runs its own RealSense/ArUco/wrist-camera sessions on the same cameras: agree on who uses the cameras before starting the stack;
  never kill their processes. They also move cameras and the arm; re-check extrinsics (`calibration/check_extrinsics.py`) when anything looks off.
- Restarting the dashboard backend (kill -9) stalls the camera publishers for a few seconds; the watchdog limit is now 15 s so it no longer flags both cameras.
- Hand-guiding the arm while it is under ROS control faults it (`/fault_controller/internal_fault` true: trajectories are accepted and time out).
  Clear it with `ros2 service call /fault_controller/reset_fault std_srvs/srv/Trigger` (the call may not return; read the flag again), with hands off.
- Heavy CPU load (a stray `du`, a multi-threaded numpy node) can freeze `ros2_control_node`; restart `robot_bringup` from the dashboard (no motion).

# STATUS 2026-10-06 (read this first; the sections below are from the migration and partly dated)

**What works, verified on the real arm** (typed commands through the dashboard; voice pipeline is built but the mic is broken):
- `pick up the <mug|mouse|plastic bottle>` -> hover, wrist camera centres on it, wrist turned so the short side is between the
  fingers, straight descent, close, 3 cm lift + grip check, lift, one automatic retry after a miss/drop.
- `... and put it down | aside | next to the <object> | at x y` -> carry, lower, release, retreat; free-spot search keeps 11 cm
  from everything and checks the open fingers will not sweep a neighbour.
- Trial counts: mug 8+, mouse 6, plastic bottle 3 (all after the fixes below). Not graspable by this gripper: steel flask
  (smooth + tapered shoulder), bowls (wider than the 11.4 cm finger gap).
- Full recipes, safety checks and tuning knobs: `docs/RUNBOOK.md` sections 15-16 and the trial-series note at its end.

**The big lessons** (each cost real time; do not rediscover them):
1. The old static camera calibrations were garbage (anchored through a wrong wrist hand-eye). Recalibrated from clicked arm
   landmarks (`calibration/pnp_extrinsics.py`, dashboard "Calibrate from arm"). Check with `calibration/check_extrinsics.py`.
2. From straight above the detectors mislabel things (bottle -> "sports ball"/"bowl", mouse -> "cup"). The arm matches wrist
   detections by POSITION (within 6 cm of the scene object), the label is only a tie-break. Never filter wrist detections by class.
3. The cameras also see the lab: the scene now keeps only what is above the recorded table, drops detections on the robot's own
   gripper/arm, drops mislabelled duplicates within 6 cm, and expires ghosts (20 s stale, 60 s gone).
4. The scene's object height is unreliable (a cup read z=0.132, real 0.05): `pick_heights` ignores a z more than 4 cm from the
   object's known size. Tall objects are grasped low on the straight body, hovered over with the fingertips 7 cm above the top.
5. Closing the fingers drops the pads ~2 cm, so the path check lets a path START below the table margin but never go lower.
6. Every controller restart returns it to Dry Run; the operator re-confirms Go Live. Status is re-published every 2 s so the
   dashboard never shows a stale LIVE. Check knobs without restarting: `ros2 param set /arm_controller <name> <value>`
   (`sweep_same_object_m`, `sweep_half_span_m`, `sweep_margin_m`, `carry_avoid_m`, `low_pick_tip_clearance_m`, ...).
7. The dashboard Stop/Start used to match `curl .../api/nodes/<id>/stop` command lines and kill its own caller; fixed.
8. A disk filled by `scene_graph_node` logging every callback (7 GB of ROS logs); throttled. If disk is full: `~/.ros/log`.

**Known gaps / state**
- OAK-D is OFF on purpose (`config/disabled_cameras.txt`): it enumerates but stays in the depthai BOOTLOADER state (marginal USB).
  Try the driver's USB2 mode / another cable, then delete the `oakd` line. Detection and placement work fine on the two RealSense
  cameras + wrist.
- Microphone hardware is broken; the voice path (audio_node, faster-whisper small.en CPU, "ALC897 Analog" device) is untested since.
- Unknown objects (a black box on the table, a laptop) are not in the scene, so the planner does not avoid them.
- The planner (qwen2.5:7b) pads plans; `llm_planner_node.py` strips unrequested go_home/open_gripper/move_to, fixes pick heights and
  targets deterministically. Always dry-run a new kind of command first (arm_controller starts in Dry Run).
- Orphan static_transform_publisher processes from old bringups are harmless duplicates; `scripts/check_system.sh` reports them.
- Not built: angled (non-vertical) approaches, grasping bowls/flasks, collision objects for unknown items in MoveIt.

**Daily start**: `bash scripts/check_system.sh` (0 FAIL expected), dashboard -> Go Live (type the phrase) only with the E-stop in
reach and the table clear. Dry-run any new command shape first.

---

# Session Handoff — continuing on REAL-1 directly

This repo was migrated from the lab laptop to REAL-1 (Ubuntu 22.04.5, i9-13900KF,
RTX 4080). This file exists so a **fresh** Claude Desktop Code session started
directly on REAL-1 can pick up full context in one read, since sessions don't
transfer automatically between machines.

## How to use this file
Open this project folder (`/mnt/ros_workspace/Shashproject`) in Claude Desktop's
Code tab, running the app natively on REAL-1. In the new chat, say something like
"Read HANDOFF_REAL1.md and continue from there." Point it at this file.

## REAL-1 environment facts
- Repo: `/mnt/ros_workspace/Shashproject` (rsync'd from GitHub `Shashy2305/KinovaARM`,
  not git-cloned — no `gh` auth on REAL-1 for the private repo yet. Push changes
  from the lab laptop checkout, or set up `gh auth login` on REAL-1).
- ROS workspace: `/mnt/ros_workspace/ros2_kortex_ws` (colcon build regenerated
  locally; `src/` rsync'd).
- Python venv: `/mnt/ros_workspace/venv` (`--system-site-packages`, `numpy<2`,
  `setuptools<80,>=30.3.0` pinned).
- Ollama models relocated to `/mnt/ros_workspace/ollama_models` via systemd
  override; model in use: `qwen2.5:7b`.
- Node 20.20.2 via `nvm` — must `source $NVM_DIR/nvm.sh` in non-interactive shells.
- Dashboard: backend (`uvicorn app.main`) + frontend (`npm run dev -- --host 0.0.0.0`),
  started via `scripts/start_everything.sh` (see that script for exact env vars:
  `SHASHPROJECT_REPO_ROOT`, `SHASHPROJECT_WORKSPACE_ROOT`, `SHASHPROJECT_VENV_ACTIVATE`).
  Reachable at `http://<REAL-1-IP>:5173` (frontend) / `:8000` (backend API).
- **Gotcha**: after any `colcon build` of `thesis_robot`, these entry-point
  scripts' shebangs must be re-patched to the venv python (heavy deps —
  scipy/numpy/torch/ollama — aren't on system Python):
  `object_detection`, `realsense_detection`, `realsense2_detection`,
  `wrist_detection`, `llm_planner_node`, `audio_node`, `scene_graph_node`.
  Pattern: `sed -i '1s|.*|#!/mnt/ros_workspace/venv/bin/python3|' <installed script path>`.
- **Gotcha**: `controller_manager` deadlocks if any node's `RMW_IMPLEMENTATION`
  differs from the rest (was fixed by removing a stray CycloneDDS override on
  `ros2_control_node` alone — see commit "Fix controller_manager service
  deadlock: don't mix DDS vendors"). Never set RMW per-node again.
- **Gotcha**: if `/mnt/ros_workspace` is a loop-mounted image on the external
  SanDisk drive, always `sudo umount /mnt/ros_workspace` before touching ANY
  USB connection on this machine — unplugging while mounted corrupted it once
  (recovered via `e2fsck -y` on the image file).
- Kinova Gen3 arm: only one control session at a time (ROS `robot_bringup` XOR
  a physical pendant/joystick). Stop `robot_bringup` before jogging by hand.
- All 3 static cameras (OAK-D, RealSense, RealSense2) + wrist camera are
  calibrated on REAL-1 (`~/.ros/{oakd,realsense,realsense2}_calibration.yaml`).
- PortAudio installed; confirmed working mic input (`HDA Intel PCH: ALC897`,
  device index 20) — audio_node.py not yet wired up for live use (see Pending).

## Recently completed (this work session)
- Full pipeline migration + controller_manager deadlock fix (DDS mismatch).
- Fixed dashboard `process_manager.py` false-"conflict" status bug.
- Fixed hardcoded `REPO_ROOT` in `ros_bridge.py` / `calibration.py` (now import
  from `config.py`, which is env-var overridable).
- Fixed headless crash in detection nodes' debug preview windows
  (`object_detection.py`, `realsense_detection.py`, `realsense2_detection.py`).
- **Scene graph occlusion-confusion fix** (`scene_graph_node.py`): replaced
  greedy nearest-match with Hungarian optimal assignment
  (`scipy.optimize.linear_sum_assignment`) per label, excluding stale tracks
  from match candidates, plus GC of tracks stale >300s, plus a
  `MAX_PLAUSIBLE_REACH_M=2.0` filter rejecting implausibly-distant detections
  (was seeing false-positive color-fallback detections 10-18m away). Verified
  with 6 passing isolated tests + live on REAL-1.
- `llm_planner_node.py`: taught the LLM when to use `move_to` (go near/look at,
  no gripper) vs `pick` (actually grab), fixed a stale x-bound in the prompt
  text (0.55 → 0.60, matching the real validator), and excluded stale objects
  from the prompt entirely (not just a soft warning).
- `scripts/start_everything_real1.sh`: idempotent one-shot startup script,
  tested working.

## In progress / where this was interrupted
Defining `~/.ros/workspace_bounds.yaml` on REAL-1 via the dashboard's
Workspace Boundary tool (it was never created — `scene_graph_node` was
falling back to an overly-permissive default z-range, letting physically
impossible negative-z "cup" detections show as `reachable: True`). Doing this
via the OAK-D static camera view (not the wrist camera, to avoid needing
someone to physically jog the arm). Confirmed via direct API test that the
backend tool works correctly; a stray diagnostic point was already reset.
**Next step**: through the actual dashboard UI, click 2+ corners of the real
reachable table area on the OAK-D tile (avoid the area where a person/laptop
sits in frame), click Save, then restart `scene_graph_node` and verify
previously-impossible negative-z objects stop showing `reachable: True`.

## Pending tasks (explicit user requests, not yet done)
1. Finish the workspace boundary (above), then run the live test: **say a
   command like "go near the cup" and confirm the arm moves near wherever the
   cup currently is** (dynamic, from the live scene graph — not a fixed
   location).
2. Wire up real audio/voice input: adapt `audio_node.py` to VAD mode (not
   push-to-talk) with a faster/smaller Whisper model than `large-v3`, register
   it in dashboard `config.py`'s `PROCESSES`, surface `/audio_status` in the UI.
3. Build/verify a live "thinking" animation in the dashboard reflecting
   audio→parse→plan→motion→complete pipeline stages (`CommandPipeline.tsx`
   partially exists from earlier work — needs audio-stage integration and
   live verification on REAL-1).
4. Verify/demo the existing 3D Fusion merged-camera-view tab
   (`FusionView.tsx`) live on REAL-1 — built earlier, not yet re-tested here.
5. Lower priority, explicitly deprioritized by the user in favor of #1/#2/#3:
   define a small set of fixed named "places" for voice-commanded navigation.
6. General ongoing goal: keep live-testing to surface and fix real bugs —
   demo was delayed only a few days, time pressure is real.

## Useful context for whoever (or whichever Claude session) picks this up
- The user (Shashwat Shah, UCR grad student, advisor Prof. Mingyu Cai) is
  working day/night toward a demo under time pressure.
- Prior research detour: LingBot / RTAB-Map / a labmate's "combined cameras"
  fork were all evaluated and judged not directly applicable — the real,
  concrete bug was scene-graph occlusion confusion (now fixed, see above) and
  the lack of dynamic object-position targeting (in progress).
- Full prior conversation transcript (for deep historical detail only) is on
  the lab laptop at:
  `/home/lab/.claude/projects/-home-lab-Shashproject/280c227c-677e-448c-aa71-02b75d50872d.jsonl`
