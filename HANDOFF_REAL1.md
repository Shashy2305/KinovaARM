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
