# KinovaARM runbook

Everything needed to start, check, run, fix and shut down the system **without
Claude**. Commands are meant to be copied as-is. Every procedure ends with a
**Check** so you know it worked before moving on.

> **Fastest way to find out what is wrong right now:**
> `bash /mnt/ros_workspace/Shashproject/scripts/check_system.sh`
> It is read-only, prints `[ OK ]` / `[WARN]` / `[FAIL]` per item and, for every
> FAIL, the command that fixes it. Exit code = number of FAILs.

## Contents

1. [Rules that keep people and the arm safe](#1-rules-that-keep-people-and-the-arm-safe)
2. [Machine map](#2-machine-map)
3. [Shell setup](#3-shell-setup)
4. [Cold start (power on to ready)](#4-cold-start-power-on-to-ready)
5. [One-time / after-change setup: record the table](#5-one-time--after-change-setup-record-the-table)
6. [Running commands (dry run first, then live)](#6-running-commands-dry-run-first-then-live)
7. [After an E-stop or a crash](#7-after-an-e-stop-or-a-crash)
8. [Shutdown](#8-shutdown)
9. [Voice input](#9-voice-input)
10. [Troubleshooting matrix](#10-troubleshooting-matrix)
11. [Developing: edit, restart, test, commit](#11-developing-edit-restart-test-commit)
12. [What each safety guard does](#12-what-each-safety-guard-does)
13. [Useful probes](#13-useful-probes)
14. [Known gaps (not verified on hardware)](#14-known-gaps-not-verified-on-hardware)
15. [Camera extrinsics: check and fix](#15-camera-extrinsics-check-and-fix)

---

## 1. Rules that keep people and the arm safe

1. **Dry Run is the default.** Restarting `arm_controller` always brings it back
   in Dry Run. Going live is a deliberate act in the dashboard (it asks for a phrase).
2. **The operator stands behind the robot** (the cable side, the side away from
   the table). Nobody stands at the sides. The software keeps the arm in front of
   x = -0.10 m for that reason.
3. **Hand on the E-stop for every first live motion** after any code, config or
   hardware change.
4. **Never send live motion while `check_system.sh` shows a FAIL** in sections
   3-6 (nodes, duplicate processes, joint states, safety state).
5. **If something looks wrong, E-stop first, diagnose after.** Section 7 says how
   to recover.
6. The dashboard has **no login**. Anyone on the lab network can open it and
   command the arm. Do not leave it live and unattended.

## 2. Machine map

| Thing | Value |
|---|---|
| Machine | REAL-1 (Ubuntu 22.04, i9-13900KF, RTX 4080) |
| Repo | `/mnt/ros_workspace/Shashproject` (GitHub `Shashy2305/KinovaARM`, branch `main`) |
| Kinova ROS workspace | `/mnt/ros_workspace/ros2_kortex_ws` |
| Python venv (YOLO, Whisper, ollama, scipy) | `/mnt/ros_workspace/venv` |
| Arm IP / PC interface | `192.168.1.10` / `192.168.1.100` on `enp2s0` |
| ROS domain | `ROS_DOMAIN_ID=42`, **one DDS vendor for everything** (never set `RMW_IMPLEMENTATION` for a single node) |
| Dashboard | frontend `http://<REAL-1 IP>:5173`, backend API `http://localhost:8000/api` |
| LLM | Ollama `qwen2.5:7b`, models in `/mnt/ros_workspace/ollama_models` |
| Speech | faster-whisper `small.en`, CPU int8, mic `HDA Intel PCH: ALC897 Analog (hw:0,0)` |
| Dashboard logs | `/tmp/dashboard_backend.log`, `/tmp/dashboard_frontend*.log`, per node `~/.ros/dashboard_logs/<node>.log` |

Files in `~/.ros/` that matter:

| File | Written by | Meaning |
|---|---|---|
| `table_geometry.yaml` | dashboard **Table safety** tab | table-top height + footprint. **Without it live motion is blocked.** |
| `workspace_bounds.yaml` | by hand / boundary tool | box the scene graph uses for "reachable" |
| `oakd_calibration.yaml`, `realsense_calibration.yaml`, `realsense2_calibration.yaml` | Calibration tab | camera to base_link TF |
| `<camera>_needs_recalibration.flag` | `camera_watchdog` | camera dropped and came back; its detections are blocked until recalibrated |

Which process runs which code:

| Process (dashboard name) | Code | Loads new code when |
|---|---|---|
| `arm_controller`, `llm_planner_node`, `scene_graph_node`, `audio_node`, detectors, watchdog | `src/thesis_robot/thesis_robot/*.py` (symlink install, **no rebuild needed**) | you restart that node |
| dashboard backend | `dashboard/backend/app/` | `scripts/restart_dashboard_backend.sh` |
| dashboard frontend | `dashboard/frontend/src/` | automatically (Vite hot reload) |
| MoveIt config (SRDF, planners) | `ros2_kortex_ws/src/ros2_kortex/kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_140_moveit_config/config/` (symlinked into `install/`) | restart `robot_bringup` |

## 3. Shell setup

Every terminal that talks to ROS:

```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
```

Harmless noise: `not found: ".../handeye_target_detection/.../local_setup.bash"` and the
`dai_ros_plugins` line.

If `ros2 ...` fails with `xmlrpc.client.Fault ... !rclpy.ok()` the ROS daemon is
confused:

```bash
ros2 daemon stop; ros2 daemon start
```

If it still fails, use the Python probes in [section 13](#13-useful-probes) instead.

## 4. Cold start (power on to ready)

### 4.1 Physical checks

- Arm powered, **E-stop released**, nothing in its reach, nobody at its sides.
- Ethernet cable to the arm connected.
- Table clear of anything you do not want the arm to touch.

**Check:**
```bash
ping -c 2 192.168.1.10
```
Expected: replies in about 0.2 ms. No reply = cable / arm power / E-stop.

### 4.2 Storage

If the workspace lives on the external SanDisk image:

```bash
mountpoint /mnt/ros_workspace && df -h /mnt/ros_workspace
```
**Never unplug any USB device while it is mounted.** To unmount first:
`sudo umount /mnt/ros_workspace`. (A yanked drive corrupted the image once; recovery was `e2fsck -y` on the image file.)

### 4.3 Language model

```bash
curl -s localhost:11434/api/tags | grep -o 'qwen2.5:7b'
```
Expected: `qwen2.5:7b`. If nothing: `systemctl status ollama`, then `sudo systemctl restart ollama`.

### 4.4 Start the dashboard (backend + frontend)

Everything at once (also starts robot bringup, cameras and the node stack; safe to re-run):

```bash
bash /mnt/ros_workspace/Shashproject/scripts/start_everything_real1.sh
```

Or only the dashboard:

```bash
bash /mnt/ros_workspace/Shashproject/scripts/restart_dashboard_backend.sh
```
```bash
cd /mnt/ros_workspace/Shashproject/dashboard/frontend && export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && nohup npm run dev -- --host 0.0.0.0 > /tmp/dashboard_frontend.log 2>&1 &
```

**Check:** open `http://<REAL-1 IP>:5173` (find the IP with `hostname -I`). The top
strip shows `LINK connected`. `curl -s localhost:8000/api/health` prints `{"ok":true}`.

### 4.5 Robot bringup

Dashboard: **Overview > Node control > Robot bringup > Start**, or:

```bash
curl -s -X POST localhost:8000/api/nodes/robot_bringup/start
```

Wait about 30 s.

**Check (all three must hold):**
```bash
bash /mnt/ros_workspace/Shashproject/scripts/check_system.sh | sed -n '/^4\./,/^6\./p'
```
- exactly **one** `move_group`, **one** `robot_state_publisher`, `ros2_control_node` alive
- `/joint_states flowing`
- if you see two of anything, see troubleshooting "Two move_group" (section 10)

### 4.6 Cameras, then the node stack

Dashboard: **Cameras bringup > Start**, then the **Full bring-up** button (starts TF
broadcasters, watchdog, detectors, scene graph, planner, voice, arm controller in Dry Run). Or:

```bash
curl -s -X POST localhost:8000/api/nodes/cameras_bringup/start
sleep 15
curl -s -X POST localhost:8000/api/bringup/full
```

Start the voice node if it is not in the bring-up (see section 9):
`curl -s -X POST localhost:8000/api/nodes/audio_node/start`

### 4.7 Verify everything

```bash
bash /mnt/ros_workspace/Shashproject/scripts/check_system.sh
```
Expected on a healthy day: no FAIL except possibly **table geometry** (until you
record it, section 5) and **wrist camera** if its driver is not running.

## 5. One-time / after-change setup: record the table

Live motion is **blocked** until `~/.ros/table_geometry.yaml` exists. It stores
the table-top height (in `base_link`, **not** measured from the floor; z = 0 is
about the robot's mounting plane and the table top is a few cm below it).

1. Dashboard > **Table safety**. The live fingertip pose must show numbers
   (if it says "No arm pose", the arm driver is down: run `check_system.sh`).
2. Keep the gripper **open**.
3. Hand-guide the arm so the **very tips of the open fingers touch the table**.
   The arm is rigid while `robot_bringup`'s controllers hold it; how to free it for
   hand-guiding is not documented here (see section 14). Do not force a held arm.
4. Press **Record point**. Do this at 4 corners of the usable area and the middle (at least 3 points).
5. Press **Save**. It refuses if the recorded heights differ by more than 3 cm
   (then not all points were touching the table; Reset and redo).

**Check:** the top strip's **TABLE** cell shows `top z -0.xx` (green), and
`curl -s localhost:8000/api/table_geometry/status` shows `"saved": {...}`.

The planner and controller re-read the file on every command, so no restart is needed.
After **moving the robot or the table**, record again.

## 6. Running commands (dry run first, then live)

### 6.1 Dry run (no motion)

The top strip's **ARM** cell must say `dry run`.

Type or speak a command in **Command console** (for example `go near the cup`) and watch
the pipeline: **Listen > Transcribe > Plan > Verify + Move**.

| What you see | Meaning / what to do |
|---|---|
| Plan `failed: ... violates floor ...` | the LLM aimed too low; just retry. If it repeats, the table geometry may be wrong |
| Plan `failed: no scene data yet` / `scene data is Ns old` | `scene_graph_node` is not publishing: check section 4.6 |
| Verify + Move `BLOCKED: no table geometry` | record the table (section 5) |
| `... no collision-free IK solution` | target is outside what the arm can reach with the gripper down, or too close to the table |
| `... needs a N rad single-joint reconfiguration` | the arm is in an awkward pose: send **go home** first |
| `... unsafe: ... below the table` / `... behind ...` | the safety check refused a path; nothing moved. Tell the developer which target caused it |
| `N move_group nodes are running (need exactly 1)` | duplicate processes (section 10) |

Voice commands are **shown for confirmation** ("Heard - confirm before the arm
moves") unless you tick *send voice commands without confirming*.

### 6.2 Going live

Preconditions (all must be true):

- [ ] `check_system.sh` has **no FAIL** (table recorded, joint states flowing, one `move_group`)
- [ ] a dry run of the same command succeeded
- [ ] area clear, **operator behind the robot with a hand on the E-stop**
- [ ] you know where the cup/target physically is

Then, in the dashboard, **Node control > Arm controller > Go live** and type the
confirmation phrase it asks for. The **ARM** cell turns red (`LIVE - MOVES`) with a
yellow/black hazard stripe. Restarting `arm_controller` for any reason drops it back to Dry Run.

**First live test after any change:** a small hover move such as `go near the cup`
at speed 0.10 (go-live starts at 0.10). Watch the whole path. Press the E-stop at the
first doubt.

## 7. After an E-stop or a crash

An E-stop (or a power blip) kills the arm's real-time driver:
`ros2_control_node` dies (log shows `sendto() failed with error code : 101` or
`Future already retrieved`) while the dashboard may still say `robot_bringup: running`.

1. Fix the cause. Release the E-stop. If the arm itself was powered off, power it on and wait for its LED.
2. Stop and restart the bringup (the Stop button now kills the whole process tree):
   ```bash
   curl -s -X POST localhost:8000/api/nodes/robot_bringup/stop
   sleep 5
   bash /mnt/ros_workspace/Shashproject/scripts/check_system.sh | sed -n '/^4\./,/^5\./p'
   ```
   There must be **no** `move_group` / `robot_state_publisher` / `ros2_control_node` left.
   Leftovers: see "Orphans" in section 10.
3. Start it again:
   ```bash
   curl -s -X POST localhost:8000/api/nodes/robot_bringup/start; sleep 30
   ```
4. Restart the controller so it reconnects (it returns in **Dry Run**):
   ```bash
   curl -s -X POST localhost:8000/api/nodes/arm_controller/stop; sleep 2
   curl -s -X POST localhost:8000/api/nodes/arm_controller/start
   ```
5. Run `check_system.sh`. Only then consider going live again (section 6.2).

## 8. Shutdown

1. Make sure the arm is at rest. Switch **arm_controller to Dry Run** (restart it).
2. Stop in this order (dashboard Stop buttons or):
   ```bash
   for n in audio_node arm_controller llm_planner_node scene_graph_node wrist_detection realsense2_detection realsense_detection object_detection camera_watchdog cameras_bringup robot_bringup; do
     curl -s -X POST localhost:8000/api/nodes/$n/stop; echo; sleep 2
   done
   ```
3. Check nothing is left over:
   ```bash
   ps -eo pid,ppid,lstart,args | grep -E 'move_group/move_group|robot_state_publisher/robot_state_publisher|ros2_control_node' | grep -v grep
   ```
   Expected: no lines.
4. Power the arm down with its own switch. Only then unmount/unplug storage (section 4.2).

## 9. Voice input

Dashboard > **Command console**: the mic button.

- Click once to record, click again to stop and transcribe (about 1.3 s).
- **hands-free** keeps listening and records when it hears speech.
- The level bars move when the microphone hears sound. Flat bars = no signal.

**Start / restart the voice node**
```bash
curl -s -X POST localhost:8000/api/nodes/audio_node/start
```
Check: top strip **VOICE** says `ready`; `~/.ros/dashboard_logs/audio_node.log` shows `microphone: HDA Intel PCH: ALC897 Analog (hw:0,0) @ 44100 Hz`.

**Which microphone**: by name via the node parameter `input_device` (default `ALC897 Analog`).
The system "default" / PulseAudio input on REAL-1 delivers **digital silence**, do not use it.
To change it edit the `audio_node` entry in `dashboard/backend/app/config.py`
(`-p input_device:='<name part or index>'`), then `scripts/restart_dashboard_backend.sh` and restart the voice node.

**Level too low** (bars barely move when you speak): raise the hardware boost, or the software gain.
```bash
alsamixer -c 0      # F4 = capture; raise "Front Mic Boost"
```
or add `-p input_gain:=4.0` to the `audio_node` command in `config.py`.

**Quick microphone test** (prints RMS per device; speak while it runs):
```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
python3 - <<'EOF'
import sounddevice as sd, numpy as np
for dev in (0, 20):
    x = sd.rec(int(2*44100), samplerate=44100, channels=1, dtype='float32', device=dev); sd.wait()
    print(dev, sd.query_devices(dev)['name'], 'rms=%.5f peak=%.4f' % (np.sqrt(np.mean(x**2)), abs(x).max()))
EOF
```
(Stop the voice node first; two programs cannot open the same capture device.)

**Whisper on GPU** is not enabled: the venv has CUDA 13 libraries but `ctranslate2` needs CUDA 12
(`libcublas.so.12`). CPU int8 is fast enough. Do not "fix" this unless you have a reason.

## 10. Troubleshooting matrix

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| Arm swings wildly / goes behind itself in planning | old default planner (random OMPL paths) | `grep -n pilz src/thesis_robot/thesis_robot/arm_controller_node.py` | the controller now uses seeded IK + Pilz PTP and an FK path check; make sure `arm_controller` was restarted after pulling |
| Fingers dive into the table | commanded z is the **wrist flange**, fingers hang 0.21 m below | `Verify + Move` should refuse low targets | record the table (section 5); never lower `flange_floor_z` |
| **Two move_group** / `N move_group nodes are running` | an old bringup was stopped without killing its children | `check_system.sh` section 4 | `ps -eo pid,ppid,lstart,args \| grep '[m]ove_group/move_group'` then `kill <older pid>`; same for `robot_state_publisher` |
| Orphans (`ppid 1`) after Stop | children survived | `ps -eo pid,ppid,args \| awk '$2==1' \| grep -E 'move_group\|robot_state\|ros2_control'` | `kill` them (add `-9` only if they ignore it). The dashboard Stop now kills the whole tree, so this should only happen after a crash |
| `No joint states` / `Could not find a connection between base_link and end_effector_link` | `ros2_control_node` is dead or the arm is not connected | `check_system.sh` sections 4-5 | section 7 |
| Planning `Start state appears to be in collision` | wrong SRDF (gripper collision matrix missing) | `grep -c finger .../config/gen3.srdf` should be about 69 | restore `gen3.srdf` from `gen3.srdf.backup2`, restart `robot_bringup` |
| `controller_manager` hangs / services time out | mixed DDS vendors | `echo $RMW_IMPLEMENTATION` in each node's env | never set `RMW_IMPLEMENTATION` per node |
| Wrist tile black, top strip `no signal: wrist` | kinova_vision driver not publishing. It retries forever if the arm's RTSP stream was not up at launch and then crashes (`ParameterAlreadyDeclaredException`, seen in `~/.ros/dashboard_logs/cameras_bringup.log`), and nothing restarts it | `ros2 topic info /camera/color/image_raw` shows `Publisher count: 0`; `gst-launch-1.0 -q rtspsrc location=rtsp://192.168.1.10/color latency=100 ! rtph264depay ! avdec_h264 ! videoconvert ! fakesink num-buffers=20` must exit 0 | start only the wrist driver, section 15.3 (do not restart the whole camera bringup) |
| A camera shows `needs recalibration` | it dropped and reconnected (bumped) | `ls ~/.ros/*_needs_recalibration.flag` | Calibration tab, recalibrate that camera |
| Detector node dies on start (Qt xcb error) | tried to open a preview window headless | log `~/.ros/dashboard_logs/` | already guarded by `DISPLAY` check; make sure you run the current code |
| Scene shows objects below the table (z about -0.28) or phantom `mouse` objects | camera extrinsics off, or false YOLO detections | dashboard scene view, `below_table` flag in `/scene_snapshot` | recalibrate; record the table so `below_table` appears and they stop counting as reachable |
| Plan hangs for a minute then everything ignores commands | Ollama hung | `curl -m 5 localhost:11434/api/tags` | `sudo systemctl restart ollama`; the planner now times out (60 s) instead of locking up |
| Mic bars flat, `no speech` every time | wrong input device (default = silence) | section 9 microphone test | set `input_device` (section 9) |
| Dashboard backend does not stop on `kill` | executor thread keeps it alive | n/a | `scripts/restart_dashboard_backend.sh` (does the safe sequence) |
| `navigate timed out` in the Claude browser tool | page keeps MJPEG/WebSocket connections open | n/a | open the URL in a normal browser |
| `git commit` says `Author identity unknown` | no git identity on REAL-1 | `git config user.name` | use one-off flags, section 11 |
| Pushing fails (auth) | no `gh auth` on REAL-1 | `gh auth status` | `gh auth login`, or push from the lab laptop |

## 11. Developing: edit, restart, test, commit

**Restart one node after editing its code** (no rebuild):
```bash
curl -s -X POST localhost:8000/api/nodes/<node>/stop; sleep 2; curl -s -X POST localhost:8000/api/nodes/<node>/start
```
`<node>` is one of: `arm_controller llm_planner_node scene_graph_node audio_node object_detection realsense_detection realsense2_detection wrist_detection camera_watchdog`.
`arm_controller` always returns in Dry Run.

**Backend:** `bash scripts/restart_dashboard_backend.sh`. **Frontend:** nothing (hot reload).

**Run the tests** (35 at the time of writing; pure logic + a real tf2 buffer; no robot needed):
```bash
cd /mnt/ros_workspace/Shashproject/src/thesis_robot
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q -p no:cacheprovider \
  --ignore=test/test_copyright.py --ignore=test/test_flake8.py --ignore=test/test_pep257.py
```
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids a broken `anyio` plugin in the venv.)

**Type-check the frontend:**
```bash
cd /mnt/ros_workspace/Shashproject/dashboard/frontend && export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && npx tsc -b --noEmit
```

**Commit and push** (this machine has no git identity; do not change the global config, pass it per command):
```bash
cd /mnt/ros_workspace/Shashproject
git status
git add <the files you changed>
git -c user.name="Shashwat Shah" -c user.email="sshah260@ucr.edu" commit -m "What and why"
git push origin main
```

**Planning-only check of the motion/safety code** (never executes; needs `robot_bringup` running):
```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
python3 -u -s /mnt/ros_workspace/Shashproject/scripts/dev/plan_only_test.py
```
Expected: six `ok` lines (swing <= ~2.2 rad, min link x >= +0.10) and three `REFUSED` lines. Run it after
touching `arm_controller_node.py`, `motion_utils.py` or the MoveIt config.

## 12. What each safety guard does

| Guard | Where | Effect |
|---|---|---|
| Table geometry required | `arm_controller_node.py`, `safety_geometry.py` | no live motion at all until `table_geometry.yaml` exists |
| Fingertip-aware floor | `safety_geometry.flange_floor_z` | target flange z >= table top + 0.215 (finger reach) + 0.05 |
| Table slab + rear wall in MoveIt | `_ensure_safety_scene` | planner and IK treat the table and the space behind the robot as solid; re-applied before every plan |
| Path check | `_check_trajectory` | every planned waypoint is forward-kinematics checked: gripper/wrist links must stay 5 cm above the table and in front of x = -0.10, else nothing executes |
| Seeded IK, yaw freedom, Pilz PTP | `_solve_goal_joints`, `motion_utils.py` | small repeatable motions instead of random wide swings; rejects goals needing > 2.6 rad on one joint |
| One `move_group` only | `_move_group_count` | refuses to move if duplicates exist |
| Fresh joint states | `_current_joint_vector` | planning from stale or missing joint states is refused (also catches a dead arm link) |
| No auto-home after failure | `_execute` | the arm stays where it is; you decide |
| Checked gripper | `_gripper` | a failed open/close fails the plan instead of continuing |
| Planner limits | `safety_geometry.PLAN_X_RANGE/PLAN_Y_RANGE` | one definition shared by planner, controller and scene graph |
| Stale scene refused | `llm_planner_node.py` | no planning on a scene older than 3 s |
| LLM timeout | `llm_planner_node.py` | a hung Ollama cannot lock up planning |
| Voice confirmation | dashboard `CommandConsole` | transcripts need a click to send unless auto-send is on |

Constants (all in `src/thesis_robot/thesis_robot/safety_geometry.py`): `TCP_REACH_M = 0.215`,
`TCP_CLEARANCE_M = 0.05`, `LINK_TABLE_CLEARANCE_M = 0.05`, `REAR_KEEP_OUT_X = -0.10`,
`REAR_WALL_DEFAULT_X = -0.20`, `FALLBACK_FLANGE_Z_MIN = 0.30` (used only when no table is recorded).

## 13. Useful probes

```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
```

**Are joint states flowing?**
```bash
python3 -s - <<'EOF'
import time, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init(); n = Node('probe'); got = []
n.create_subscription(JointState, '/joint_states', lambda m: got.append(m), 10)
t = time.time()
while not got and time.time() - t < 6: rclpy.spin_once(n, timeout_sec=0.2)
print(dict(zip(got[-1].name, got[-1].position)) if got else 'NO JOINT STATES')
EOF
```

**How many move_group nodes, and what is in MoveIt's scene?**
```bash
python3 -s - <<'EOF'
import time, rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPlanningScene
rclpy.init(); n = Node('probe'); time.sleep(2)
print('move_group nodes:', sum(1 for name, _ in n.get_node_names_and_namespaces() if name == 'move_group'))
c = n.create_client(GetPlanningScene, '/get_planning_scene'); c.wait_for_service(5)
r = GetPlanningScene.Request(); r.components.components = 1023
f = c.call_async(r)
while not f.done(): rclpy.spin_once(n, timeout_sec=0.1)
print('collision objects:', [o.id for o in f.result().scene.world.collision_objects])
EOF
```
Expected while the controller is live: `['rear_safety_wall', 'table_slab']`.

**Live scene as the planner sees it:**
```bash
curl -s localhost:8000/api/status | python3 -c "import json,sys; d=json.load(sys.stdin)['ros']['scene_snapshot']; [print(k, v['label'], 'x=%.2f y=%.2f z=%s' % (v['x'], v['y'], v['z']), 'reachable' if v['reachable'] else '-', 'BELOW TABLE' if v.get('below_table') else '') for k, v in d.items()]"
```

**Why did the arm controller refuse / fail?**
```bash
curl -s "localhost:8000/api/nodes/arm_controller/log?lines=40" | python3 -c "import json,sys; print(json.load(sys.stdin)['log'])"
```
The same works for `llm_planner_node`, `scene_graph_node`, `audio_node`, `robot_bringup`.
(`robot_bringup` logs get flooded by repeated hardware errors; read the file directly:
`grep -v 'No controller active' ~/.ros/dashboard_logs/robot_bringup.log | tail -50`.)

**Which process is the dashboard talking about?**
```bash
curl -s localhost:8000/api/nodes | python3 -m json.tool | grep -B1 -A3 '"status"' | head -80
```

## 14. Known gaps (not verified on hardware)

Be honest about these before a demo:

- **Live motion with the new guards has not been run on the arm.** They were
  verified by planning-only tests against the live MoveIt and by unit tests. The first
  live move after this change set must be done with a hand on the E-stop (section 6.2).
- **Hand-guiding while the arm publishes joint states** (needed to record the table)
  is untested. While controllers are active the arm holds position. If it will not move
  by hand, free it with the Kinova pendant/web app while `ros2_control_node` is still publishing,
  or tell the developer.
- **`pick` is not fingertip-aware.** `pick` targets the wrist flange at the object's height, so the
  new floor makes it stop above low objects; it will not grasp small flat things yet.
- **Gripper checking** (`_gripper`) was written but not run against the real gripper controller.
- **`x` max 0.60 / `|y|` max 0.35** are conservative planner limits; the table is larger
  (about 0.75 m forward, +-0.8 m sideways) but the arm's reach is the real limit.
- **Camera TF for the wrist** uses the image capture time; with the wrist driver offline this path is untested live.
- **Dashboard has no authentication** (section 1, rule 6).

## 15. Camera extrinsics: check and fix

Symptom: one physical object shows up as two or three objects (one per camera), 20-30 cm
apart, or the 3D view shows the table tilted. Detection is fine; the static cameras'
`~/.ros/*_calibration.yaml` poses are wrong. The pick will go to a wrong place, so fix this
before trusting any detected position.

### 15.1 Measure it (read-only, nothing moves)

Needs the camera drivers running and an object (a cup works) on the table in view of two cameras.

```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
python3 /mnt/ros_workspace/Shashproject/calibration/check_extrinsics.py --label cup
```

It prints, per camera, the tilt and height of the table plane (should be ~0 deg and the same
height everywhere) and how far apart the cameras place the same object (should be < 5 cm).
It ends with `OK` or a list of `PROBLEM` lines. Last measured on REAL-1: object spread 30 cm,
one camera's table tilted 46 deg.

### 15.2 Fix the static cameras from the arm itself (dashboard, no board needed)

Dashboard > **Calibrate from arm** tab. The arm is the calibration target: its joint positions are
known exactly, so you only tell the tool where known parts of the arm appear in a still image.

1. Pick the camera. Move the arm (by hand or any safe jog) to a pose that is spread out and clearly
   visible in that camera, e.g. extended over the table. The arm must stay still for the whole session.
2. The red circles show where the CURRENT (wrong) calibration thinks each landmark is. Ignore them.
3. Click, in the image, at least 5 of: base centre, shoulder hub, elbow hub, wrist hub, wrist flange,
   the two fingertips. Skip anything hidden. The two fingertips are interchangeable.
4. **Solve.** Check the RMS (under about 6 px is good). If it names a suspect landmark, re-click
   that one. Cyan circles show the solved projection; they should land on the real arm parts.
5. **Save candidate** writes `~/.ros/calibration_candidates/<camera>_calibration.candidate.yaml`.
   It does NOT change the live calibration. After comparing, install it yourself:

```bash
cp ~/.ros/calibration_candidates/realsense_calibration.candidate.yaml ~/.ros/realsense_calibration.yaml
```
then restart the TF broadcaster (Full bring-up restart or Stop/Start `static_tf_broadcaster`) and
rerun 15.1. Repeat per camera, with a different arm pose for a second view if the RMS is high.

### 15.2b Status on 2026-10-05

The three static calibrations were replaced using 15.2 (RealSense 1: 10 px RMS, RealSense 2: 8 px, OAK-D: 9 px).
The old files are in `~/.ros/calibration_backup_20261005`. They were wrong because they had been anchored through the
wrist camera's incorrect hand-eye transform (fixed in `gen3_macro.xacro`, section 15.3). After the swap the table
tilt went from 46 deg to 1.9 deg and the cameras agree on the cup within 5 cm. The OAK-D's depth is still poor at this
range (cup about 7 cm too high), and its `camera_info` distortion values are not usable. To install a new calibration
restart that camera's TF broadcaster from the dashboard. Switch the arm back to Dry Run with
`curl -s -X POST localhost:8000/api/arm_controller/go_dry_run`.

### 15.3 Wrist camera

The wrist camera is the kinova_vision driver, started as part of `cameras_bringup`. If the arm's
camera stream is not up when that launch starts, the driver retries forever and finally crashes, and
nothing restarts it (found 2026-10-05: `Publisher count: 0` on `/camera/color/image_raw`). Start just
the wrist driver, detached so it survives this shell:

```bash
source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
setsid nohup ros2 launch kinova_vision kinova_vision.launch.py device:=192.168.1.10 camera:=camera \
  launch_color:=true launch_depth:=true depth_registration:=false \
  max_color_pub_rate:=15.0 max_depth_pub_rate:=10.0 \
  "depth_rtsp_element_config:=depth latency=100 timeout=10000000" \
  "color_rtsp_element_config:=color latency=100" \
  > ~/.ros/dashboard_logs/wrist_camera.log 2>&1 < /dev/null &
```
The top strip should go back to `cameras ok` within a few seconds (about 14 frames/s on the colour topic).

Its hand-eye transform (the `camera_module` joint in the URDF) looks wrong: projecting the
gripper pads into the image fits the nominal mount but not the calibrated one. Check it with the
finger-pad overlay before trusting wrist-based detections.

## 16. Pick and place by command

Verified on the real arm on 2026-10-05 with a mug (several picks, both place forms). Type or say:

| Command | What happens |
|---|---|
| `pick up the cup` | hover over it, wrist camera centres the fingers on it (turning the wrist 90 deg if a handle would meet a finger), straight descent, close, lift. Refuses to descend if the wrist cannot see the object. |
| `pick up the cup and put it down` | the same, then set it back where it was picked up (place `here`). |
| `pick up the cup and put it next to the mouse` | the planner finds a free table spot 13-20 cm from the mouse, at least 11 cm from everything else, then carries (object ~12 cm above the table), lowers, opens, backs straight up. |
| `pick up the cup and put it aside` | a free spot 15-23 cm from where it stood (the model's own coordinates are overridden). |
| `pick up the cup and put it down at x 0.3 y 0.1` | explicit spot (moved to the nearest free one if it is taken). |

How it is kept safe:

- After a pick the planner removes anything the command did not ask for (the 7B model likes to add `go_home`, `open_gripper`,
  or a `move_to` over another object, any of which would drop or drag the object). A `move_to` before a pick is removed too.
- `place` needs the fingers partly closed (holding something), checks that every move ends where it should, and stops
  with the object still held if one does not. It refuses if a tall object (bottle, vase) is within 12 cm of the carry path.
- The object is carried and set down relative to the height it was grasped at, so it lands on the table, not above it.

Limits: the wrist must see the object (top-down view); only a mug has been tried; `place` assumes the object stays in the
fingers (no slip detection beyond the finger position); the cup is not in the MoveIt scene while carried.

Verified by voice on 2026-10-05: "pick up the mouse and put it aside" and "pick up the cup and put it aside" (both succeeded after
fixing four things the first attempt exposed: the low-object clearance reverting to 5 cm after a restart, a scene height of
0.132 for a cup (0.05 real), the model picking a phantom mouse below the table, and a stray `go_home`). The grip check stopped
the two misses safely. Free-spot search treats every real object on the table as an obstacle, including ones beyond the
picking limit (a bottle at the table edge).

### Trial series, 2026-10-05 (voice commands, mug and mouse, "put it aside")

First series: 0 of 2 completed (the mouse slipped on the lift; then a phantom "bottle" that was the gripper's own finger and a mislabelled
duplicate of the mouse made the neighbour check refuse). After the fixes: 4 of 4 (mouse, cup, mouse, cup), no retries, no drops.
What made the difference: the scene graph drops detections on the robot's own body and mislabelled duplicates within 6 cm; the wrist
detector uses the segmentation model so the object's axis angle is measured and the wrist is turned to put the short side between the
fingers; the pick lifts 3 cm and checks the grip before the full lift, retries once after a miss, and checks that the open fingers will
not sweep a neighbour. The place does the same sweep check for the release.
Tunable at run time (no restart): `ros2 param set /arm_controller sweep_same_object_m|sweep_half_span_m|sweep_margin_m|carry_avoid_m <value>`.

## 17. Trajectory smoothing (2026-10-07)

Every verified path is re-timed in `trajectory_smoothing.py` along a minimum-jerk curve before it is executed
(`_smoothed` in `arm_controller_node.py`, the single place that calls `moveit2.execute`). The path is the one the
safety check approved; only the timing changes, so velocity and acceleration are zero at both ends and nothing steps.
Pilz PTP's trapezoid had acceleration jumping to ~7 rad/s^2 at each move boundary.

Measured on the arm (3 high moves, joint states at 100 Hz): peak jerk 40.5 -> 7.4 rad/s^3, peak acceleration
7.06 -> 1.36 rad/s^2. Runtime parameters (no restart): `smooth_trajectories` (false = planner timing, for A/B),
`smooth_vel_scale` 0.30 and `smooth_acc_scale` 0.20 (fractions of the joint limits), `smooth_min_duration_s` 0.6.
`/tmp/jerkprobe.py`-style probes: record /joint_states while a plan runs and Savitzky-Golay the derivatives.

The controller process runs on the system Python, whose user-site scipy cannot import `scipy.interpolate`, so the
spline is numpy-only. If smoothing ever fails the move runs with the planner's timing and logs a warning.
