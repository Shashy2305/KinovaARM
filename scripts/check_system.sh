#!/bin/bash
# check_system.sh -- one-shot health check of the whole KinovaARM stack.
# Read-only: it never starts, stops or moves anything.
#
#     bash /mnt/ros_workspace/Shashproject/scripts/check_system.sh
#
# Every FAIL prints the command that fixes it. Exit code = number of FAILs.

API=http://localhost:8000/api
ROBOT_IP=${KINOVA_ROBOT_IP:-192.168.1.10}
pass=0; fail=0; warn=0
ok()   { printf '  [ OK ] %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  [FAIL] %s\n         -> %s\n' "$1" "$2"; fail=$((fail+1)); }
wrn()  { printf '  [WARN] %s\n         -> %s\n' "$1" "$2"; warn=$((warn+1)); }
hdr()  { printf '\n%s\n' "$1"; }
tally() { # print a block of [ OK ]/[WARN]/[FAIL] lines and add them to the counters
  printf '%s\n' "$1"
  pass=$((pass + $(grep -c '\[ OK \]' <<<"$1"))); warn=$((warn + $(grep -c '\[WARN\]' <<<"$1"))); fail=$((fail + $(grep -c '\[FAIL\]' <<<"$1")))
}
pidcount() { ps -eo args | grep -E "$1" | grep -v grep | wc -l; }

hdr "1. Robot and network"
if ping -c 1 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then ok "robot $ROBOT_IP answers ping"
else bad "robot $ROBOT_IP does not answer ping" "check the Ethernet cable and that the arm is powered; E-stop released"; fi

hdr "2. Dashboard"
if curl -sf -m 4 $API/health >/dev/null; then ok "backend answers on :8000"
else bad "backend not answering" "see the 'Start the dashboard' section of docs/RUNBOOK.md; log: /tmp/dashboard_backend.log"; fi
n=$(pidcount '^python3 -m uvicorn app.main:app')
[ "$n" = "1" ] && ok "exactly one backend process" || bad "$n backend processes (need 1)" "ps -eo pid,args | grep '[u]vicorn app.main' ; kill the extra pid"
if curl -sf -m 3 -o /dev/null http://localhost:5173; then ok "frontend answers on :5173"; else wrn "frontend not answering on :5173" "see RUNBOOK 'Start the dashboard'"; fi

hdr "3. Nodes (from the dashboard)"
STATUS=$(curl -sf -m 6 $API/nodes || echo '{}')
out=$(python3 - "$STATUS" <<'PY'
import json, sys
try: d = json.loads(sys.argv[1])
except Exception: d = {}
essential = ['robot_bringup', 'scene_graph_node', 'llm_planner_node', 'arm_controller']
import os
off = set()
try:
    off = {l.strip() for l in open(os.environ.get('SHASHPROJECT_DISABLED_CAMERAS', '/mnt/ros_workspace/Shashproject/config/disabled_cameras.txt'))
           if l.strip() and not l.strip().startswith('#')}
except OSError:
    pass
for k, v in d.items():
    up = v['status'] in ('running', 'running_external')
    if not up and any(k.startswith(c) for c in off):
        print(f"  [ OK ] {k:28s} {v['status']} (camera disabled on purpose)")
        continue
    tag = ' OK ' if up else ('FAIL' if k in essential else 'WARN')
    print(f"  [{tag}] {k:28s} {v['status']}")
if not d: print('  [FAIL] could not read /api/nodes')
PY
); tally "$out"

hdr "4. Duplicate / orphaned processes (the cause of erratic planning)"
mg=$(pidcount 'moveit_ros_move_group/move_group')
[ "$mg" = "1" ] && ok "exactly one move_group" || bad "$mg move_group processes (need 1)" "ps -eo pid,ppid,lstart,args | grep '[m]ove_group/move_group'  then kill the older one(s)"
rsp=$(pidcount 'robot_state_publisher/robot_state_publisher')
[ "$rsp" = "1" ] && ok "exactly one robot_state_publisher" || bad "$rsp robot_state_publisher processes (need 1)" "kill the older one(s); duplicate /tf publishers fight"
rc=$(pidcount 'controller_manager/ros2_control_node')
[ "$rc" = "1" ] && ok "ros2_control_node alive" || bad "$rc ros2_control_node processes (need 1)" "0 = the hardware driver died: stop robot_bringup (dashboard), make sure E-stop is released, start it again"
orph=$(ps -eo ppid,args | awk '$1==1' | grep -E 'move_group/move_group|robot_state_publisher/robot_state_publisher|ros2_control_node' | grep -v grep | wc -l)
[ "$orph" = "0" ] && ok "no orphaned bringup processes" || wrn "$orph bringup process(es) with parent PID 1 (left over from an earlier stop)" "ps -eo pid,ppid,lstart,args | awk '\$2==1' | grep -E 'move_group|robot_state_publisher|ros2_control'"

hdr "5. Arm connection (joint states from the arm itself)"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source /mnt/ros_workspace/ros2_kortex_ws/install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=42
out=$(python3 -s - <<'PY' 2>/dev/null
import time, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init(); n = Node('check_system'); got = []
n.create_subscription(JointState, '/joint_states', lambda m: got.append(m), 10)
t = time.time()
while not got and time.time() - t < 6: rclpy.spin_once(n, timeout_sec=0.2)
if got:
    d = dict(zip(got[-1].name, got[-1].position))
    print('  [ OK ] /joint_states flowing:', {k: round(v, 2) for k, v in d.items() if k.startswith('joint')})
else:
    print('  [FAIL] no /joint_states in 6 s\n         -> arm driver is not publishing: see check 4 (ros2_control_node) and RUNBOOK troubleshooting "No joint states"')
rclpy.shutdown()
PY
); [ -n "$out" ] || out='  [FAIL] joint-state probe produced no output\n         -> is ROS sourced? try: source scripts/ros_env.sh'; tally "$out"

hdr "6. Safety state"
STAT=$(curl -sf -m 5 $API/status || echo '{}')
TBL=$(curl -sf -m 8 $API/table_geometry/status || echo '{}')
out=$(python3 - "$STAT" "$TBL" <<'PY'
import json, sys
try: s = json.loads(sys.argv[1]).get('ros', {})
except Exception: s = {}
try: t = json.loads(sys.argv[2])
except Exception: t = {}
arm = (s.get('arm_status') or 'unknown')
if 'LIVE' in arm.upper(): print(f"  [WARN] arm is LIVE ({arm})\n         -> it will move. Be at the E-stop. Dashboard 'Arm' cell shows a hazard stripe.")
else: print(f"  [ OK ] arm is not live ({arm})")
sv = t.get('saved')
if sv: print(f"  [ OK ] table geometry recorded: top z={sv['table_top_z']:.3f}, flange floor z={t.get('flange_floor_z', 0):.3f}")
else: print("  [FAIL] table geometry NOT recorded -> live motion is blocked.\n         -> dashboard > Table safety > record points > Save (RUNBOOK section 6)")
pose = t.get('pose')
print("  [ OK ] fingertip pose readable" if pose else f"  [WARN] fingertip pose unavailable: {t.get('pose_message','')[:90]}")
PY
); tally "$out"

hdr "7. Cameras"
out=$(python3 - "$STAT" <<'PY'
import json, sys
try: c = json.loads(sys.argv[1])['ros'].get('camera_calibration_status') or {}
except Exception: c = {}
if not c: print('  [WARN] no camera status (camera_watchdog not running?)')
for k, v in c.items():
    if v == 'disabled':
        print(f'  [ OK ] {k}: disabled on purpose (config/disabled_cameras.txt)')
        continue
    tag = ' OK ' if v == 'ok' else 'FAIL'
    hint = '' if v == 'ok' else ('\n         -> no frames: check that camera\'s driver / USB; wrist needs the kinova_vision driver from cameras_bringup' if v == 'no_signal' else '\n         -> flagged after a drop/bump: recalibrate in the Calibration tab')
    print(f'  [{tag}] {k}: {v}{hint}')
PY
); tally "$out"

hdr "8. Language model and voice"
if curl -sf -m 4 http://localhost:11434/api/tags | grep -q 'qwen2.5:7b'; then ok "Ollama up with qwen2.5:7b"
else bad "Ollama not answering or qwen2.5:7b missing" "systemctl status ollama ; ollama pull qwen2.5:7b"; fi
AUD=$(python3 - "$STAT" <<'PY'
import json, sys
try: print(json.loads(sys.argv[1])['ros'].get('audio_status') or 'none')
except Exception: print('none')
PY
)
case "$AUD" in none) wrn "voice node not publishing" "start 'Voice input' in Node Control";; *) ok "voice node status: $AUD";; esac

hdr "9. Disk / repo"
if mountpoint -q /mnt/ros_workspace 2>/dev/null; then
  free=$(df -BG --output=avail /mnt/ros_workspace | tail -1 | tr -dc 0-9)
  [ "${free:-0}" -ge 5 ] && ok "/mnt/ros_workspace mounted, ${free} GB free" || wrn "only ${free} GB free on /mnt/ros_workspace" "clean build/ and log/ directories"
else wrn "/mnt/ros_workspace is not a separate mount" "fine if the folder exists; if it should be the SanDisk image, mount it (RUNBOOK 'Storage')"; fi
chg=$(cd /mnt/ros_workspace/Shashproject && git status --short | wc -l)
[ "$chg" = "0" ] && ok "git working tree clean" || wrn "$chg uncommitted change(s)" "cd /mnt/ros_workspace/Shashproject && git status"

printf '\nSummary: %d ok, %d warnings, %d FAILED\n' $pass $warn $fail
exit $fail
