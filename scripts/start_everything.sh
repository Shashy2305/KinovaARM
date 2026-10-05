#!/bin/bash
# start_everything.sh — bring up the full KinovaARM pipeline on REAL-1 in one
# shot: dashboard backend + frontend, then robot bringup, then cameras, then
# the rest of the perception/planning/control stack. Safe to re-run — the
# dashboard's own process_manager refuses to double-start anything already
# running (that's the whole reason this goes through the dashboard's REST
# API rather than launching things directly).
#
# Usage:
#   bash start_everything.sh
#
# What it does NOT do: calibrate cameras, or put the arm in LIVE mode (both
# are deliberate manual steps — calibration needs a human watching the
# board, and LIVE mode needs the two-step confirm in the dashboard UI).
#
# Logs: ~/.ros/dashboard_logs/<process>.log for each ROS node,
#       /tmp/dashboard_backend.log and /tmp/dashboard_frontend.log for the
#       dashboard itself.

set -uo pipefail

REPO=/mnt/ros_workspace/Shashproject
WORKSPACE=/mnt/ros_workspace/ros2_kortex_ws
VENV=/mnt/ros_workspace/venv
API=http://localhost:8000/api

echo "=== 1/5: Checking robot connectivity ==="
if ! ping -c 1 -W 2 192.168.1.10 > /dev/null 2>&1; then
    echo "ERROR: robot (192.168.1.10) is not reachable. Check the Ethernet"
    echo "cable and that the robot is powered on before continuing."
    exit 1
fi
echo "Robot reachable."

echo
echo "=== 2/5: Dashboard backend ==="
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "Already running."
else
    (
        cd "$REPO/dashboard/backend"
        export SHASHPROJECT_REPO_ROOT="$REPO"
        export SHASHPROJECT_WORKSPACE_ROOT="$WORKSPACE"
        export SHASHPROJECT_VENV_ACTIVATE="$VENV/bin/activate"
        source /opt/ros/humble/setup.bash
        source "$WORKSPACE/install/setup.bash"
        source "$REPO/install/setup.bash"
        source "$VENV/bin/activate"
        export ROS_DOMAIN_ID=42
        nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
            > /tmp/dashboard_backend.log 2>&1 &
        disown
    )
    echo "Started. Waiting for it to come up..."
    for i in $(seq 1 15); do
        curl -sf "$API/status" > /dev/null 2>&1 && break
        sleep 1
    done
    curl -sf "$API/status" > /dev/null 2>&1 || {
        echo "ERROR: backend didn't come up — check /tmp/dashboard_backend.log"
        exit 1
    }
    echo "Backend is up."
fi

echo
echo "=== 3/5: Dashboard frontend ==="
if pgrep -f "vite --host" > /dev/null; then
    echo "Already running."
else
    (
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        cd "$REPO/dashboard/frontend"
        nohup npm run dev -- --host 0.0.0.0 > /tmp/dashboard_frontend.log 2>&1 &
        disown
    )
    echo "Started (http://$(hostname -I | awk '{print $1}'):5173)."
fi

echo
echo "=== 4/5: Robot bringup ==="
status=$(curl -s "$API/status" | python3 -c "import json,sys; print(json.load(sys.stdin)['processes']['robot_bringup']['status'])")
if [ "$status" = "running" ] || [ "$status" = "running_external" ]; then
    echo "Already running."
else
    curl -s -X POST "$API/nodes/robot_bringup/start"
    echo
    echo "Waiting for move_group readiness (up to 40s)..."
    for i in $(seq 1 40); do
        grep -q "You can start planning now" ~/.ros/dashboard_logs/robot_bringup.log 2>/dev/null && break
        sleep 1
    done
    echo "Robot bringup launched — verify controllers with:"
    echo "  curl -s $API/status | python3 -m json.tool | grep -A2 robot_bringup"
fi

echo
echo "=== 5/5: Cameras + full perception/planning/control bring-up ==="
status=$(curl -s "$API/status" | python3 -c "import json,sys; print(json.load(sys.stdin)['processes']['cameras_bringup']['status'])")
if [ "$status" = "running" ] || [ "$status" = "running_external" ]; then
    echo "Cameras already running."
else
    echo "Starting cameras (takes ~10-15s, has built-in settle waits)..."
    curl -s -X POST "$API/nodes/cameras_bringup/start"
    echo
fi

sleep 3
echo "Starting full perception/planning/control bring-up..."
curl -s -X POST "$API/bringup/full"
echo

echo
echo "=== Done ==="
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):5173"
echo "Check full status any time with:"
echo "  curl -s $API/status | python3 -m json.tool"
echo
echo "NOT started automatically (deliberate manual steps):"
echo "  - Camera calibration (Calibration tab in the dashboard)"
echo "  - Arm LIVE mode (dry-run is the default; use the dashboard's"
echo "    Go-LIVE button with its confirm phrase when you're ready)"
