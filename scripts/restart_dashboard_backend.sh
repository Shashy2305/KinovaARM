#!/bin/bash
# Restart ONLY the dashboard backend (needed after editing anything under
# dashboard/backend/). The ROS nodes it started keep running (they live in
# their own sessions) and show up as "running_external" afterwards.
#
#     bash /mnt/ros_workspace/Shashproject/scripts/restart_dashboard_backend.sh
#
# Why a script: uvicorn can ignore SIGTERM while a ROS executor thread is
# alive, and `pkill -f uvicorn` would also match the shell running it.
REPO=/mnt/ros_workspace/Shashproject
PID=$(ps -eo pid,args | grep -E '^ *[0-9]+ python3 -m uvicorn app.main:app' | grep -v grep | awk '{print $1}' | head -1)
if [ -n "$PID" ]; then
  echo "stopping backend pid $PID"
  kill "$PID"
  for i in 1 2 3 4 5 6 7 8; do sleep 1; kill -0 "$PID" 2>/dev/null || break; done
  if kill -0 "$PID" 2>/dev/null; then echo "still alive after 8 s -> kill -9 (only the backend)"; kill -9 "$PID"; sleep 1; fi
fi
cd "$REPO/dashboard/backend" || exit 1
export SHASHPROJECT_REPO_ROOT=$REPO SHASHPROJECT_WORKSPACE_ROOT=/mnt/ros_workspace/ros2_kortex_ws
export SHASHPROJECT_VENV_ACTIVATE=/mnt/ros_workspace/venv/bin/activate
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source /mnt/ros_workspace/ros2_kortex_ws/install/setup.bash >/dev/null 2>&1
source "$REPO/install/setup.bash" >/dev/null 2>&1
source /mnt/ros_workspace/venv/bin/activate
export ROS_DOMAIN_ID=42
setsid nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/dashboard_backend.log 2>&1 < /dev/null &
for i in $(seq 1 20); do curl -sf -m 2 http://localhost:8000/api/health >/dev/null && { echo "backend is up"; exit 0; }; sleep 1; done
echo "backend did not come up -- see /tmp/dashboard_backend.log"; exit 1
