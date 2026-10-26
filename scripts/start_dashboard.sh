#!/bin/bash
# Start the dashboard (backend on :8000 and the frontend dev server on :5173). Touches NO hardware: it does not start the robot driver, the cameras or any ROS node.
#
#     bash /mnt/ros_workspace/Shashproject/scripts/start_dashboard.sh
#
# Use it after a logout of the desktop session: everything started from that session (this dashboard too, but also robot bringup and the cameras, which is what releases the arm for the
# other accounts on this machine) is stopped when the session ends. Then press "Robot bringup" / "Cameras bringup" in the dashboard (or the health banner) as needed.
REPO=/mnt/ros_workspace/Shashproject
bash "$REPO/scripts/restart_dashboard_backend.sh" || exit 1
if curl -sf -m 3 http://localhost:5173/ >/dev/null; then
  echo "frontend already up on :5173"
else
  cd "$REPO/dashboard/frontend" || exit 1
  setsid nohup npx vite --host 0.0.0.0 > /tmp/dashboard_frontend.log 2>&1 < /dev/null &
  for i in $(seq 1 20); do curl -sf -m 2 http://localhost:5173/ >/dev/null && { echo "frontend is up on :5173"; exit 0; }; sleep 1; done
  echo "frontend did not come up -- see /tmp/dashboard_frontend.log"; exit 1
fi
