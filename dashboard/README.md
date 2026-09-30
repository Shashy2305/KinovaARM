# KinovaARM Dashboard

A web dashboard for demoing the pipeline instead of juggling a dozen terminal
windows: live 3-camera view, one-click calibration (wrist-anchored, same
engine as `calibration/multi_camera_calibrate.py`), click-to-define workspace
boundary, node start/stop with a duplicate-process guard, a live top-down
scene plot, and a text command console wired to the LLM planner.

Network-accessible by design — reachable from off the lab PC, not just
`localhost`. **No auth.** Don't expose this past the lab network.

## Run it

**Backend** (FastAPI + a single shared rclpy context):

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/Shashproject/install/setup.bash
export ROS_DOMAIN_ID=42
cd ~/Shashproject/dashboard/backend
pip install -r requirements.txt   # first time only
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend** (React + Vite):

```bash
cd ~/Shashproject/dashboard/frontend
npm install   # first time only
npm run dev
```

Vite prints the URLs to open — a `Network:` one (not `Local:`) is what a
remote machine needs. The dev server proxies `/api/*` to the backend
(`vite.config.ts`), so the frontend never hardcodes a backend host — that
matters here since this is meant to be opened from machines other than the
one it's running on.

## What's real vs. what's a thin wrapper

- **Camera streams, node status, scene view, command console**: live data
  from the actual running pipeline, not mocked.
- **Calibration wizard**: runs the real `MultiCameraCalibrate` class from
  `calibration/multi_camera_calibrate.py` inside the backend's own rclpy
  context (refactored to expose `capture_camera()`/`board_visible()` as
  plain callable methods instead of only being driven by the CLI's
  `input()` loop — the CLI script itself still works unchanged).
- **Workspace boundary**: same deal with `DefineWorkspaceBoundary` from
  `calibration/define_workspace_boundary.py`.
- **Node Control**: every "node" is a real process (`ros2 run`/`ros2 launch`/
  a calibration script), started via `process_manager.py`.

## Design notes worth knowing before touching this

1. **One shared rclpy executor, not one thread per node.** Discovered during
   development: `rclpy.spin(node)` implicitly builds its own executor per
   call, and running several of those concurrently in separate threads
   against nodes in the same context crashes with `ValueError: generator
   already executing`. `ros_bridge.py` uses one `MultiThreadedExecutor`,
   with nodes added to it dynamically as they're created (the bridge at
   startup, calibration/boundary tools lazily on first use) — never spin a
   new node on its own thread here.

2. **The duplicate-process guard is the whole point of `process_manager.py`.**
   It checks `ps aux` for a process's known signature before starting it —
   whether the dashboard started the existing copy or a person did by hand
   in a terminal. This exists because of two real bugs hit during
   development: two OAK-D drivers fighting over one USB device, and an old
   RealSense TF publisher left running alongside a new one.

3. **`stop` kills by signature, not just by PID the dashboard itself
   started.** That's necessary for it to work against processes started
   outside the dashboard, but it means a `stop` call is not scoped to
   "things this dashboard is responsible for" — be careful testing against
   a live pipeline; this bit us once during development (killed the real
   OAK-D driver mid-test).

4. **The WS status feed and the REST `/api/nodes` endpoint return the exact
   same process-info shape** (`process_manager.all_status()`), on purpose —
   they used to differ (WS sent bare status strings), and every frontend
   component reading the WS feed silently rendered nothing as a result.
   Keep them sharing one function if this changes again.

5. **`arm_controller` going LIVE requires typing `MAKE IT LIVE` exactly**,
   not a toggle — enforced server-side (`routers/nodes.py`), not just in the
   UI, since this dashboard is reachable off the lab PC.

6. **Frontend pinned to Vite 6**, not the current default (Vite 8, which
   ships a Rolldown-based bundler requiring a native binding this machine's
   Node version — `v20.18.0`, below Rolldown's `^20.19.0` floor — can't
   load). If Node gets upgraded later, Vite could move back to current.
