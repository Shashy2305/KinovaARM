# Trajectory optimisation: what was measured, what cuRobo does and does not buy us (2026-10-20)

Question: make our trajectories more optimal, with NVIDIA cuRobo (GPU motion generation) as the suggested tool. I installed it in an isolated environment, measured it against the
current pipeline on identical inputs, and found a bigger and safer win that needs no new planner. Everything below is reproducible from `docs/spikes/curobo/`.

## 1. Where the time goes today
A pick takes about 15-27 s and the log shows where: hover transit 6.0 s, wrist turn 5.3 s, centring 4.4 s, descend 2.4 s (OVERNIGHT_2026-10-09.md). Those are MOTION times at deliberately low speed caps
(`smooth_vel_scale` 0.30, `smooth_acc_scale` 0.20), not planning times.
* **Planning is already fast**: MoveIt IK answers in a median 1 ms (p99 9 ms; `scripts/ik_baseline.py`, `docs/data/ik_baseline_moveit_2026-10-20.json`), 99% of 160 pick-style targets solved within the controller's two tries.
* **What is not good is the SIZE of the moves**: the best-cost IK solution differs from the current pose by a median of 2.7 rad in one joint (p90 5.0), which is why swings are long and why the controller sometimes refuses with
  "needs a 5.1 rad reconfiguration". The controller already tries several solutions and keeps the nearest, so this is mitigated, not solved.

## 2. cuRobo (v2, Apache-2.0, built on PyTorch + NVIDIA Warp), installed in `/mnt/ros_workspace/curobo_spike` (conda CUDA 12.4 + torch 2.4.1; the production venv is untouched)
Built a robot model for the Gen3 + Robotiq 2F-140 from the live URDF with cuRobo's own builder: 117 collision spheres on 19 links (`gen3_robotiq_flange.yml`; tool frame `end_effector_link`, gripper joint locked).
| test (RTX 4080) | result |
|---|---|
| batched IK, 160 pick-style poses | 100% solved, **7-10 ms for the whole batch** (0.05 ms per pose) |
| IK solution quality | best-cost solution moves a joint 4.7 rad (median; it uses +-2pi ranges for the continuous joints); picking the nearest of 16 returned seeds: 2.5 rad median, 3.9 max (MoveIt: 2.7 median, 5.0 p90) |
| collision-aware planning home -> 24 hover poses (table + rear wall) | 24/24, **72 ms median**, 85 ms p90 |
| trajectory duration at its default (full-speed) limits | 3.0 s median (peak 1.1 rad/s): not comparable, our caps are 30-45% of the limits |
| trajectory duration under OUR caps (speed 45%, acceleration 20% x 8.6, jerk 8 rad/s^3) | **11.5 s median** (it could not produce a trajectory at 30% at all: its fixed optimisation horizon is too short for slow limits) |
| our minimum-jerk retimer, same caps, same endpoints | 7.35 s median |
So cuRobo is fast to PLAN and good at collision avoidance, but its trajectories are LONGER than what we already execute when the same conservative limits are imposed.

## 3. The change that does pay: a jerk-limited S-curve instead of minimum jerk (implemented, default OFF)
A minimum-jerk curve has its peak speed at the middle only (1.875 x the mean speed), so with a speed cap of v the move takes 1.875 x longer than a profile that REACHES v and holds it.
The same joint path (already verified by the forward-kinematics check) can instead be timed with a rest-to-rest **S-curve**: acceleration ramps up with a jerk limit, holds the acceleration cap, ramps down, cruises at the speed cap, mirrors for the stop.
Same speed cap, same acceleration cap, an explicit jerk cap (8 rad/s^3 = a 0.2 s ramp, far gentler than the trapezoid Pilz produces, which has infinite jerk at the corners).
* Measured on the 24 moves above (same caps for all three): **minimum-jerk 7.35 s, S-curve 4.45 s median: 39% shorter**; cuRobo 11.5 s.
* Where: `trajectory_smoothing.scurve_profile()` and `smooth(profile='s_curve', jerk_max=...)`; parameters `smooth_profile` (`min_jerk` default, `s_curve`) and `smooth_jerk_limit` (8.0) on `/arm_controller` and in `config/arm_controller_params.yaml`.
  It applies to every path the retimer gets: the long PTP swings arrive as dozens of collinear waypoints (the smoother calls that a 'spline'), and curved paths use bounds that add the curvature terms; if those bounds are infeasible it falls back to minimum jerk.
  The path itself, the safety check and the limits are untouched, only the time law changes. (The first version only handled paths labelled 'linear' and would have silently done nothing on real PTP moves; a controller-level test caught it.)
* Tests (8 new in `test_trajectory_smoothing.py`, 1 in `test_grasp_integration.py`): every joint's speed, acceleration and jerk stay inside the limits on 20 random moves (numerical check), rest at both ends, exact end positions, at least 25% shorter than minimum jerk,
  short moves that cannot reach the cap, the 0.6 s floor, spline paths fall back, default unchanged.
* **Not yet run on the arm.** Live test order (E-stop in hand, `go home` first): (1) `ros2 param set /arm_controller smooth_profile s_curve` with the default caps (speed 0.30): watch the first swings, expect about 35-40% shorter hover transits and a clearly smooth start/stop;
  (2) compare mean pick seconds with `scripts/outcome_report.py` (baseline 26.8 s at 0.30, 15.8 s at 0.45 with minimum jerk); (3) only then think about raising `smooth_free_vel_scale`; (4) if any move feels harsh, raise nothing, lower `smooth_jerk_limit` to 4.

## 4. Where cuRobo WOULD help (not needed for the above)
* **Collision-aware motion around real obstacles from the cameras** (it builds a signed-distance field from depth images on the GPU): our safety checks today are table + joint limits + a neighbour sweep test; the planner does not know about the other objects.
* **Choosing among many grasps / approach angles in one call** (goal sets, batched IK in milliseconds), e.g. ranking the closing directions and the lean azimuths by reachability instead of trying them one by one.
* **Reactive control (MPC)** for tracking a moving object or a hand, if that is ever wanted.
* Integration cost: a separate service process (its own CUDA 12.4 environment), a ROS 2 wrapper, the robot model kept in sync with the URDF, GPU memory (about 1-2 GB next to Ollama's 5 GB and the detectors), and a staged live test like the others. Its output would still go through our forward-kinematics path check and the retimer.
  Recommendation: do the S-curve first (hours, no new dependency), measure the picks, and only then decide if obstacle-aware planning is worth a second control stack.

## 5. Reproduce
`bash docs/spikes/curobo/setup.sh` (builds the environment; about 30 min), then `python cu_bench.py`, `cu_plan.py`, `cu_compare.py` inside it; `python3 scripts/ik_baseline.py` needs the ROS stack (read-only).
