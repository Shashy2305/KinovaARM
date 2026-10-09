# Overnight work log, 2026-10-09 (arm stays in Dry Run; no arm motion)

Goal: planner memory from the outcome log, OpenCV shape analysis (real widths, unknown-obstacle guard), a grasp-angle planner (retry from a
NEW angle), an angled-approach prototype (feasibility by MoveIt IK/FK only), move-merging analysis, and a staged test plan.
Everything that changes arm motion is behind a parameter; the defaults are listed in TEST_PLAN_2026-10-09.md. Times are commit times (PDT).

| Time | Commit | What |
|---|---|---|
| 00:03 | `2efb511` | (carried over from the evening) planner must return a pick/place; wrist centring tracks the locked-on object |
| 00:21 | `1ca8576` | `shape_analysis.py` (contour, minAreaRect, approxPolyDP, hull/defects/holes, ellipse, Hough, width + contact profiles), `grasp_planner.py`; validated on real wrist frames |
| 00:23 | `4ef3ee7` | controller: retry from a NEW closing angle (ON for retries), `grasp_candidates`, `use_shape_width`, 'record the grasp direction' step, shape events in the outcome log |
| 00:43 | `d33fa1e` | planner: `command_grammar.py` (verify + repair), `planner_memory.py`, `scripts/planner_benchmark.py` |
| 01:00 | `63c3f51` | unknown-obstacle guard from depth (`obstacle_guard*.py`), bring-up skips disabled cameras, camera watchdog stall limit 15 s |
| 01:06 | `e674ecd` | `grasp_policy.py` (learned from the outcome log), `angled_approach.py` + feasibility study, `smooth_free_vel_scale` |
| 01:07 | `bee545f` | TEST_PLAN, ANGLED_APPROACH, HANDOFF status, RUNBOOK section 19 |
| 01:08 | `80c597d` | `docs/FUSION.md` |
| 01:13 | `d0380b8` | spatial relations: left/right of, in front of, behind |
| 01:14 | `0c47fdf` | dashboard Robot chip turns red when nothing publishes /joint_states |
| 01:15 | `f461e49` | shape handle threshold |
| 01:16 | `2beb447` | flow harness: the REAL `_pick`/`_place` against a fake world (11 tests) |
| 01:21 | `b9b21d4` | planner 8192-token context, prompt-size log; health monitor script |
| 01:24 | `c043990` | self-labelled training data from successful picks + export to a YOLOv8-seg dataset |

## Measured
- Planner benchmark (Dry Run, real planner, 8 -> 14 command forms): baseline 27/32 (84%): "go home" -> a move_to, "put it down" -> empty plan,
  "move X next to Y" -> two move_to steps. With verification + retry hints 32/32; with memory 50/50; with spatial relations 54/56 (the 2 "failures" are correct
  refusals: no free spot to the right of an object at the workspace edge); latest 28/28.
- Angled approach feasibility (MoveIt IK/FK, 20 table targets per cell): tilt 15 deg reachable 18-20/20 everywhere, 30 deg 10-19/20 (best toward +-y), 45 deg 0-4/20.
- Pick timing from the outcome log (24 s average): hover transit 6.0 s, wrist turn 5.3 s, centring 4.4 s, descend 2.4 s, the 0.5 s grip checks are safety steps.
- Tests: 291 pass (ROS sourced BEFORE the venv). Flow harness covers retry-from-new-angle, narrowing, refusal, carry detour, outcome-log and training-sample requests.

## Incidents during the night (all resolved or documented)
- My own stray `du` (from the Elements-drive listing, 5.5 h at 98% CPU) and the first obstacle node (741% CPU: BLAS/OpenCV threads, np.maximum.at) pushed the load to 15.
  That froze `ros2_control_node` (no /joint_states publisher, controller_manager unresponsive; restarted robot_bringup, no motion) and made the camera watchdog
  flag BOTH RealSense cameras three times. Fixes: node now 13% CPU, watchdog limit 6 -> 15 s (restarting the dashboard backend also stalls the camera publishers briefly),
  dashboard Robot chip shows NO JOINT STATES, flags cleared after `check_extrinsics.py` showed the cameras had not moved.
- Never run `pkill -f` / `kill $(pgrep -f ...)` with a pattern that appears in your own command line (killed my shell twice).

## State left running
Dry Run. Stack up (robot bringup, cameras RS1/RS2/wrist, scene graph, planner with memory, obstacle guard, health monitor `~/.ros/health/health.csv`).
OAK-D processes stopped (the camera is disabled on purpose). Ollama warm for 6 h with an 8192 context.
