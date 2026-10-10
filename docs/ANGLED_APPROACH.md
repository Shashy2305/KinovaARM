# Angled (tilted-tool) approach: findings and integration design (2026-10-09)

Status: **geometry module, read-only feasibility study (endpoints AND the straight line) and the pick-sequence wiring are done, all parameters default OFF; the sequence is verified in the flow harness (fake hardware) but has NEVER moved the arm.** The arm has only ever come
straight down. This document says what a tilt can and cannot buy, what the arm can actually reach, and the exact way to
integrate it so that it can be tested step by step with the arm live and someone watching.

## 1. What a tilt buys (and what it does not)

The fingers close along a horizontal axis `c`; the tool axis `a` points from the flange to the fingertips.

| Problem | Does tilting help? |
|---|---|
| A neighbour lies ALONG the closing axis (the mouse 1 cm from the bowl rim) | **No.** The fingers still straddle the object along `c`; a tilt about `c` does not change the span, and a tilt about the other axis raises one tip by only `(gap/2)*sin(tilt)` (about 3 cm at 30 deg). The fix for that case is the shape-based direction choice (grasp_planner.py) or the preshape, not a tilt. |
| A neighbour lies PERPENDICULAR to `c` (beside the gripper body) | **Yes.** The flange sits `0.215*sin(tilt)` behind the fingertips (10.8 cm at 30 deg), so leaning away from the neighbour moves the wrist and finger bases out of its way. |
| Tall object (bottle) in a crowd, or object near the table edge / a wall | **Yes** (comes in from the open side). |
| Slippery smooth object | Maybe: a tilt changes where the pads touch, but this is not the failure we measured. |
| Wrist camera view | **Worse**: the camera tilts with the tool and no longer looks straight down (the centring maths is general, but accuracy drops with tilt). |

## 2. What the arm can reach: `scripts/angled_approach_study.py`

Read-only (MoveIt `/compute_ik` + `/compute_fk`; the arm never moves). 20 fingertip targets on the table
(x 0.25-0.55, y -0.30..0.30), 4 wrist yaws tried per target, a target counts when BOTH the pregrasp (0.12 m back along the tool
axis) and the grasp have an IK solution inside the planner's joint limits, an acceptable swing from the seed pose, and FK
shows both finger pads >= 5 cm above the table. Data: `docs/data/angled_study_2026-10-09_*.log`.

Fingertips 3 cm above the table (a flat object such as the mouse):

| tilt | az 0 (leads toward +x) | az 90 (+y) | az 180 (-x) | az 270 (-y) |
|---|---|---|---|---|
| 0 deg | 20/20 | | | |
| 15 deg | 18/20 | 20/20 | 18/20 | 20/20 |
| 30 deg | 15/20 | 19/20 | 10/20 | 13/20 |
| 45 deg | 4/20 | 0/20 | 0/20 | 0/20 |

Fingertips 10 cm above the table (side grasp of a bottle), tilts 0/15/30: 30 deg reaches 18/20 for az 90 and 270, 9/20 for az 180.

Reading: **tilts up to 15 deg are safe almost everywhere; 30 deg works well when leaning toward +-y (sideways); 45 deg is out.**
Leaning toward -x (the fingertips leading toward the base) is the hardest. The main failure reasons at 30 deg are `no_ik`
(the wrist runs out of range) and `low_pad` (one pad would dip under the 5 cm table margin when the fingers close along the lean).

## 3. Geometry (`thesis_robot/angled_approach.py`, 56 tests)

`tool_axis(tilt, az)`, `tool_quat(base_quat, tilt, az, yaw)`, `tilted_from(current_quat, tilt, az)` (leans the CURRENT orientation,
which already carries the wrist alignment), `approach_poses(tip, tilt, az, standoff)` -> (pregrasp flange, grasp flange, axis),
`flange_for_tip`, `lowest_tool_z`. The flange of a vertical grasp sits `tip_z + 0.215`; a tilted one at `tip_z + 0.215*cos(tilt)`,
so the controller's vertical "flange floor" rule (table + 0.215 + 0.05) would wrongly refuse a tilted grasp: the path check's
PAD rule (pads >= 5 cm above the table) is the one that applies, and `_move_to(..., min_flange_z=...)` must be lowered for it.

### Cartesian-line check (added later the same night)
The study now also asks MoveIt's `/compute_cartesian_path` for the straight line from the pregrasp to the grasp (>= 98% followable). Result for fingertips
3 cm above the table, tilts 0/10/20/30: **line failures 0 in every cell**; whenever both ends have an IK solution the straight approach exists.
Reachable targets (of 20): 10 deg: 20/20 (az 0, 90, 270), 18/20 (az 180); 20 deg: 17 (az 0), 19 (az 90), 15 (az 180), 19 (az 270); 30 deg: 15, 19, 10, 12.
Log: `docs/data/angled_study_2026-10-09_tip3cm_with_line_check.log`. **Recommended first live values: 10 deg, then 20 deg, leaning toward +-y.**

## 4. Integration (done, parameters OFF) - the sequence the code runs (`_pick_once`, `_tilt_for`, `_tilted_pose`)

Parameters (all default OFF): `approach_tilt_deg` (0), `retry_tilt_deg` (0, retries only), `tilt_azimuth_deg` (-1 = automatic). Capped at 30 deg; never for tall objects;
falls back to the straight-down descent whenever IK fails or the pregrasp/grasp flange would leave the planner's x/y limits (nothing has moved yet at that point).

Pick sequence with a tilt (replaces `descend` and the three lifts, everything before `centre` is unchanged):
1. open, raise, hover, align, centre, fingers_clear: as today (vertical tool, over the object).
2. `tilt in`: T = fingertip target = (pad midpoint xy, grasp tip z). `a = tool_axis(tilt, az)`; pregrasp flange = `T - standoff*a - 0.215*a`.
   PTP to the pregrasp with `tilted_from(actual tool quat, tilt, az)` (use `_solve_goal_joints(..., allow_yaw=False)` then `_guarded_move(joint_positions=...)`).
3. `descend along the tool axis`: cartesian move to the grasp flange (orientation held by `_actual_tool_quat`), `min_flange_z` lowered.
4. close, `check the grip`.
5. `retreat along the tool axis`: cartesian back to the pregrasp, then `stand the tool upright`: PTP to the vertical hover pose over the
   object (the existing lift steps assume the flange is at the object's xy; with a tilted tool it is not, so they must not run before this).
6. `check it is still held`, then the existing carry/lift.
`az` 'auto': lean AWAY from the nearest neighbour whose bearing is within 90 deg of the approach, restricted to azimuths the study found
reachable for that tilt (+-y first, then +x; avoid -x).

Risks to watch while testing: wrist camera accuracy after the lean (centre BEFORE leaning, as above); IK flips on the lean (the
study used a seed near the current pose, so keep the seed = the current joints); the held-object volume filter in the scene graph assumes
a vertical tool; the finger-sweep check assumes the fingers span +-9.5 cm horizontally (true for a lean about the closing axis only).

## 5. Test order (details in TEST_PLAN_2026-10-09.md, stage 8)

Tilt 10 deg on the retry only, flat object (mouse), az +-y, in the open -> 15 deg -> 25 deg -> side grasp of the bottle at 30 deg.
Stop at the first unexpected motion; the rollback is `ros2 param set /arm_controller approach_tilt_deg 0.0` (and `retry_tilt_deg 0.0`).

## 6. First live results and the wrist flip (2026-10-09 evening)
Live on the real arm (E-stop in hand, defaults otherwise): 10 deg on the mouse 4/4 held (3 x azimuth 270, 1 x azimuth 90), 15 deg on the cup 1/1 (azimuth 0); 15 deg on the mouse is refused by the pad-height rule (lower pad 1 mm under table + 5 cm);
20 deg found no tilted approach twice (`no collision-free IK solution`, then `needs a 5.1 rad single-joint reconfiguration`) and fell back to straight down. Details in TEST_PLAN_2026-10-09.md.
**Wrist flip (code, not yet run live):** the live tilt solve is locked to the one wrist yaw chosen for the grip, while the study tried four. When it fails, `descend` now retries once with the goal orientation turned 180 deg about the tool axis (same grip because the gripper is
symmetric, same lean direction, other joint configuration). It is skipped if a neighbour whose top reaches the pregrasp fingertip height stands within the swing circle of the open fingers (`_flip_swing_blocker`: half-span + its radius + margin), because the
wrist turn sweeps the fingers around the object. Parameter `tilt_try_wrist_flip` (default true; it only acts when the plain lean has no solution, so nothing changes for leans that already work). The log says `... - with the wrist turned 180 deg` and the outcome log's
`tilt` event has `flipped: true`. 5 flow tests. Unknown until tried: the later `stand the tool upright` PTP after a flipped lean (its yaw candidates are +-90 deg around the vertical grasp orientation, so it may turn the wrist back by up to 90 deg while holding the object).
