# Side-pick reach window for the coffee cup (IK scan, 2026-11-03)

Successes out of 4, side grasp toward azimuth 90, cup at (x, y) in base_link, IK service only (no arm motion).
"now" = planner joint limit 2.88 rad (Pilz `joint_limits` in kortex_moveit_config, mirrored by `motion_utils.PLANNER_JOINT_LIMIT`); "3.5" = the same scan with the unwrap limit 3.5.

| x | limit | y -0.10 | -0.07 | -0.04 | -0.02 | 0.00 | +0.02 | +0.04 | +0.06 | +0.09 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 2.88 | 0 | 0 | 4 | 4 | 1 | 0 | 0 | 3 | 2 |
| 0.40 | 3.5 | 0 | 0 | 4 | 4 | 3 | 4 | 4 | 3 | 4 |
| 0.45 | 2.88 | 0 | 0 | 0 | 3 | 4 | 0 | 1 | 3 | 0 |
| 0.45 | 3.5 | 0 | 0 | 0 | 3 | 4 | 4 | 4 | 3 | 4 |
| 0.50 | 2.88 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| 0.50 | 3.5 | 0 | 0 | 0 | 0 | 0 | 3 | 4 | 4 | 2 |

Why: the failing solutions are rejected at `unwrap_to_seed` (a continuous joint, mostly joint 3, would have to sit between about 165 and 195 deg, outside +-2.88 rad). The real joints 1/3/5/7 are continuous; the limit is a planner setting.
Live evidence: a cup at (0.458, 0.024) was refused three times in a row (three fresh home -> hover tries), no motion; a cup at (0.45, 0.0) works.

## After widening joints 3 and 7 to 3.5 rad (motion_utils.CONTINUOUS_JOINT_LIMITS)

Successes out of 4, az 90, the code's own per-joint limits:

| x | y -0.04 | -0.02 | 0.00 | +0.02 | +0.04 | +0.06 | +0.08 | +0.10 |
|---|---|---|---|---|---|---|---|---|
| 0.38 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 4 |
| 0.41 | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 4 |
| 0.44 | 0 | 4 | 2 | 4 | 4 | 3 | 4 | 4 |
| 0.47 | 0 | 0 | 3 | 4 | 4 | 3 | 2 | 2 |
| 0.50 | 0 | 0 | 0 | 3 | 3 | 4 | 4 | 3 |

Coffee settings from this: serve_xy (0.41, 0.03), window x +-3 cm, y +-5 cm. The pilz yaml was NOT changed: a plan-only probe showed Pilz PTP already plans joint 3/5/7 goals up to 4.4 rad (error code 1);
the 2.88 was only our own constant. The Kortex driver wraps feedback to +-pi and commands modulo 360 deg; the installed joint_trajectory_controller (2.53) has angle_wraparound.

## Arm-link clearance (the forearm), 2026-11-03 afternoon

The pad-only scans above were optimistic: the safety check also keeps the forearm / wrist links 8 cm (7.5 cm) above the table, and the IK picks a branch at random; three live picks at x 0.39-0.46 were refused at execution by 1 mm (forearm 0.067 vs 0.068).
Median margin (mm over the limit) along the chain, one IK ask per pose, was erratic (+28 at (0.43, 0.00) ... -93 at (0.46, 0.00)).
`ArmControllerNode._chain_links_clear` now checks the links on every candidate pose (and the joint-space midpoint) while choosing the approach, asks the IK up to 4 times per pose, and needs 3 mm of margin. Offline result, successes of 3:

| x | y -0.04 | 0.00 | +0.03 | +0.06 | +0.09 |
|---|---|---|---|---|---|
| 0.40 | 3 | 0 | 3 | 3 | 3 |
| 0.43 | 0 | 2 | 3 | 2 | 3 |
| 0.46 | 0 | 3 | 3 | 2 | 3 |
| 0.49 | 0 | 0 | 3 | 3 | 2 |

Coffee settings from this: serve_xy (0.44, 0.03), window x +-3 cm, y +-3 cm.
