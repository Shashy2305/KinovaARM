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
