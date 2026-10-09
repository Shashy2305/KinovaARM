# Overnight work log, 2026-10-09 (arm stays in Dry Run; no arm motion)

Goal: planner memory from the outcome log, OpenCV shape analysis (real object widths, unknown-obstacle guard),
a grasp-angle planner (retry from a NEW angle), an angled-approach prototype (feasibility by MoveIt planning only),
move-merging design, and a staged test plan. Everything that changes arm motion is behind a parameter, default OFF.

Log (newest last):
- 00:15 arm switched to Dry Run, work started.
- 01:25 shape_analysis.py + grasp_planner.py done, tested (synthetic + real wrist frames), wrist_detection publishes det['shape'] (advisory). Pushed.
- 01:50 controller integration (retry-new-angle ON for retries, grasp_candidates/use_shape_width OFF), 143 tests pass, pushed; test env: source /opt/ros/humble/setup.bash THEN venv/bin/activate
- 02:35 Planner: command_grammar.py (parse + verify + repair), planner_memory.py (template examples + warnings, results learned from /arm_status), planner changes (verification with corrective hint, grammar repair after 3 wrong answers, memory block), scripts/planner_benchmark.py.
  Benchmark (Dry Run, real planner): baseline 27/32 (84%): "go home" answered with a move_to, "put it down" empty plan, "move X next to Y" with two move_to steps.
  Verification + hints: 32/32. + memory: 50/50 on an extended 10-form suite (compound and polite forms included), mostly 1.0-1.4 attempts.
  Params (runtime): plan_verification, grammar_repair, planner_memory on /llm_planner (all default True).
- 03:20 Obstacle guard (obstacle_guard.py + obstacle_guard_node.py, /unknown_obstacles): depth -> base points (flying pixels cleaned, robot body removed) -> 1 cm height grid -> connected components -> minus what the scene explains; 'confirmed' = seen by 2 cameras. Controller consumes it behind use_unknown_obstacles (default OFF) and always logs nearby ones (unknown_near) in the outcome log. Dashboard process 'obstacle_guard' (in FULL_BRINGUP_ORDER).
  Incidents while testing (all fixed or documented): my own stray `du` (from the Elements-drive check, 5.5 h at 98% CPU) and a first obstacle node at 741% CPU (BLAS/OpenCV threads, np.maximum.at) pushed load to 15 and made the watchdog flag BOTH RealSense cameras as 'reconnected' three times (also reproducible by restarting the dashboard backend: kill -9 leaves a dead reader on the reliable image topics). Fixes: node now 13% CPU (1 thread, stride 4, 1 Hz, sort-based max), watchdog STALE_SEC 6 -> 15 s, flags cleared after extrinsics check (cameras unmoved: RS2 plane tilt 2.3 deg / z -0.020 as before).
  robot_bringup's ros2_control_node had frozen (no /joint_states publisher, controller_manager unresponsive) ~00:08; restarted robot_bringup (no motion), controllers active, 923 Hz joint_states. Scene graph + RS detectors restarted (scene was empty after the stalls).
  Dashboard full bring-up now skips the OAK-D processes while config/disabled_cameras.txt lists oakd (object_detection alone was ~45% CPU for a disabled camera).
