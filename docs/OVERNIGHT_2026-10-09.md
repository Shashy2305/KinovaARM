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
