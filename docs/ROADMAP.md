# Roadmap: what to do next, in order of value (written 2026-10-09)

Effort: S = an hour or two, M = a day, L = several days. "Needs" = what must exist first.

| # | Item | Why it matters | Effort | Needs |
|---|---|---|---|---|
| 1 | **Run the staged tests** (docs/TEST_PLAN_2026-10-09.md), starting with `scripts/preflight.py --tests --bench` | Nothing from the overnight work has touched the arm. The live baseline (stage 4) is also the reference for every later claim. | M | arm live, someone at the E-stop |
| 2 | **Fine-tune YOLOv8-seg on the self-labelled wrist frames** (`scripts/export_training_dataset.py`) | Fixes the top-down mislabelling (bottle -> "sports ball", mouse -> "cup") at its source and gives the bottle a mask (so shape analysis, width and the grasp-angle planner work for it). | M | ~30 kept samples per class (they accumulate during stage 4); review the `auto-mask` ones |
| 3 | **Angled approach on the real arm** (stage 9; wired, default off; 10 deg then 20 deg toward +-y) | The one real change in HOW the arm approaches. Biggest unknown, biggest upside for crowded tables and for a side grasp of the bottle. | M | stages 4-5 pass |
| 4 | **Bigger planner model benchmark**: `ollama pull` a 14B model (Qwen3-14B class fits 16 GB at 4-bit), `ros2 param set /llm_planner model <name>`, rerun `scripts/planner_benchmark.py` | The benchmark and its 14 command forms already exist, so comparing models is one command each. Qwen2.5-7B is 100% on it with the verifier; a larger model should need fewer retries and handle novel phrasing. | S | pull time; GPU memory (the 7B uses 5 GB, other users hold 5 GB) |
| 5 | **Third static camera** (the spare D435, serial 207522071578, at the old OAK-D mount): global_camera_3 driver, TF broadcaster, detector, watchdog entry, scene-graph subscription, arm-click calibration | Better coverage of the table; a third vote for fusion and for the obstacle guard's `confirmed`. | M | a USB-C 3.0 cable |
| 6 | **Safety: stop when a person reaches in.** The obstacle guard already sees anything above the table; a confirmed obstacle within ~25 cm of the arm's links while live could pause motion (`use_unknown_obstacles` only changes planning today) | Hands were on the live arm twice during testing. A real safety layer, not a convenience. | M | stage 8 shows the guard has no confirmed phantoms |
| 7 | **Learning loop, offline**: nightly `scripts/outcome_report.py --csv`, retune per-object `GRIP_WIDTH_M`, grasp heights and `grasp_policy` priors from the log; LoRA-tune the planner on `~/.ros/planner_memory` successes only when several hundred exist | Honest "learns from its mistakes": measured outcomes bias which safe grasp is tried. Do NOT do reinforcement learning on the live arm. | M-L | 100+ live picks |
| 8 | **Storage hardening**: copy `/ros_workspace.img` to the internal disk (or make `/mnt/ros_workspace` a plain folder) so a USB reset cannot take the workspace again | A single detached external disk cost hours on 2026-10-08. | S | free internal space (119 GB now) |
| 9 | **Per-joint jog in the dashboard** (hold-to-jog, ~2 deg steps, live only, joint limits, FK table/wall check per step, one big Stop) | Recovering the arm after a fault or a hand-guided move without the Kinova web app. | M | |
| 10 | **Voice by microphone** | Hardware fault (mic); the voice path itself is built. | S | a working microphone |
| 11 | **Move merging** | Not worth it: the profile shows time is spent in the hover transit (6.0 s), the wrist turn (5.3 s), centring (4.4 s); `smooth_free_vel_scale` addresses the first. | - | |

## Experiments worth running for the thesis (the outcome log records everything needed)
1. Success rate vs nearest-neighbour distance (crowding), per object: `outcome_report.py --csv`, plot success against `nearest_m`.
2. Retry-from-a-new-angle vs repeat: force first-attempt failures (stage 5) and compare second-attempt success with `grasp_retry_new_angle` on/off.
3. Shape-ranked angle on every attempt vs the short-side rule (stage 6), by object.
4. Planner: baseline vs verification vs memory (`planner_benchmark.py`: already measured 84% -> 100%), then model size.
5. Detector before/after fine-tuning: label accuracy on top-down wrist frames of the six objects.
6. Angled approach: success vs tilt (0/10/20 deg) on the mouse and cup, and whether it rescues picks refused for a neighbour.
Keep one table of {condition, n, successes, notes}; at tens of trials per cell only large effects are visible, so say so.
