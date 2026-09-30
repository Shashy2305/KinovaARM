# robot_bringup — backup of untracked ros2_kortex packages

This is a backup, **not a working colcon package** — these files aren't meant to
be built directly from here. The actual robot bringup runs from
`~/workspace/ros2_kortex_ws/src/ros2_kortex/`, a clone of
[Kinovarobotics/ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex).
This directory mirrors the parts of that clone that were **never committed
anywhere** — found by running `git status` inside that clone and seeing them
come up untracked (`??`). Before this backup, if that one directory on that
one lab PC were ever lost, this content — this project's entire MoveIt config,
its gripper wiring, its calibration TF notes — would be gone with no history.

## What's here

| Path (relative to `ros2_kortex/`) | What it is |
|---|---|
| `kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_140_moveit_config/` | The MoveIt config package for this thesis's actual robot+gripper. Was entirely untracked. Includes the `robot.launch.py`/`gen3.srdf` fixes from 2026-09-29 (gripper made a proper launch argument defaulting to `robotiq_2f_140`; SRDF tip_link fixed from `pen_tip` back to `end_effector_link`) — see [README.md](../README.md#camera-calibration-tf) and git log for why. |
| `kortex_description/grippers/thesis_ee/` | A labmate's custom container-insertion end effector (their "UCR Surgical Arm" thesis), tested on this **same physical robot**. Not this project's gripper — backed up here only because it's referenced by the shared moveit_config package above and losing it would break their setup, not because it belongs to this project. |
| `kortex_bringup/` (partial) | A few untracked launch files and scripts (`insert_to_container.py`, `setup_planning_scene.py`, `InsertContainer.action`, some `.launch.py` files) plus the current content of a few files that ARE tracked upstream but were modified locally (`CMakeLists.txt`, `package.xml`, `gen3.xacro`, `gen3_macro.xacro`, `ros2_controllers.yaml`) — full current content, not diffs, for simplicity. |

## Restoring on a fresh machine

```bash
mkdir -p ~/workspace/ros2_kortex_ws/src
cd ~/workspace/ros2_kortex_ws/src
git clone https://github.com/Kinovarobotics/ros2_kortex.git
# ... clone whatever else robot.launch.py/cameras.launch.py need
# (ros2_robotiq_gripper, kortex_api, etc. — not covered by this backup,
# they're upstream/vendor and never modified here)

# then copy this backup on top, preserving paths:
cp -r /path/to/KinovaARM/robot_bringup/ros2_kortex/* \
      ~/workspace/ros2_kortex_ws/src/ros2_kortex/

colcon build --packages-select kinova_gen3_7dof_robotiq_2f_140_moveit_config kortex_bringup
```

## Keeping this in sync

This is a manual snapshot, not a live mirror — if you or your labmate edit
`robot.launch.py`, `gen3.srdf`, or anything else in the real
`~/workspace/ros2_kortex_ws`, this backup goes stale until someone re-copies.
Worth doing after any meaningful change to shared bringup config, not just
once.
