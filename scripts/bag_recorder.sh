#!/usr/bin/env bash
# Episode bag: joint states, end-effector pose, gripper state and the command / status / scene topics. NO camera images (they fill the disk).
#
#   scripts/bag_recorder.sh start      begin a new session bag in ~/.ros/bags/session-YYYYmmdd-HHMMSS
#   scripts/bag_recorder.sh stop       close it cleanly
#   scripts/bag_recorder.sh status     what is running, the newest bag and its size
#
# Rolls over to a new file every hour (--max-bag-duration), zstd-compressed. About 15 MB per hour of idle arm, a few times that while moving. Only LISTENS; never commands the arm.
# (no set -u: the ROS setup files read unset variables)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$HOME/.ros/bags"
mkdir -p "$DIR"
source "$HERE/ros_env.sh" >/dev/null 2>&1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
case "${1:-status}" in
  start)
    if alive "$DIR/recorder.pid"; then echo "already recording (pid $(cat "$DIR/recorder.pid"))"; exit 0; fi
    OUT="$DIR/session-$(date +%Y%m%d-%H%M%S)"
    setsid nohup python3 "$HERE/episode_bag_helper.py" > "$DIR/helper.log" 2>&1 &
    echo $! > "$DIR/helper.pid"
    sleep 3
    setsid nohup ros2 bag record -o "$OUT" --compression-mode file --compression-format zstd --max-bag-duration 3600 \
      /rec/joint_states /rec/gripper /rec/ee_pose /rec/pad_midpoint /rec/meta \
      /arm_status /pick_place_status /planner_status /action_plan /voice_command /gripper_held /scene_snapshot > "$DIR/recorder.log" 2>&1 &
    echo $! > "$DIR/recorder.pid"
    echo "recording to $OUT (recorder pid $(cat "$DIR/recorder.pid"), helper pid $(cat "$DIR/helper.pid"))"
    ;;
  stop)
    for name in recorder helper; do
      if alive "$DIR/$name.pid"; then kill -INT "-$(cat "$DIR/$name.pid")" 2>/dev/null || kill -INT "$(cat "$DIR/$name.pid")" 2>/dev/null; echo "stopped $name"; fi
      rm -f "$DIR/$name.pid"
    done
    ;;
  status)
    for name in recorder helper; do if alive "$DIR/$name.pid"; then echo "$name running (pid $(cat "$DIR/$name.pid"))"; else echo "$name not running"; fi; done
    ls -dt "$DIR"/session-* 2>/dev/null | head -3 | while read -r d; do echo "$d  $(du -sh "$d" | cut -f1)"; done
    ;;
  *) echo "usage: $0 start|stop|status"; exit 2;;
esac
