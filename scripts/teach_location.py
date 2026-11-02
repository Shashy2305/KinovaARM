#!/usr/bin/env python3
"""
Record where the arm is NOW as a named location (READ-ONLY: it only listens; it never moves the arm). Use it after you have jogged the gripper onto a cup standing in the coffee machine's tray and closed
the gripper: it stores the cup position (the pad midpoint), the way the tool points, the finger position and the joints in ~/.ros/locations.yaml.

    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/teach_location.py record coffee_tray_grasp --note "cup in the tray, gripper closed on it"
    python3 scripts/teach_location.py list
    python3 scripts/teach_location.py show coffee_tray_grasp
"""
import argparse
import math
import os
import time

import yaml

LOCATIONS = os.path.expanduser('~/.ros/locations.yaml')


def load():
    try:
        with open(LOCATIONS) as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}


def summarise(name, e):
    pm, ax = e['pad_midpoint'], e['tool_axis']
    return (f"{name}: cup/pad midpoint ({pm[0]:.3f}, {pm[1]:.3f}, {pm[2]:.3f}) m | tool points toward azimuth {e['approach_azimuth_deg']:.0f} deg, {e['tool_tilt_below_horizontal_deg']:.1f} deg below horizontal"
            f" | finger {e['finger']:.3f} | flange ({e['flange'][0]:.3f}, {e['flange'][1]:.3f}, {e['flange'][2]:.3f}) | {e['recorded']} | {e.get('note', '')}")


def record(name, note):
    import rclpy
    import tf2_ros
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    rclpy.init()
    n = Node('teach_location')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    js = {}
    n.create_subscription(JointState, '/joint_states', lambda m: js.update(dict(zip(m.name, m.position))), 5)
    t0 = time.time()
    while time.time() - t0 < 4.0 and ('finger_joint' not in js or not buf.can_transform('base_link', 'end_effector_link', rclpy.time.Time())):
        rclpy.spin_once(n, timeout_sec=0.1)

    def pos(frame):
        t = buf.lookup_transform('base_link', frame, rclpy.time.Time()).transform
        return [t.translation.x, t.translation.y, t.translation.z], [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    flange, fq = pos('end_effector_link')
    a, _ = pos('left_inner_finger_pad')
    b, _ = pos('right_inner_finger_pad')
    pad = [(a[i] + b[i]) / 2.0 for i in range(3)]
    ax = [pad[i] - flange[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in ax))
    ax = [v / norm for v in ax]
    entry = {
        'recorded': time.strftime('%Y-%m-%d %H:%M:%S'), 'note': note or '',
        'pad_midpoint': [round(v, 4) for v in pad], 'flange': [round(v, 4) for v in flange], 'flange_quat_xyzw': [round(v, 5) for v in fq],
        'tool_axis': [round(v, 4) for v in ax], 'approach_azimuth_deg': round(math.degrees(math.atan2(ax[1], ax[0])) % 360.0, 1),
        'tool_tilt_below_horizontal_deg': round(math.degrees(math.asin(max(-1.0, min(1.0, -ax[2])))), 1),
        'finger': round(float(js.get('finger_joint', float('nan'))), 4),
        'joints_rad': [round(float(js.get(f'joint_{i}', float('nan'))), 4) for i in range(1, 8)],
    }
    rclpy.shutdown()
    data = load()
    data[name] = entry
    with open(LOCATIONS, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print('saved ->', LOCATIONS)
    print(summarise(name, entry))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('name')
    r.add_argument('--note', default='')
    sub.add_parser('list')
    s = sub.add_parser('show')
    s.add_argument('name')
    a = ap.parse_args()
    if a.cmd == 'record':
        record(a.name, a.note)
    elif a.cmd == 'list':
        for k, v in load().items():
            print(summarise(k, v))
    else:
        v = load().get(a.name)
        print(yaml.safe_dump({a.name: v}, sort_keys=False) if v else f'no location called {a.name}')


if __name__ == '__main__':
    main()
