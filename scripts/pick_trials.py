#!/usr/bin/env python3
"""
Repeatable pick trials on the LIVE arm: pick an object, put it back where it was, repeat, and tabulate what happened. THIS MOVES THE ARM: run it only with the table set and nobody within reach of the robot.

    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/pick_trials.py --plan "blue cup:3,white cup:3,mouse:3" --check     # prints what it would do, publishes nothing
    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/pick_trials.py --plan "blue cup:3,white cup:3,mouse:3"             # runs the trials

Each trial publishes the plan the planner would make for "pick up the <colour> <object> and put it aside" (command_grammar.build_plan), but with the place step at the object's ORIGINAL position, so the
scene stays as it was. The arm controller does everything else (its own checks, side/top choice, retries), and its outcome log (~/.ros/outcomes) is read for the result of every pick and place.

Stops by itself when: the arm is not READY or something is held at the start of a trial, the object is not in the scene (fresh, reachable) or has moved off, a trial fails twice in a row, a put-back fails
(the object may still be in the gripper), or the time limit of a trial passes. Ctrl-C stops publishing new trials (it cannot stop a plan already running: that is the hardware E-stop).
Results are printed and written to docs/data/pick_trials_<date>.txt.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from thesis_robot import command_grammar as cg

OUTCOMES = os.path.expanduser('~/.ros/outcomes')
COLOURS = tuple(sorted(cg.COLOURS.values()))
TRIAL_TIMEOUT_S = 200.0


def parse_plan(text):
    """'blue cup:3,white cup:2,mouse:3' -> [(colour or None, label, n)]."""
    out = []
    for item in text.split(','):
        name, _, n = item.strip().rpartition(':')
        words = name.split()
        colour = words[0] if len(words) > 1 and words[0] in COLOURS else None
        label = ' '.join(words[1:] if colour else words)
        out.append((colour, cg.SYNONYMS.get(label, label), int(n)))
    return out


def outcome_file():
    files = sorted(glob.glob(os.path.join(OUTCOMES, 'outcomes-*.jsonl')))
    return files[-1] if files else None


def new_records(path, offset):
    with open(path) as f:
        f.seek(offset)
        data = f.read()
        end = f.tell()
    out = []
    for ln in data.splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out, end


class Trials(Node):
    def __init__(self):
        super().__init__('pick_trials')
        self.scene, self.arm_status, self.held = {}, '', ''
        self.create_subscription(String, '/scene_snapshot', lambda m: self._set_scene(m.data), 5)
        self.create_subscription(String, '/arm_status', lambda m: setattr(self, 'arm_status', m.data), 5)
        self.create_subscription(String, '/gripper_held', lambda m: setattr(self, 'held', m.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub = self.create_publisher(String, '/action_plan', 10)

    def _set_scene(self, data):
        try:
            self.scene = json.loads(data)
        except ValueError:
            pass

    def spin_for(self, secs):
        t = time.time() + secs
        while time.time() < t and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def find(self, colour, label, near=None):
        """(object_id, entry) of the fresh, reachable object of that label/colour (the one nearest `near` when given), or None."""
        c = [(oid, o) for oid, o in self.scene.items()
             if o.get('label') == label and not o.get('stale') and o.get('reachable') and (colour is None or o.get('color') == colour)]
        if not c:
            return None
        if near is not None:
            c.sort(key=lambda it: math.hypot(it[1]['x'] - near[0], it[1]['y'] - near[1]))
        return c[0]

    def holding(self):
        """True when the arm controller says something is held ({'held': {...}}); no message yet = nothing held."""
        try:
            return bool(json.loads(self.held).get('held')) if self.held.strip() else False
        except ValueError:
            return True                                       # unreadable: treat as held, stop

    def make_plan(self, colour, label, oid):
        words = f'{colour} {label}' if colour else label
        intent = cg.parse(f'pick up the {words} and put it aside')
        plan = cg.build_plan(intent, self.scene) if intent else None
        if not plan or plan[0].get('action') != 'pick':
            return None
        o = self.scene[oid]
        plan[0]['object_id'] = oid
        return [plan[0], {'action': 'place', 'x': round(o['x'], 4), 'y': round(o['y'], 4)}]


def summarise(rows):
    lines = ['trial  object      start (x,y)       pick                         place     secs   end drift', '-' * 98]
    for r in rows:
        lines.append(f"{r['n']:>3}   {r['name']:<10}  ({r['x']:.3f},{r['y']:.3f})   {r['pick']:<28} {r['place']:<9} {r['secs']:>5.0f}   {r['drift']}")
    ok = sum(1 for r in rows if r['pick'].startswith('ok'))
    lines.append(f'picks that worked: {ok}/{len(rows)}')
    for name in sorted({r['name'] for r in rows}):
        sub = [r for r in rows if r['name'] == name]
        lines.append(f"  {name}: {sum(1 for r in sub if r['pick'].startswith('ok'))}/{len(sub)}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True, help='"blue cup:3,white cup:3,mouse:3"')
    ap.add_argument('--check', action='store_true', help='only print the plans, publish nothing')
    ap.add_argument('--pause', type=float, default=6.0, help='seconds between trials')
    ap.add_argument('--max-consecutive-failures', type=int, default=2)
    args = ap.parse_args()
    rclpy.init()
    node = Trials()
    node.spin_for(3.0)
    rows, fails, n = [], 0, 0
    path = outcome_file()
    stop = False
    try:
        for colour, label, count in parse_plan(args.plan):
            name = f'{colour} {label}' if colour else label
            for k in range(count):
                if stop:
                    break
                node.spin_for(1.0)
                found = node.find(colour, label)
                if not found:
                    print(f'STOP: no fresh reachable {name} in the scene')
                    stop = True
                    break
                oid, o = found
                if 'READY' not in node.arm_status or node.holding():
                    print(f'STOP: the arm is not ready/empty (status {node.arm_status!r}, held {node.held!r})')
                    stop = True
                    break
                plan = node.make_plan(colour, label, oid)
                if plan is None:
                    print(f'STOP: could not build a plan for {name}')
                    stop = True
                    break
                n += 1
                print(f'\n=== trial {n}: {name} {oid} at ({o["x"]:.3f},{o["y"]:.3f}) z={o["z"]}  plan {json.dumps(plan)}')
                if args.check:
                    continue
                offset = os.path.getsize(path) if path else 0
                node.pub.publish(String(data=json.dumps({'command': f'trial: pick up the {name} and put it back', 'plan': plan, 'model': 'pick_trials'})))
                t0 = time.time()
                pick = place = None
                while time.time() - t0 < TRIAL_TIMEOUT_S and rclpy.ok():
                    node.spin_for(1.0)
                    recs, offset = new_records(path, offset)
                    for r in recs:
                        if r.get('kind') == 'pick' and pick is None and r.get('live'):
                            pick = r
                        elif r.get('kind') == 'place' and r.get('live'):
                            place = r
                    if place is not None or (pick is not None and not pick.get('ok')):
                        break
                node.spin_for(max(args.pause, 2.0))
                after = node.find(colour, label, near=(o['x'], o['y']))
                drift = '-' if after is None else f'{math.hypot(after[1]["x"] - o["x"], after[1]["y"] - o["y"]) * 100:.1f} cm'
                pk = 'NO RECORD (timeout)' if pick is None else ('ok ' + str(pick.get('method', '')) + f" a{pick.get('attempt', 1)}" if pick.get('ok')
                                                                  else f"FAILED {pick.get('failed_step')}: {str(pick.get('reason'))[:60]}")
                pl = '-' if place is None else ('ok' if place.get('ok') else 'FAILED')
                rows.append({'n': n, 'name': name, 'x': o['x'], 'y': o['y'], 'pick': pk, 'place': pl, 'secs': time.time() - t0, 'drift': drift})
                print('   ->', rows[-1]['pick'], '| place', pl, '|', f'{rows[-1]["secs"]:.0f}s', '| object now', drift, 'from where it started')
                if pick is None or not pick.get('ok'):
                    fails += 1
                else:
                    fails = 0
                if pick is not None and pick.get('ok') and (place is None or not place.get('ok')):
                    print('STOP: the pick worked but the put-back did not: the object may still be in the gripper. Check the arm.')
                    stop = True
                if fails >= args.max_consecutive_failures:
                    print(f'STOP: {fails} failures in a row')
                    stop = True
    except KeyboardInterrupt:
        print('interrupted: no further trials will be published')
    node.destroy_node()
    rclpy.shutdown()
    if rows:
        text = summarise(rows)
        print('\n' + text)
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'data')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'pick_trials_{time.strftime("%Y-%m-%d_%H%M")}.txt'), 'w') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
