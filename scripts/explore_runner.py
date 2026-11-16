#!/usr/bin/env python3
"""
Runs an exploration campaign for ONE object class without an operator typing a command per pick: every step picks the object up where it stands and puts it down at a NEW spot, so the next step
picks it from a new place (each command gives one pick and one place of data). THIS MOVES THE ARM: run it only with the table set, nobody within reach of the arm, the arm controller LIVE and
exploration mode already ON (it never switches exploration on: that is a typed step of the operator).

    source scripts/ros_env.sh; python3 scripts/explore_runner.py --label mouse --picks 50 --scene-prefix mouse_2026-11-16 --check     # prints the spots, publishes nothing
    source scripts/ros_env.sh; python3 scripts/explore_runner.py --label mouse --picks 50 --scene-prefix mouse_2026-11-16

What it does for you:
  * spots come from a quasi-random (Halton) sequence over the reachable part of the table, at least 15 cm from every other object and 10 cm from where the object is now; a spot that was refused
    at the hover is never tried again (and neither is one within 5 cm of it)
  * `--picks N` counts picks that REACHED the grasp (the exploration draws top or side at random and a side grasp of a low object is refused at once, which is quick and still logged, but is not a pick)
  * a layout id (scene_id, so the train/test split by scene works) is set on the arm controller every --per-scene picks
  * waits for READY and an empty gripper before every command, stops after 3 failures in a row, when the object is not in the scene, or on Ctrl-C (a plan already running is stopped only by the E-stop)
Results: printed, and written to docs/data/explore_runner_<label>_<date>.txt.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from rcl_interfaces.srv import SetParameters

OUTCOMES = os.path.expanduser('~/.ros/outcomes')
REGION_X, REGION_Y = (0.28, 0.56), (-0.42, 0.14)         # where a spot may be (the machine stands at y 0.20 to 0.42)
MIN_FROM_OTHERS_M, MIN_MOVE_M, BAD_RADIUS_M = 0.15, 0.10, 0.05
EARLY_STEPS = ('open gripper', 'raise', 'hover', 'choose how to grasp it')


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def read_new_records(path, offset):
    out = []
    with open(path) as f:
        f.seek(offset)
        data = f.read()
        offset = f.tell()
    for ln in data.splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out, offset


class Runner(Node):
    def __init__(self):
        super().__init__('explore_runner')
        self.scene, self.held = {}, ''
        self.create_subscription(String, '/scene_snapshot', self._scene, 5)
        self.create_subscription(String, '/gripper_held', lambda m: setattr(self, 'held', m.data), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub = self.create_publisher(String, '/action_plan', 10)
        self.param_client = self.create_client(SetParameters, '/arm_controller/set_parameters')

    def _scene(self, data):
        try:
            self.scene = json.loads(data.data)
        except ValueError:
            pass

    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def arm_status(self):
        try:
            with urllib.request.urlopen('http://localhost:8000/api/status', timeout=3) as r:
                return str((json.load(r).get('ros') or {}).get('arm_status') or '')
        except Exception:
            return ''

    def holding(self):
        try:
            return bool(json.loads(self.held).get('held'))
        except ValueError:
            return False

    def find(self, label):
        best = None
        for oid, o in self.scene.items():
            if isinstance(o, dict) and o.get('label') == label and not o.get('stale') and o.get('reachable') and o.get('x') is not None:
                if best is None or o.get('confidence', 0) > best[1].get('confidence', 0):
                    best = (oid, o)
        return best

    def others(self, oid):
        return [(o['x'], o['y']) for k, o in self.scene.items()
                if isinstance(o, dict) and k != oid and not o.get('stale') and o.get('x') is not None and o.get('label') not in ('red_obj', 'blue_obj', 'green_obj', 'yellow_obj')]

    def set_scene_id(self, name):
        if not self.param_client.wait_for_service(timeout_sec=3.0):
            print('  (could not reach /arm_controller/set_parameters: scene_id not changed)')
            return False
        req = SetParameters.Request(parameters=[Parameter('scene_id', Parameter.Type.STRING, name).to_parameter_msg()])
        fut = self.param_client.call_async(req)
        end = time.time() + 5
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return bool(fut.done() and fut.result().results and fut.result().results[0].successful)


def pick_spot(runner, oid, here, bad, counter):
    others = runner.others(oid)
    for _ in range(400):
        counter[0] += 1
        i = counter[0]
        x = REGION_X[0] + (REGION_X[1] - REGION_X[0]) * halton(i, 2)
        y = REGION_Y[0] + (REGION_Y[1] - REGION_Y[0]) * halton(i, 3)
        if math.hypot(x - here[0], y - here[1]) < MIN_MOVE_M:
            continue
        if any(math.hypot(x - ox, y - oy) < MIN_FROM_OTHERS_M for ox, oy in others):
            continue
        if any(math.hypot(x - bx, y - by) < BAD_RADIUS_M for bx, by in bad):
            continue
        return round(x, 3), round(y, 3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='mouse')
    ap.add_argument('--picks', type=int, default=50, help='picks that reached the grasp')
    ap.add_argument('--max-attempts', type=int, default=0, help='commands to send at most (default 2.5 x picks)')
    ap.add_argument('--scene-prefix', default='')
    ap.add_argument('--per-scene', type=int, default=5)
    ap.add_argument('--start-index', type=int, default=1, help='where the spot sequence starts (so a second run does not repeat the first)')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    max_attempts = args.max_attempts or int(args.picks * 2.5)
    rclpy.init()
    node = Runner()
    node.spin_for(2.0)
    month = time.strftime('%Y-%m')
    path = os.path.join(OUTCOMES, f'outcomes-{month}.jsonl')
    offset = os.path.getsize(path)
    bad, counter, rows, fails_in_row, reached, attempts = [], [args.start_index], [], 0, 0, 0
    try:
        while reached < args.picks and attempts < max_attempts:
            node.spin_for(1.0)
            found = node.find(args.label)
            if not found:
                print(f'STOP: no fresh reachable {args.label} in the scene')
                break
            oid, o = found
            here = (o['x'], o['y'])
            if not args.check:
                st = node.arm_status()
                if 'READY' not in st or node.holding():
                    print(f'STOP: the arm is not ready/empty (status {st!r}, held {node.held!r})')
                    break
            spot = pick_spot(node, oid, here, bad, counter)
            if spot is None:
                print('STOP: no free spot left')
                break
            attempts += 1
            scene_name = f'{args.scene_prefix}_{(reached // args.per_scene):04d}' if args.scene_prefix else ''
            cmd = f'explore: move the {args.label} to ({spot[0]:.2f},{spot[1]:+.2f})'
            plan = [{'action': 'pick', 'object_id': oid, 'approach_z': 0.35}, {'action': 'place', 'x': spot[0], 'y': spot[1]}]
            print(f'#{attempts} ({reached} picks so far): {args.label} {oid} at ({here[0]:.3f},{here[1]:+.3f}) -> ({spot[0]:.3f},{spot[1]:+.3f})  scene {scene_name or "(unchanged)"}', flush=True)
            if args.check:
                continue
            if scene_name:
                node.set_scene_id(scene_name)
            m = String()
            m.data = json.dumps({'command': cmd, 'plan': plan, 'model': 'manual'})
            node.pub.publish(m)
            t0, seen_exec, done = time.time(), False, False
            while time.time() - t0 < 300:
                node.spin_for(1.0)
                st = node.arm_status()
                if 'EXECUTING' in st:
                    seen_exec = True
                recs, offset = read_new_records(path, offset)
                if any(r.get('kind') == 'command' and r.get('command') == cmd for r in recs):
                    rows_new = [r for r in recs if r.get('live') and r.get('kind') in ('pick', 'place', 'command')]
                    done = True
                    break
                if seen_exec and 'READY' in st and time.time() - t0 > 8:
                    node.spin_for(2.0)
                    recs2, offset = read_new_records(path, offset)
                    rows_new = [r for r in recs + recs2 if r.get('live') and r.get('kind') in ('pick', 'place', 'command')]
                    done = True
                    break
                if 'BLOCKED' in st or 'ERROR' in st:
                    break
            if not done:
                print(f'STOP: the command did not finish in time (status {node.arm_status()!r}); check the arm')
                break
            pk = next((r for r in rows_new if r['kind'] == 'pick'), None)
            pl = next((r for r in rows_new if r['kind'] == 'place'), None)
            cmdrec = next((r for r in rows_new if r['kind'] == 'command'), None)
            ok = bool(cmdrec and cmdrec.get('ok'))
            step = (pk or {}).get('failed_step') if pk and not pk.get('ok') else ((pl or {}).get('failed_step') if pl and not pl.get('ok') else None)
            got_to_grasp = bool(pk and (pk.get('ok') or pk.get('failed_step') not in EARLY_STEPS))
            if got_to_grasp:
                reached += 1
            if pk and not pk.get('ok') and pk.get('failed_step') in ('hover', 'raise'):
                bad.append(here)                                                        # a pick from HERE is refused: never come back to this spot
            rows.append({'n': attempts, 'from': here, 'to': spot, 'ok': ok, 'reached': got_to_grasp, 'step': step, 'draw': ((pk or {}).get('explore') or {}).get('method')})
            fails_in_row = 0 if ok else fails_in_row + 1
            print(f'   -> {"ok" if ok else "FAILED " + str(step)}  (draw {rows[-1]["draw"]}, reached the grasp: {got_to_grasp}, picks so far {reached})', flush=True)
            if not ok and (pk or {}).get('failed_step') == 'choose how to grasp it':
                fails_in_row = 0                                                       # a refused side grasp is quick, harmless and expected for a low object
            if fails_in_row >= 3:
                print('STOP: 3 failures in a row')
                break
    except KeyboardInterrupt:
        print('stopped by the operator (a plan already running keeps going until it ends)')
    ok_n = sum(1 for r in rows if r['ok'])
    lines = [f'attempts {len(rows)}, picks that reached the grasp {reached}, commands ok {ok_n}']
    for r in rows:
        lines.append(f"{r['n']:>3} ({r['from'][0]:.3f},{r['from'][1]:+.3f}) -> ({r['to'][0]:.3f},{r['to'][1]:+.3f})  {'ok' if r['ok'] else 'FAILED ' + str(r['step']):<28} draw {r['draw']}")
    text = '\n'.join(lines)
    print('\n' + text)
    if rows and not args.check:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'data', f'explore_runner_{args.label}_{time.strftime("%Y-%m-%d_%H%M")}.txt')
        with open(out, 'w') as f:
            f.write(text + '\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
