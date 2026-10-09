#!/usr/bin/env python3
"""
Planner benchmark: send a suite of commands through the REAL planner and check each plan's structure.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 scripts/planner_benchmark.py --reps 4 --label memory-on
    ros2 param set /llm_planner planner_memory false          # then run again with --label memory-off to compare

Safety: the arm controller must be in DRY RUN (the script refuses to start if the arm status says LIVE). The commands
are executed by the controller in dry run, which only logs. Objects come from the live scene: it uses whatever
reachable labels are on the table (cup, bowl, bottle, mouse...).

Results are printed and saved to ~/.ros/planner_benchmark/<timestamp>-<label>.json
"""
import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def pick_ids(plan):
    return [s.get('object_id') for s in plan if isinstance(s, dict) and s.get('action') == 'pick']


def label_of(scene, oid):
    return (scene.get(oid) or {}).get('label')


def near_target(step, scene, label_b, tol=0.30):
    """A place step puts the object next to an object of label_b: by `near` id, or by x,y within `tol` of it."""
    if step.get('near'):
        return label_of(scene, step['near']) == label_b
    if 'x' in step and 'y' in step:
        return any(math.hypot(o['x'] - step['x'], o['y'] - step['y']) <= tol
                   for o in scene.values() if isinstance(o, dict) and o.get('label') == label_b and o.get('x') is not None)
    return False


def suite(a, b):
    """[(name, command text, check(plan, scene) -> (ok, why))] for object labels a and b."""
    def one_pick(plan, scene):
        p = pick_ids(plan)
        if len(p) != 1:
            return False, f'{len(p)} pick steps'
        return (label_of(scene, p[0]) == a, f'picked a {label_of(scene, p[0])}, wanted a {a}')

    def pick_place_next(plan, scene):
        ok, why = one_pick(plan, scene)
        if not ok:
            return ok, why
        places = [s for s in plan if s.get('action') == 'place']
        if len(places) != 1:
            return False, f'{len(places)} place steps'
        return (near_target(places[0], scene, b), f'place is not next to the {b}')

    def pick_aside(plan, scene):
        ok, why = one_pick(plan, scene)
        places = [s for s in plan if s.get('action') == 'place']
        return (ok and len(places) == 1, why if not ok else 'no single place step')

    def put_down(plan, scene):
        acts = [s.get('action') for s in plan]
        return (acts == ['place'] and plan[0].get('here') is True, f'steps {acts}')

    def go_home(plan, scene):
        return ([s.get('action') for s in plan] == ['go_home'], 'not a lone go_home')

    def go_near(plan, scene):
        acts = [s.get('action') for s in plan]
        if acts != ['move_to']:
            return False, f'steps {acts}'
        s = plan[0]
        return (any(math.hypot(o['x'] - s['x'], o['y'] - s['y']) < 0.08 for o in scene.values()
                    if isinstance(o, dict) and o.get('label') == a and o.get('x') is not None), f'not over the {a}')

    def pick_next_home(plan, scene):
        acts = [s.get('action') for s in plan]
        if acts != ['pick', 'place', 'go_home']:
            return False, f'steps {acts}'
        return pick_place_next(plan[:2], scene)

    return [
        ('pick', f'pick up the {a}', one_pick),
        ('pick_next', f'pick up the {a} and put it next to the {b}', pick_place_next),
        ('pick_next_para', f'grab the {a} and set it beside the {b}', pick_place_next),
        ('move_next', f'move the {a} next to the {b}', pick_place_next),
        ('pick_aside', f'pick up the {a} and put it aside', pick_aside),
        ('put_down', 'put it down', put_down),
        ('go_home', 'go home', go_home),
        ('go_near', f'go near the {a}', go_near),
        ('polite', f'could you please pick up the {a} and place it near the {b}', pick_place_next),
        ('compound', f'pick up the {a}, put it next to the {b} and then go home', pick_next_home),
    ]


class Bench(Node):
    def __init__(self):
        super().__init__('planner_benchmark')
        self.pub = self.create_publisher(String, '/voice_command', 10)
        self.plans, self.statuses, self.scene, self.arm = [], [], {}, ''
        self.last_meta = {}
        self.create_subscription(String, '/action_plan', lambda m: self.plans.append((time.time(), json.loads(m.data))), 10)
        self.create_subscription(String, '/planner_status', lambda m: self.statuses.append((time.time(), m.data)), 10)
        self.create_subscription(String, '/scene_snapshot', self._scene, 10)
        self.create_subscription(String, '/arm_status', lambda m: setattr(self, 'arm', m.data), 10)

    def _scene(self, m):
        try:
            self.scene = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_idle(self, timeout=60):
        """Planner not planning and the arm back to READY."""
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            last = self.statuses[-1][1] if self.statuses else 'READY'
            if last.startswith('READY') and self.arm.startswith('READY'):
                return True
        return False

    def ask(self, command, timeout=120):
        """(plan or None, seconds, planner status text) for one command."""
        self.wait_idle()
        n_plans, n_stat = len(self.plans), len(self.statuses)
        t0 = time.time()
        self.pub.publish(String(data=command))
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.plans) > n_plans:
                for ts, p in self.plans[n_plans:]:
                    if p.get('command') == command:
                        self.last_meta = {'attempts': p.get('attempts'), 'repaired': p.get('repaired'), 'memory': p.get('memory')}
                        return p.get('plan', []), ts - t0, 'APPROVED'
            for ts, s in self.statuses[n_stat:]:
                if s.startswith('REJECTED') or s.startswith('ERROR'):
                    return None, ts - t0, s
        return None, time.time() - t0, 'TIMEOUT'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--label', default='run')
    args = ap.parse_args()
    rclpy.init()
    b = Bench()
    b.spin_for(3.0)
    if 'LIVE' in b.arm:
        print(f'REFUSING TO RUN: the arm status is "{b.arm}". Put the arm in Dry Run first.')
        sys.exit(2)
    labels = sorted({o['label'] for o in b.scene.values()
                     if isinstance(o, dict) and o.get('reachable') and not o.get('stale') and o.get('label') in
                     ('cup', 'bowl', 'bottle', 'mouse')})
    if len(labels) < 2:
        print(f'need at least two reachable objects (cup/bowl/bottle/mouse) in the scene, have {labels}')
        sys.exit(2)
    pairs = list(itertools.permutations(labels, 2))
    results = {}
    for rep in range(args.reps):
        a, bb = pairs[rep % len(pairs)]
        for name, command, check in suite(a, bb):
            b.last_meta = {}
            plan, secs, status = b.ask(command)
            meta = dict(b.last_meta)
            scene = dict(b.scene)
            if plan is None:
                ok, why = False, status
            else:
                try:
                    ok, why = check(plan, scene)
                except Exception as e:
                    ok, why = False, f'check error {e}'
            results.setdefault(name, []).append({'command': command, 'ok': bool(ok), 'why': '' if ok else why,
                                                 'seconds': round(secs, 1), 'plan': plan, **meta})
            print(f'  [{args.label}] {name:15s} {"PASS" if ok else "FAIL"}  {secs:4.1f}s  {command!r}' + ('' if ok else f'  -> {why}'),
                  flush=True)
    print('\nSummary', args.label)
    total = ok_total = 0
    for name, rs in results.items():
        k = sum(r['ok'] for r in rs)
        total += len(rs)
        ok_total += k
        att = [r.get('attempts') for r in rs if r.get('attempts')]
        rep_n = sum(1 for r in rs if r.get('repaired'))
        extra = f'   attempts {statistics.mean(att):.1f}   repaired {rep_n}' if att else ''
        print(f'  {name:15s} {k}/{len(rs)}   mean {statistics.mean(r["seconds"] for r in rs):.1f}s{extra}')
    print(f'  TOTAL {ok_total}/{total} ({100 * ok_total / max(1, total):.0f}%)')
    out_dir = os.path.expanduser('~/.ros/planner_benchmark')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime('%Y%m%d-%H%M%S') + f'-{args.label}.json')
    with open(path, 'w') as f:
        json.dump({'label': args.label, 'reps': args.reps, 'results': results}, f, indent=1)
    print('saved', path)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
