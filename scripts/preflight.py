#!/usr/bin/env python3
"""
Preflight: is the whole stack healthy enough to start testing? One command, PASS / WARN / FAIL per check, a non-zero exit on FAIL.
Observes only; commands nothing.

    source scripts/ros_env.sh && python3 scripts/preflight.py [--tests] [--bench]

  --tests   also run the unit tests (needs ROS sourced before the venv; this script does that in a subprocess)
  --bench   also run a short planner benchmark (Dry Run only)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import rclpy
from example_interfaces.msg import Bool
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = []


def report(level, name, detail=''):
    RESULTS.append((level, name))
    colour = {'PASS': '\033[92m', 'WARN': '\033[93m', 'FAIL': '\033[91m'}[level]
    print(f'{colour}{level}\033[0m  {name}' + (f'  - {detail}' if detail else ''), flush=True)


def api(path, timeout=5):
    with urllib.request.urlopen(f'http://localhost:8000/api{path}', timeout=timeout) as r:
        return json.loads(r.read())


class Probe(Node):
    def __init__(self):
        super().__init__('preflight')
        self.js, self.fault, self.scene, self.unknown = 0, None, {}, None
        self.create_subscription(JointState, '/joint_states', lambda m: setattr(self, 'js', self.js + 1), 10)
        self.create_subscription(Bool, '/fault_controller/internal_fault', lambda m: setattr(self, 'fault', bool(m.data)), 1)
        self.create_subscription(String, '/scene_snapshot', self._scene, 2)
        self.create_subscription(String, '/unknown_obstacles', lambda m: setattr(self, 'unknown', m.data), 2)

    def _scene(self, m):
        try:
            self.scene = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)


def check_overnight(hours=14):
    path = os.path.expanduser('~/.ros/health/health.csv')
    if not os.path.exists(path):
        return report('WARN', 'overnight health log', f'{path} does not exist (health_monitor.py not running?)')
    cutoff = time.time() - hours * 3600
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                if time.mktime(time.strptime(r['time'], '%Y-%m-%d %H:%M:%S')) >= cutoff:
                    rows.append(r)
            except (ValueError, KeyError, TypeError):
                continue
    if len(rows) < 5:
        return report('WARN', 'overnight health log', f'only {len(rows)} samples in the last {hours} h')
    times = [time.mktime(time.strptime(r['time'], '%Y-%m-%d %H:%M:%S')) for r in rows]
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a > 180]
    faults = sum(1 for r in rows if r['fault'] == 'FAULT')
    low_js = sum(1 for r in rows if float(r['joint_states_hz'] or 0) < 100)
    cam_bad = sum(1 for r in rows if any(c in r['cam_status'] for c in ('rec', 'no_')) or r['flags'] not in ('-', ''))
    live = sum(1 for r in rows if 'LIVE' in r['arm_status'])
    load = max(float(r['load1']) for r in rows)
    detail = f"{len(rows)} samples, max load {load:.0f}, joint_states<100Hz {low_js}, faults {faults}, camera problems {cam_bad}, monitor gaps>3min {len(gaps)}, LIVE minutes {live}"
    level = 'PASS' if (not faults and not low_js and not cam_bad and not gaps) else 'WARN'
    report(level, 'overnight health log', detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', action='store_true')
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()

    try:
        status = api('/status')
        report('PASS', 'dashboard backend', 'answers on :8000')
    except Exception as e:
        report('FAIL', 'dashboard backend', str(e))
        status = {'ros': {}, 'processes': {}}
    ros = status.get('ros') or {}
    arm = ros.get('arm_status') or '?'
    report('PASS' if 'DRY RUN' in arm else 'WARN', 'arm mode', f'{arm}' + ('' if 'DRY RUN' in arm else '  (LIVE: the arm moves)'))
    procs = status.get('processes') or {}
    down = [k for k, v in procs.items() if v.get('status') == 'stopped' and k not in ('oakd_driver', 'oakd_tf_broadcaster', 'object_detection', 'audio_node')]
    report('PASS' if not down else 'WARN', 'processes', 'all expected nodes running' if not down else f'stopped: {", ".join(down)}')

    rclpy.init()
    n = Probe()
    n.spin(6.0)
    report('PASS' if n.js / 6.0 > 100 else 'FAIL', 'joint states', f'{n.js / 6.0:.0f} Hz' + ('' if n.js / 6.0 > 100 else '  (restart robot_bringup)'))
    report('PASS' if n.fault is False else ('FAIL' if n.fault else 'WARN'), 'arm fault flag',
           {False: 'clear', True: 'FAULT: reset with the fault controller, hands off', None: 'no reading'}[n.fault])
    cam = ros.get('camera_calibration_status') or {}
    bad = {k: v for k, v in cam.items() if k != 'oakd' and v != 'ok'}
    report('PASS' if cam and not bad else 'FAIL', 'cameras', ', '.join(f'{k}:{v}' for k, v in cam.items()) or 'no status')
    flags = [f for f in ('realsense', 'realsense2') if os.path.exists(os.path.expanduser(f'~/.ros/{f}_needs_recalibration.flag'))]
    report('PASS' if not flags else 'WARN', 'recalibration flags', 'none' if not flags else f'present for {flags}: run calibration/check_extrinsics.py, then delete the flag if the cameras agree')
    fresh = {k: (v['label'], round(v['x'], 2), round(v['y'], 2)) for k, v in n.scene.items() if isinstance(v, dict) and not v.get('stale')}
    report('PASS' if fresh else 'WARN', 'scene', f'{len(fresh)} fresh objects: ' + ', '.join(f'{l}({x},{y})' for l, x, y in fresh.values()) if fresh else 'no fresh objects')
    try:
        unk = json.loads(n.unknown or '{}').get('obstacles', [])
        conf = [o for o in unk if o.get('confirmed')]
        report('PASS', 'unknown obstacles', f'{len(conf)} confirmed, {len(unk) - len(conf)} tentative')
    except json.JSONDecodeError:
        report('WARN', 'unknown obstacles', 'obstacle_guard is not publishing')
    n.destroy_node()
    rclpy.shutdown()

    try:
        out = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10).stdout.splitlines()
        line = out[1] if len(out) > 1 else ''
        parts = line.split()
        ctx = parts[6] if len(parts) > 6 else '?'                        # NAME ID SIZE(2) PROCESSOR(2) CONTEXT UNTIL...
        report('PASS' if 'qwen2.5' in line else 'WARN', 'ollama',
               f'{parts[0]} loaded, context {ctx}' if line else 'model not loaded (the first command takes ~30-80 s)')
    except Exception as e:
        report('FAIL', 'ollama', str(e))

    free = shutil.disk_usage('/').free / 1e9
    report('PASS' if free > 20 else 'FAIL', 'disk /', f'{free:.0f} GB free')
    mounted = subprocess.run(['findmnt', '-n', '/mnt/ros_workspace'], capture_output=True, text=True).stdout.strip()
    report('PASS' if mounted else 'FAIL', 'workspace volume', mounted.split('  ')[0] if mounted else 'NOT MOUNTED')
    try:
        subprocess.run(['touch', '/mnt/ros_workspace/.preflight'], check=True, timeout=10)
        os.remove('/mnt/ros_workspace/.preflight')
        report('PASS', 'workspace volume writable')
    except Exception as e:
        report('FAIL', 'workspace volume writable', str(e))
    check_overnight()
    head = subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', REPO, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
    report('PASS' if not dirty else 'WARN', 'git', head[:80] + ('' if not dirty else f'  ({len(dirty.splitlines())} uncommitted files)'))

    if args.tests:
        cmd = (f'source /opt/ros/humble/setup.bash && source /mnt/ros_workspace/venv/bin/activate && cd {REPO} && '
               f'PYTHONPATH=$PYTHONPATH:src/thesis_robot python3 -m pytest -p no:anyio src/thesis_robot/test -q '
               f'--ignore=src/thesis_robot/test/test_copyright.py --ignore=src/thesis_robot/test/test_flake8.py --ignore=src/thesis_robot/test/test_pep257.py 2>&1 | tail -1')
        out = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=600).stdout.strip()
        report('PASS' if 'passed' in out and 'failed' not in out else 'FAIL', 'unit tests', out)
    if args.bench:
        if 'DRY RUN' not in arm:
            report('WARN', 'planner benchmark', 'skipped: the arm is not in Dry Run')
        else:
            out = subprocess.run(['python3', os.path.join(REPO, 'scripts', 'planner_benchmark.py'), '--reps', '2', '--label', 'preflight'],
                                 capture_output=True, text=True, timeout=1500).stdout.strip().splitlines()
            total = next((l for l in reversed(out) if 'TOTAL' in l), '?')
            ok = '100%' in total or any(f'({p}%)' in total for p in range(90, 100))
            report('PASS' if ok else 'WARN', 'planner benchmark', total.strip())

    print()
    counts = {k: sum(1 for r in RESULTS if r[0] == k) for k in ('PASS', 'WARN', 'FAIL')}
    print(f"{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    sys.exit(1 if counts['FAIL'] else 0)


if __name__ == '__main__':
    main()
