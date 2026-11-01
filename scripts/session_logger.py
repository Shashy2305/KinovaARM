#!/usr/bin/env python3
"""
Passive session logger: records what happens while a person drives the robot (voice or dashboard commands) so that the errors can be found afterwards. It ONLY LISTENS: it publishes nothing and
sends nothing to the arm.

    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 -u scripts/session_logger.py --minutes 60 --out /path/session.log

One timestamped line per event:
  VOICE / CMD-IN   what was heard (/voice_transcript) and what was submitted as a command (/voice_command)
  PLAN             the planner's status messages (/planner_status)
  ARM / PP         arm status changes and the pick-and-place step messages
  PICK / PLACE / COMMAND   the arm's outcome records (~/.ros/outcomes, newest file, follows month roll-over): method, attempt, result, failed step, reason, seconds
  LOG              WARN / ERROR lines and the key decisions (grasp methods ranked, learned here, retries, refusals) of the arm controller's log
  SCENE            every 30 s: each fresh object (label, colour, position, reachable) and the camera-agreement medians
  HEALTH           whenever the set of dashboard health issues changes
"""
import argparse
import glob
import json
import os
import re
import signal
import time
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

OUTCOME_DIR = os.path.expanduser('~/.ros/outcomes')
ARM_LOG = os.path.expanduser('~/.ros/dashboard_logs/arm_controller.log')
PLANNER_LOG = os.path.expanduser('~/.ros/dashboard_logs/llm_planner_node.log')
PLAN_KEEP = re.compile(r'Command:|APPROVED|REJECTED|target \d|plan built by|LLM returned|repaired|dropped|WARN|ERROR', re.I)
KEEP = re.compile(r'ERROR|WARN|MOTION REFUSED|FAILED|retry|trying once more|grasp methods ranked|learned here|Executing:|side grasp|go_home|REJECTED|refus', re.I)
SKIP = re.compile(r'Joint states are available|not found: "/mnt|deprecat|UserWarning|declare_namespace|from scipy|pkg_resources', re.I)


class Logger(Node):
    def __init__(self, out):
        super().__init__('session_logger')
        self.out = out
        self.scene, self.agree, self.last_status = {}, {}, {}
        for topic, tag in (('/voice_transcript', 'VOICE'), ('/voice_command', 'CMD-IN'), ('/planner_status', 'PLAN'), ('/arm_status', 'ARM'), ('/pick_place_status', 'PP')):
            self.create_subscription(String, topic, lambda m, t=tag: self.on_text(t, m.data), 20)
        self.create_subscription(String, '/scene_snapshot', lambda m: self._load('scene', m.data), 5)
        self.create_subscription(String, '/camera_agreement', lambda m: self._load('agree', m.data), 2)
        self.out_pos = {}                                   # outcome file -> offset
        self.arm_log_pos, self.arm_log_id = 0, None
        self.plan_log_pos, self.plan_log_id = 0, None
        self.health_ids = None

    def w(self, tag, text):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {tag:<7} {text}'
        self.out.write(line + '\n')
        self.out.flush()

    def _load(self, name, data):
        try:
            setattr(self, name, json.loads(data))
        except ValueError:
            pass

    def on_text(self, tag, data):
        if tag in ('ARM', 'PP', 'PLAN'):
            if self.last_status.get(tag) == data:
                return
            self.last_status[tag] = data
        text = data
        if tag == 'CMD-IN':
            self.scene_line('SCENE@CMD')
        if tag == 'PLAN':
            try:
                d = json.loads(data)
                text = json.dumps(d)[:400]
            except ValueError:
                text = data[:400]
        self.w(tag, text.replace('\n', ' ')[:500])

    # ---- files --------------------------------------------------------------------------------------------------------------------------------------------
    def tail_outcomes(self):
        files = sorted(glob.glob(os.path.join(OUTCOME_DIR, 'outcomes-*.jsonl')))
        for f in files[-2:]:
            size = os.path.getsize(f)
            pos = self.out_pos.setdefault(f, size if self.first_pass else 0)   # at start: only what happens from now on; a file that appears later: from its start
            if size <= pos:
                continue
            with open(f) as fh:
                fh.seek(pos)
                data = fh.read()
                self.out_pos[f] = fh.tell()
            for ln in data.splitlines():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                self.outcome(r)

    def outcome(self, r):
        k = r.get('kind')
        if k == 'pick':
            self.w('PICK', f"{r.get('object_id')} ({r.get('label')}) at {r.get('xy')} method={r.get('method')} az={r.get('side_az', '')} attempt={r.get('attempt')} -> "
                           f"{'OK' if r.get('ok') else 'FAILED'} step={r.get('failed_step')} reason={str(r.get('reason'))[:140]} {r.get('seconds')}s "
                           f"neighbours={[(n.get('label'), n.get('dist_m')) for n in (r.get('neighbours') or [])][:3]}")
        elif k == 'place':
            self.w('PLACE', f"{r.get('object_id')} target={r.get('target')} strategy={r.get('strategy')} -> {'OK' if r.get('ok') else 'FAILED'} step={r.get('failed_step')} "
                            f"reason={str(r.get('reason'))[:140]} {r.get('seconds')}s")
        elif k == 'command':
            self.w('COMMAND', f"{r.get('command')!r} model={r.get('model')} plan={json.dumps(r.get('plan'))[:300]} -> {'OK' if r.get('ok') else 'FAILED'} step={r.get('failed_step')} "
                              f"reason={str(r.get('reason'))[:140]} {r.get('seconds')}s")

    def tail_arm_log(self):
        try:
            st = os.stat(ARM_LOG)
        except OSError:
            return
        if self.arm_log_id != st.st_ino or st.st_size < self.arm_log_pos:
            self.arm_log_id, self.arm_log_pos = st.st_ino, 0 if self.arm_log_id is not None else st.st_size       # restarted: read the new log from its start
        if st.st_size <= self.arm_log_pos:
            return
        with open(ARM_LOG, errors='replace') as fh:
            fh.seek(self.arm_log_pos)
            data = fh.read()
            self.arm_log_pos = fh.tell()
        for ln in data.splitlines():
            if KEEP.search(ln) and not SKIP.search(ln):
                self.w('LOG', re.sub(r'^\[(\w+)\] \[\d+\.\d+\] ', r'[\1] ', ln)[:330])

    # ---- periodic ------------------------------------------------------------------------------------------------------------------------------------------
    def scene_line(self, tag):
        objs = []
        for oid, o in sorted((self.scene or {}).items()):
            if isinstance(o, dict) and not o.get('stale') and o.get('label') not in ('coffee machine',) and o.get('x') is not None:
                objs.append(f"{oid}:{o.get('color')}@({o['x']:.2f},{o['y']:.2f}){'' if o.get('reachable') else '!unreach'}")
        self.w(tag, ' '.join(objs) or '(no objects)')

    def tail_planner_log(self):
        try:
            st = os.stat(PLANNER_LOG)
        except OSError:
            return
        if self.plan_log_id != st.st_ino or st.st_size < self.plan_log_pos:
            self.plan_log_id, self.plan_log_pos = st.st_ino, 0 if self.plan_log_id is not None else st.st_size
        if st.st_size <= self.plan_log_pos:
            return
        with open(PLANNER_LOG, errors='replace') as fh:
            fh.seek(self.plan_log_pos)
            data = fh.read()
            self.plan_log_pos = fh.tell()
        for ln in data.splitlines():
            if PLAN_KEEP.search(ln) and not SKIP.search(ln):
                self.w('PLANLOG', re.sub(r'^\[(\w+)\] \[\d+\.\d+\] \[llm_planner\]: ', r'[\1] ', ln)[:360])

    def periodic(self):
        objs = []
        for oid, o in sorted((self.scene or {}).items()):
            if isinstance(o, dict) and o.get('label') in ('cup', 'mouse', 'bottle', 'bowl') and not o.get('stale'):
                objs.append(f"{oid}:{o.get('color')}@({o['x']:.2f},{o['y']:.2f}){'' if o.get('reachable') else '!unreach'}")
        pairs = {k: f"{v['median_m'] * 100:.1f}cm" for k, v in (self.agree or {}).get('pairs', {}).items()}
        self.w('SCENE', ' '.join(objs) or '(no objects)')
        if pairs:
            self.w('SCENE', f'camera agreement {pairs}')
        try:
            issues = json.loads(urllib.request.urlopen('http://localhost:8000/api/health/issues', timeout=5).read())['issues']
            ids = sorted(i['id'] for i in issues)
            if ids != self.health_ids:
                self.health_ids = ids
                self.w('HEALTH', ', '.join(f"{i['id']} ({i['title']})" for i in issues) or 'no issues')
        except Exception as e:
            self.w('HEALTH', f'backend not answering: {e}')

    first_pass = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=60.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    out = open(a.out, 'a')
    node = Logger(out)
    node.w('START', f'session logger for {a.minutes:.0f} min; listens only')
    node.tail_outcomes()
    node.first_pass = False                                   # from now on files that appear later are read from their start
    node.tail_arm_log()
    node.tail_planner_log()
    stop = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__('now', True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__('now', True))
    t_end, t_per = time.time() + a.minutes * 60.0, 0.0
    while time.time() < t_end and not stop['now'] and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.25)
        node.tail_outcomes()
        node.tail_arm_log()
        node.tail_planner_log()
        if time.time() - t_per > 30.0:
            t_per = time.time()
            node.periodic()
    node.w('END', 'session logger stopped')
    out.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
