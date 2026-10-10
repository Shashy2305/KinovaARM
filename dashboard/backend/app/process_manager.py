"""
Starts, stops, and reports status for every process config.PROCESSES knows
about. The one property this module exists to guarantee: never start a
second copy of something that's already running, whether the dashboard
started the first copy or a person did by hand in a terminal — that
exact failure mode (two OAK-D drivers fighting over one USB device, an
old RealSense TF publisher left running alongside a new one) is what cost
the most debugging time during development.
"""
import getpass
import os
import re
import signal
import subprocess
import time

import psutil

from . import config

os.makedirs(config.LOG_DIR, exist_ok=True)


class ManagedProcess:
    def __init__(self, proc_id):
        self.proc_id = proc_id
        self.popen = None           # set only if THIS manager started it
        self.log_path = os.path.join(config.LOG_DIR, f'{proc_id}.log')


class ProcessManager:
    def __init__(self):
        self._procs = {pid: ManagedProcess(pid) for pid in config.PROCESSES}
        self._me = getpass.getuser()
        self._owners = {}           # {pid: username} of the last _snapshot_cmdlines() scan

    # ── inspection ───────────────────────────────────────────────────
    def _snapshot_cmdlines(self):
        """One system-wide process scan, {pid: cmdline}. Scanning once and
        matching signatures against this in memory, instead of re-running
        psutil.process_iter() per signature, is the difference between one
        /proc walk and twelve -- with ~600 processes on this shared lab
        machine, twelve walks every 0.5s WS tick was ~250ms of synchronous
        work landing straight on the asyncio event loop thread (nothing
        here awaited it), which is what was stalling/hanging the backend:
        see routers/status.py's ws_status for the other half of the fix."""
        snapshot, owners = {}, {}
        for p in psutil.process_iter(['pid', 'cmdline', 'username']):
            try:
                snapshot[p.info['pid']] = ' '.join(p.info['cmdline'] or [])
                owners[p.info['pid']] = p.info['username']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self._owners = owners
        return snapshot

    def _foreign(self, pid):
        """Username of another account that owns this pid (None: it is ours, or unknown). On this shared lab machine a
        matching process owned by another user is NOT ours to start, stop or count as 'already running for us'."""
        owner = self._owners.get(pid)
        return owner if owner and owner != self._me else None

    def _matching_pids(self, signature, cmdlines):
        """PIDs (from a pre-fetched {pid: cmdline} snapshot) whose command
        line contains `signature`."""
        if not signature:
            return []
        return [pid for pid, cmdline in cmdlines.items()
                if signature in cmdline and not self._is_api_client(cmdline)]

    def status(self, proc_id, cmdlines=None):
        """Returns one of: 'stopped', 'starting', 'running', 'running_external',
        'conflict' (this AND something matching a conflict_signature are
        both alive). Pass a pre-fetched cmdlines snapshot (_snapshot_cmdlines)
        when checking several processes at once -- see all_status()."""
        if cmdlines is None:
            cmdlines = self._snapshot_cmdlines()
        cfg = config.PROCESSES[proc_id]
        mp = self._procs[proc_id]

        owned_alive = mp.popen is not None and mp.popen.poll() is None
        matching = [
            pid for pid in self._matching_pids(cfg['signature'], cmdlines)
            if not (owned_alive and pid == mp.popen.pid)
        ]
        external_pids = [pid for pid in matching if not self._foreign(pid)]
        foreign_pids = [pid for pid in matching if self._foreign(pid)]

        for other_sig in cfg.get('conflict_signatures', []):
            if other_sig == cfg['signature']:
                continue
            conflict_pids = [
                pid for pid in self._matching_pids(other_sig, cmdlines)
                if pid not in external_pids
                # A conflict_signature that's a substring of this process's
                # own signature (e.g. 'oak_camera_node.py' inside
                # 'drivers/oak_camera_node.py') matches our OWN owned
                # process too -- exclude it the same way external_pids
                # already does, or a lone correctly-running driver falsely
                # reports as conflicting with itself.
                and not (owned_alive and pid == mp.popen.pid)
            ]
            if conflict_pids and (owned_alive or external_pids):
                return 'conflict'

        if owned_alive:
            return 'running'
        if external_pids:
            return 'running_external'
        if foreign_pids:
            return 'running_other_user'
        return 'stopped'

    def other_users(self, proc_id, cmdlines):
        """Sorted usernames of OTHER accounts running something that matches this process's signature."""
        sig = config.PROCESSES[proc_id]['signature']
        return sorted({self._foreign(pid) for pid in self._matching_pids(sig, cmdlines)} - {None})

    # Hardware another account may be holding: what to look for in their command lines, and what it means for us.
    _HARDWARE_PATTERNS = (
        ('ros2_control_node', 'the arm driver (only one program can drive the arm)'),
        ('kinova_vision_node', 'the wrist camera stream (the arm serves one RTSP client)'),
        ('realsense2_camera_node', 'a RealSense camera'),
    )

    def hardware_holders(self, cmdlines=None):
        """Processes of OTHER users that occupy robot hardware we need: [{user, pid, what, detail}]. A RealSense node is
        matched to its camera through the serial in its launch command when that is visible."""
        if cmdlines is None:
            cmdlines = self._snapshot_cmdlines()
        serial_of = {v: k for k, v in config.CAMERA_SERIALS.items()}
        out, seen = [], set()
        for pid, cmd in cmdlines.items():
            user = self._foreign(pid)
            if not user or self._is_shell(cmd) or self._is_api_client(cmd) or self._is_multiplexer(cmd):
                continue
            for needle, what in self._HARDWARE_PATTERNS:
                if needle in cmd:
                    key = (user, needle)
                    if key in seen:
                        break
                    seen.add(key)
                    out.append({'user': user, 'pid': pid, 'what': what, 'detail': needle})
                    break
            else:
                m = re.search(r'serial_no:=_?(\d{9,})', cmd)
                if m and 'realsense' in cmd and (user, m.group(1)) not in seen:
                    seen.add((user, m.group(1)))
                    out.append({'user': user, 'pid': pid, 'what': f'RealSense {serial_of.get(m.group(1), m.group(1))}',
                                'detail': f'serial {m.group(1)}'})
        # one RealSense node per launch is enough: drop the generic row when a serial row names the same user
        named = {h['user'] for h in out if h['what'].startswith('RealSense ') and h['what'] != 'a RealSense camera'}
        return [h for h in out if not (h['what'] == 'a RealSense camera' and h['user'] in named)]

    def all_status(self):
        """Full info (label/category/etc, not just the bare status string)
        for every tracked process -- the single shape both the REST /nodes
        endpoint and the WS status feed serve, so the frontend never has to
        special-case which one it got its process data from (that mismatch
        was a real bug during development: the WS feed used to send only
        {proc_id: status_string}, and every component reading it via the WS
        silently rendered nothing, needing a full page inspection to find)."""
        cmdlines = self._snapshot_cmdlines()
        return {
            pid: {**{k: v for k, v in cfg.items() if k != 'cmd'}, 'status': self.status(pid, cmdlines),
                  'other_users': self.other_users(pid, cmdlines)}
            for pid, cfg in config.PROCESSES.items()
        }

    def status_bundle(self):
        """(all_status, hardware_holders) from ONE process scan; the 0.5 s WebSocket tick must not walk /proc twice."""
        cmdlines = self._snapshot_cmdlines()
        procs = {
            pid: {**{k: v for k, v in cfg.items() if k != 'cmd'}, 'status': self.status(pid, cmdlines),
                  'other_users': self.other_users(pid, cmdlines)}
            for pid, cfg in config.PROCESSES.items()
        }
        return procs, self.hardware_holders(cmdlines)

    # ── control ──────────────────────────────────────────────────────
    def start(self, proc_id):
        cfg = config.PROCESSES[proc_id]
        current = self.status(proc_id)
        if current in ('running', 'running_external'):
            return False, (
                f'{cfg["label"]} is already running'
                f'{" (started outside the dashboard)" if current == "running_external" else ""}'
                f' — not starting a second copy.')
        if current == 'running_other_user':
            who = ', '.join(self.other_users(proc_id, self._snapshot_cmdlines()))
            return False, (
                f'{cfg["label"]} is being run by the user {who}, not by you. Starting a second copy would fight over the '
                f'same hardware. Ask {who} to stop it, then start yours.')
        if current == 'conflict':
            return False, (
                f'{cfg["label"]} has a conflicting process already running '
                f'(matches one of its conflict_signatures) — stop that first.')

        if cfg['cmd'] is None:
            return False, f'{cfg["label"]} has no direct start (composite action).'

        mp = self._procs[proc_id]
        log_f = open(mp.log_path, 'w')
        mp.popen = subprocess.Popen(
            ['bash', '-c', cfg['cmd']],
            stdout=log_f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        return True, f'Started {cfg["label"]} (pid {mp.popen.pid}).'

    _SHELLS = {'bash', 'sh', 'dash', 'zsh', 'fish'}
    _API_CLIENTS = {'curl', 'wget', 'http', 'httpie'}

    @classmethod
    def _is_api_client(cls, cmdline):
        """`curl .../api/nodes/<id>/stop` contains the node's signature as plain
        text; it is a request to this backend, not the node, and must never match
        (it made Start report 'already running' and let Stop kill its own caller)."""
        first = cmdline.split(None, 1)[0] if cmdline else ''
        return os.path.basename(first) in cls._API_CLIENTS or '/api/nodes/' in cmdline

    @classmethod
    def _is_multiplexer(cls, cmdline):
        """A tmux/screen server carries the command it was started with on its own command line; it is not a hardware user."""
        first = cmdline.split(None, 1)[0] if cmdline else ''
        return os.path.basename(first) in ('tmux', 'screen')

    @classmethod
    def _is_shell(cls, cmdline):
        first = cmdline.split(None, 1)[0] if cmdline else ''
        return os.path.basename(first) in cls._SHELLS

    @staticmethod
    def _protected_pids():
        try:
            me = psutil.Process()
            return {me.pid} | {p.pid for p in me.parents()}
        except psutil.Error:
            return {os.getpid()}

    @staticmethod
    def _terminate_tree(pid, timeout=8.0):
        """SIGTERM the process and every descendant, then SIGKILL whatever is
        still alive after `timeout`. Signalling only the parent (what this
        used to do) kills `ros2 launch` but orphans its children -- a stale
        move_group / robot_state_publisher kept running after every
        robot_bringup Stop and then competed with the next Start's copy."""
        try:
            root = psutil.Process(pid)
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return False
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(procs, timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        return True

    def stop(self, proc_id):
        cfg = config.PROCESSES[proc_id]
        mp = self._procs[proc_id]
        stopped_any = False

        if mp.popen is not None and mp.popen.poll() is None:
            if self._terminate_tree(mp.popen.pid):
                stopped_any = True
            mp.popen = None

        protected = self._protected_pids()
        not_ours = set()
        for pid, cmdline in self._snapshot_cmdlines().items():
            if cfg['signature'] and cfg['signature'] in cmdline and not self._is_api_client(cmdline):
                if self._foreign(pid):
                    if not self._is_shell(cmdline):
                        not_ours.add(self._foreign(pid))
                    continue
                # Never stop a shell (an operator's `bash -c "... grep <name> ..."`
                # contains the signature as plain text) or this backend or
                # anything that launched it. The real process still matches.
                if pid in protected or self._is_shell(cmdline):
                    continue
                if self._terminate_tree(pid):
                    stopped_any = True

        if not stopped_any:
            if not_ours:
                who = ', '.join(sorted(not_ours))
                return False, (f'{cfg["label"]} is being run by the user {who}, so this dashboard may not stop it. '
                               f'Ask {who} to stop it (nothing of yours was running).')
            return False, f'{cfg["label"]} was not running.'
        return True, f'Stopped {cfg["label"]}.'

    def tail_log(self, proc_id, lines=200):
        mp = self._procs[proc_id]
        if not os.path.exists(mp.log_path):
            return ''
        with open(mp.log_path, 'r', errors='replace') as f:
            content = f.readlines()
        return ''.join(content[-lines:])

    # ── composite: cameras bring-up ─────────────────────────────────
    def start_cameras(self):
        """Encodes the exact manual cleanup sequence developed today:
        launch cameras.launch.py (which brings up RealSense + a stale
        external OAK-D copy + an old RealSense TF publisher by default),
        then kill both of those known-bad stragglers and start this
        repo's own OAK-D driver in their place."""
        steps = []

        cameras_cfg = config.PROCESSES['cameras_bringup']
        current = self.status('cameras_bringup')
        if current in ('running', 'running_external'):
            steps.append({'step': 'cameras.launch.py', 'ok': False,
                           'message': 'already running — not relaunching.'})
        else:
            cmd = (
                f'{config.ROS_ENV_CMD} cd {config.WORKSPACE_ROOT} && '
                f'ros2 launch kinova_gen3_7dof_robotiq_2f_140_moveit_config '
                f'cameras.launch.py robot_ip:={config.ROBOT_IP}'
            )
            log_f = open(os.path.join(config.LOG_DIR, 'cameras_bringup.log'), 'w')
            popen = subprocess.Popen(
                ['bash', '-c', cmd], stdout=log_f, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid)
            self._procs['cameras_bringup'].popen = popen
            steps.append({'step': 'cameras.launch.py', 'ok': True,
                           'message': f'started (pid {popen.pid}), waiting for it to settle...'})
            time.sleep(6.0)

        # Kill the stale external OAK-D copy (NOT our own, which isn't
        # started yet at this point) and the old RealSense TF publisher.
        cmdlines = self._snapshot_cmdlines()
        killed_oak = 0
        for pid in self._matching_pids('oak_camera_node.py', cmdlines):
            if self._foreign(pid):
                continue                      # another user's driver: not ours to kill
            try:
                os.kill(pid, signal.SIGTERM)
                killed_oak += 1
            except ProcessLookupError:
                pass
        steps.append({'step': 'stop stale external OAK-D driver', 'ok': True,
                       'message': f'stopped {killed_oak} process(es).' if killed_oak
                       else 'none found (already clean).'})

        killed_tf = 0
        for pid in self._matching_pids('calibration_tf_publisher', cmdlines):
            try:
                os.kill(pid, signal.SIGTERM)
                killed_tf += 1
            except ProcessLookupError:
                pass
        steps.append({'step': 'stop stale RealSense TF publisher', 'ok': True,
                       'message': f'stopped {killed_tf} process(es).' if killed_tf
                       else 'none found (already clean).'})

        time.sleep(1.0)
        if 'oakd_driver' in config.disabled_camera_procs():
            steps.append({'step': 'start this repo\'s OAK-D driver (640x400)', 'ok': True,
                          'message': 'skipped: the OAK-D is listed in config/disabled_cameras.txt'})
            return steps
        ok, msg = self.start('oakd_driver')
        steps.append({'step': 'start this repo\'s OAK-D driver (640x400)', 'ok': ok, 'message': msg})

        return steps


manager = ProcessManager()
