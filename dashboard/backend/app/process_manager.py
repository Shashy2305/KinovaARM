"""
Starts, stops, and reports status for every process config.PROCESSES knows
about. The one property this module exists to guarantee: never start a
second copy of something that's already running, whether the dashboard
started the first copy or a person did by hand in a terminal — that
exact failure mode (two OAK-D drivers fighting over one USB device, an
old RealSense TF publisher left running alongside a new one) is what cost
the most debugging time during development.
"""
import os
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
        snapshot = {}
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                snapshot[p.info['pid']] = ' '.join(p.info['cmdline'] or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return snapshot

    def _matching_pids(self, signature, cmdlines):
        """PIDs (from a pre-fetched {pid: cmdline} snapshot) whose command
        line contains `signature`."""
        if not signature:
            return []
        return [pid for pid, cmdline in cmdlines.items() if signature in cmdline]

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
        external_pids = [
            pid for pid in self._matching_pids(cfg['signature'], cmdlines)
            if not (owned_alive and pid == mp.popen.pid)
        ]

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
        return 'stopped'

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
            pid: {**{k: v for k, v in cfg.items() if k != 'cmd'}, 'status': self.status(pid, cmdlines)}
            for pid, cfg in config.PROCESSES.items()
        }

    # ── control ──────────────────────────────────────────────────────
    def start(self, proc_id):
        cfg = config.PROCESSES[proc_id]
        current = self.status(proc_id)
        if current in ('running', 'running_external'):
            return False, (
                f'{cfg["label"]} is already running'
                f'{" (started outside the dashboard)" if current == "running_external" else ""}'
                f' — not starting a second copy.')
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
        for pid, cmdline in self._snapshot_cmdlines().items():
            if cfg['signature'] and cfg['signature'] in cmdline:
                # Never stop a shell (an operator's `bash -c "... grep <name> ..."`
                # contains the signature as plain text) or this backend or
                # anything that launched it. The real process still matches.
                if pid in protected or self._is_shell(cmdline):
                    continue
                if self._terminate_tree(pid):
                    stopped_any = True

        if not stopped_any:
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
        ok, msg = self.start('oakd_driver')
        steps.append({'step': 'start this repo\'s OAK-D driver (640x400)', 'ok': ok, 'message': msg})

        return steps


manager = ProcessManager()
