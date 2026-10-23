import asyncio
import os
import re
import subprocess
import threading
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config, health_issues, log_relay, process_manager, ros_bridge

router = APIRouter()

DISABLED_FILE = os.environ.get('SHASHPROJECT_DISABLED_CAMERAS', os.path.join(config.REPO_ROOT, 'config', 'disabled_cameras.txt'))
CAMERA_LABELS = {'realsense': 'RealSense 1', 'realsense2': 'RealSense 2', 'realsense3': 'RealSense 3', 'wrist': 'Wrist camera', 'oakd': 'OAK-D'}
ARM_IFACE = os.environ.get('SHASHPROJECT_ARM_IFACE', 'enp2s0')


def _disabled():
    try:
        with open(DISABLED_FILE) as f:
            return {ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')}
    except OSError:
        return set()


def _arm_link_up():
    try:
        with open(f'/sys/class/net/{ARM_IFACE}/operstate') as f:
            return f.read().strip() == 'up'
    except OSError:
        return None


def collect():
    bridge = ros_bridge.get_bridge()
    facts = bridge.get_health_facts()
    processes, _holders = process_manager.manager.status_bundle()
    robot = (processes.get('robot_bringup') or {}).get('status')
    facts['robot_expected'] = bool(robot) and robot not in ('stopped', 'running_other_user')
    disabled = _disabled()
    cams = {}
    for name in config.CAMERA_TOPICS:
        if name in disabled or name == 'oakd':
            continue
        jpeg, stale = bridge.get_jpeg(name)
        cams[name] = {'has_frame': jpeg is not None, 'stale': stale, 'label': CAMERA_LABELS.get(name, name), 'age_s': None}
    facts['cameras'] = cams
    log = log_relay.tail_file(os.path.join(config.LOG_DIR, 'robot_bringup.log'), lines=400, max_bytes=256 * 1024)
    facts['log_hits'] = health_issues.log_hits(log.splitlines(), time.time())
    facts['arm_link_up'] = _arm_link_up()
    return facts


@router.get('/health/issues')
async def health_issues_endpoint():
    """What is wrong right now, with what to do about it (see health_issues.py). The browser also treats a failed/slow answer from this very call as 'the backend is not responding'."""
    facts = await asyncio.to_thread(collect)
    return {'issues': health_issues.evaluate(facts), 'time': time.time()}


# ── the buttons next to the issues ────────────────────────────────────────────────
_fix_lock = threading.Lock()


def _ros(cmd, timeout=40):
    """Run a ROS 2 CLI command in the same environment as the backend's nodes; (returncode, output)."""
    r = subprocess.run(['bash', '-c', f'{config.ROS_ENV_CMD} {cmd}'], capture_output=True, text=True, timeout=timeout)
    return r.returncode, re.sub(r'\x1b\[[0-9;]*m', '', (r.stdout + r.stderr)).strip()


def fix_controllers():
    """Configure / activate the controllers whose job is only to publish state (joint_state_broadcaster, fault_controller) if they are stuck. When the controller manager is slow at
    startup its spawner times out and leaves the broadcaster unconfigured: no joint states, no TF (2026-10-22). Never touches a controller that moves the arm."""
    code, out = _ros('ros2 control list_controllers')
    if code != 0:
        return False, f'cannot reach the controller manager (is Robot bringup running?): {out[-200:]}'
    state = {ln.split()[0]: ln.split()[-1] for ln in out.splitlines() if len(ln.split()) >= 2 and ln.split()[0] in ('joint_state_broadcaster', 'fault_controller')}
    done, failed = [], []
    for name in ('joint_state_broadcaster', 'fault_controller'):
        st = state.get(name)
        if st == 'active':
            continue
        if st is None:
            failed.append(f'{name}: not loaded (restart Robot bringup)')
            continue
        steps = (['inactive'] if st == 'unconfigured' else []) + ['active']
        for target in steps:
            c, o = _ros(f'ros2 control set_controller_state {name} {target}')
            if c != 0 or 'Successfully' not in o:
                failed.append(f'{name} -> {target}: {o[-120:]}')
                break
        else:
            done.append(f'{name} activated')
    if failed:
        return False, '; '.join(failed)
    return True, ('; '.join(done) if done else 'both controllers were already active: nothing to fix (use Restart Robot bringup)')


def _after_robot_restart():
    """The spawners can time out while the controller manager is still loading (seen live): after a restart, look again every 15 s for two minutes and fix what is stuck."""
    time.sleep(40)
    for _ in range(8):
        try:
            fix_controllers()
        except Exception:
            pass
        time.sleep(15)


def _restart_robot():
    ok1, m1 = process_manager.manager.stop('robot_bringup')
    time.sleep(4)
    ok2, m2 = process_manager.manager.start('robot_bringup')
    threading.Thread(target=_after_robot_restart, daemon=True).start()
    return ok2, f'{m1} {m2} The controllers are checked automatically over the next two minutes; this banner clears when joint states and TF are back.'


def _restart_cameras():
    process_manager.manager.stop('cameras_bringup')
    time.sleep(5)
    steps = process_manager.manager.start_cameras()
    return all(s['ok'] for s in steps), 'Cameras bringup restarted; images return in about 30 s. If a tile stays black after a minute, restart the dashboard backend.'


def _restart_backend():
    script = os.path.join(config.REPO_ROOT, 'scripts', 'restart_dashboard_backend.sh')
    log = open('/tmp/dashboard_restart.log', 'ab')
    subprocess.Popen(['bash', '-c', f'sleep 1.5; bash {script}'], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    return True, 'Restarting the backend: this page reconnects in about 10 s.'


FIXES = {'restart_robot_bringup': _restart_robot, 'fix_controllers': fix_controllers, 'restart_cameras': _restart_cameras, 'restart_backend': _restart_backend}


class FixRequest(BaseModel):
    action: str


@router.post('/health/fix')
async def health_fix(req: FixRequest):
    """Run one of the fixes the health banner offers (health_issues.ACTIONS). Only these named actions exist; one at a time."""
    if req.action not in FIXES:
        raise HTTPException(404, f'unknown fix {req.action!r}')
    if not _fix_lock.acquire(blocking=False):
        return {'ok': False, 'message': 'another fix is still running; wait for it to finish'}

    def run():
        try:
            return FIXES[req.action]()
        finally:
            _fix_lock.release()
    ok, message = await asyncio.to_thread(run)
    return {'ok': bool(ok), 'message': message}
