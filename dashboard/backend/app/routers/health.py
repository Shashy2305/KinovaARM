import asyncio
import os
import time

from fastapi import APIRouter

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
