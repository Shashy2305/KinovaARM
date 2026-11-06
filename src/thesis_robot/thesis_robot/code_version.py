"""Which code and which settings produced a record (2026-11-05).

Every new outcome record (pick, place, command) carries a small `code` dict:

    {'git': '6162af6', 'dirty': False, 'branch': 'main', 'cfg': 'a1b2c3d4'}

    git     short hash of the commit the repository was at when the record was written
    dirty   True when tracked files had uncommitted changes (the record then does not match the hash exactly)
    branch  the branch
    cfg     first 8 hex digits of the SHA-1 of the settings files that change behaviour without a commit
            (~/.ros/coffee.yaml, ~/.ros/locations.yaml, ~/.ros/fixed_obstacles.yaml)

The dict is cached for CACHE_S seconds, so logging stays cheap. When git is missing or the code is not in a repository it says 'unknown'.
"""
import hashlib
import os
import subprocess
import time

CACHE_S = 30.0
CONFIG_FILES = ('~/.ros/coffee.yaml', '~/.ros/locations.yaml', '~/.ros/fixed_obstacles.yaml')
_cache = {'t': 0.0, 'value': None}


def repo_root(start=None):
    """The directory holding .git above this file (symlinks resolved: the installed package links back into the repository), or None."""
    path = os.path.dirname(os.path.realpath(start or __file__))
    for _ in range(8):
        if os.path.exists(os.path.join(path, '.git')):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def _git(root, *args):
    return subprocess.run(['git', '-C', root, *args], capture_output=True, text=True, timeout=5).stdout.strip()


def config_hash(files=CONFIG_FILES):
    h = hashlib.sha1()
    for f in files:
        try:
            with open(os.path.expanduser(f), 'rb') as fh:
                h.update(fh.read())
        except OSError:
            h.update(b'-')
    return h.hexdigest()[:8]


def tag(force=False):
    """The code tag dict (a fresh copy each call)."""
    now = time.monotonic()
    if not force and _cache['value'] is not None and now - _cache['t'] < CACHE_S:
        return dict(_cache['value'])
    value = {'git': 'unknown', 'dirty': None, 'branch': None, 'cfg': config_hash()}
    try:
        root = repo_root()
        if root:
            value['git'] = _git(root, 'rev-parse', '--short', 'HEAD') or 'unknown'
            value['dirty'] = bool(_git(root, 'status', '--porcelain', '--untracked-files=no'))
            value['branch'] = _git(root, 'rev-parse', '--abbrev-ref', 'HEAD') or None
    except Exception:
        pass
    _cache['t'], _cache['value'] = now, value
    return dict(value)
