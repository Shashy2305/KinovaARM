"""
Process logs that cannot fill the disk or the backend's memory.

On 2026-10-22 the dashboard went down in a loop: robot_bringup.log had grown to 14 GB (the driver prints one debug line per millisecond), and the "System log" panel's
request for the last 200 lines made the backend read the whole file into memory (about 15 GB, swap thrashing, every endpoint timing out). cameras_bringup.log had the same
problem with 5 million `xioctl` lines. Two fixes live here:
  * tail_file(): the last N lines by reading only the end of the file, whatever its size;
  * LogRelay: a thread that copies a process's output into its log file, keeps the file under MAX_LOG_BYTES (the previous part is kept as <log>.1),
    and writes at most one sample per SPAM_PERIOD_S of each known spam line, followed by how many were dropped.
"""
import os
import threading
import time

MAX_LOG_BYTES = 40 * 1024 * 1024
SPAM_PERIOD_S = 30.0
SPAM_MARKERS = (b'No controller active in LOW_LEVEL_SERVOING mode', b'xioctl(VIDIOC_S_FMT) failed', b'Stream is PAUSED', b'Failed to start stream',
                b'Trying to connect... (attempt')


def tail_file(path, lines=200, max_bytes=2 * 1024 * 1024):
    """The last `lines` lines of a text file as one string, reading at most `max_bytes` from the end (a partial first line is dropped)."""
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read()
    text = data.decode('utf-8', errors='replace')
    out = text.splitlines(True)
    if start > 0 and out:
        out = out[1:]
    return ''.join(out[-lines:])


class LogRelay(threading.Thread):
    def __init__(self, stream, path, max_bytes=MAX_LOG_BYTES, spam_period_s=SPAM_PERIOD_S):
        super().__init__(daemon=True)
        self.stream, self.path, self.max_bytes, self.spam_period_s = stream, path, max_bytes, spam_period_s
        self.dropped = 0
        self._last = {}
        self._suppressed = {}

    def _spam_key(self, line):
        for m in SPAM_MARKERS:
            if m in line:
                return m
        return None

    def run(self):
        out = open(self.path, 'wb')
        size = 0
        try:
            for line in iter(self.stream.readline, b''):
                key = self._spam_key(line)
                if key is not None:
                    now = time.monotonic()
                    if now - self._last.get(key, -1e9) < self.spam_period_s:
                        self._suppressed[key] = self._suppressed.get(key, 0) + 1
                        self.dropped += 1
                        continue
                    n = self._suppressed.pop(key, 0)
                    self._last[key] = now
                    if n:
                        line = line.rstrip(b'\n') + f'   [+{n} similar lines dropped by the dashboard log relay]\n'.encode()
                out.write(line)
                out.flush()
                size += len(line)
                if size > self.max_bytes:
                    out.close()
                    os.replace(self.path, self.path + '.1')           # the previous part stays for one rotation, the one before it is gone
                    out = open(self.path, 'wb')
                    size = 0
        except (OSError, ValueError):
            pass
        finally:
            try:
                out.close()
            except OSError:
                pass


def spawn_with_log(popen_factory, path, **kw):
    """popen_factory(stdout=PIPE, stderr=STDOUT, **kw) -> Popen; its output is relayed into `path` by a LogRelay thread."""
    import subprocess
    proc = popen_factory(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    LogRelay(proc.stdout, path).start()
    return proc
