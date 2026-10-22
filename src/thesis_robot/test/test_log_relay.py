"""The dashboard's process logs: tail without reading the file, spam sampling, size cap (2026-10-22: a 14 GB log took the backend down)."""
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
from app import log_relay as lr  # noqa: E402


def test_tail_of_a_huge_sparse_file_returns_the_end_without_reading_it(tmp_path):
    p = tmp_path / 'huge.log'
    with open(p, 'wb') as f:
        f.truncate(3 * 1024 ** 3)                                       # 3 GB apparent size, no real data
        f.seek(0, os.SEEK_END)
        f.write(b'\n'.join(f'line {i}'.encode() for i in range(500)) + b'\n')
    t0 = time.time()
    out = lr.tail_file(str(p), lines=200)
    assert time.time() - t0 < 2.0
    got = out.splitlines()
    assert got[-1] == 'line 499' and len(got) == 200


def test_tail_of_a_small_or_missing_file(tmp_path):
    assert lr.tail_file(str(tmp_path / 'nope.log')) == ''
    p = tmp_path / 'a.log'
    p.write_text('one\ntwo\nthree\n')
    assert lr.tail_file(str(p), lines=2) == 'two\nthree\n'


def run_relay(lines, tmp_path, **kw):
    path = str(tmp_path / 'out.log')
    r = lr.LogRelay(io.BytesIO(b''.join(lines)), path, **kw)
    r.run()
    return path, r


def test_spam_lines_are_sampled_and_counted_and_normal_lines_are_kept(tmp_path):
    spam = b'[DEBUG] [1.2] [KortexMultiInterfaceHardware]: No controller active in LOW_LEVEL_SERVOING mode !\n'
    lines = [b'start\n'] + [spam] * 1000 + [b'real warning\n']
    path, r = run_relay(lines, tmp_path)
    text = open(path, 'rb').read().decode()
    assert text.count('No controller active') == 1                      # one sample inside the 30 s window
    assert 'start' in text and 'real warning' in text
    assert r.dropped == 999


def test_the_dropped_count_is_written_with_the_next_sample(tmp_path):
    spam = b'xioctl(VIDIOC_S_FMT) failed, errno=16\n'
    path, r = run_relay([spam] * 50 + [spam], tmp_path, spam_period_s=0.0)         # a zero window: every line is a new sample
    assert open(path, 'rb').read().decode().count('xioctl') == 51


def test_the_log_is_capped_and_rotated(tmp_path):
    lines = [f'line {i:06d} ' .encode() + b'x' * 80 + b'\n' for i in range(5000)]       # about 450 KB
    path, r = run_relay(lines, tmp_path, max_bytes=100_000)
    assert os.path.getsize(path) <= 100_000 + 200
    assert os.path.exists(path + '.1') and os.path.getsize(path + '.1') > 90_000
    assert b'line 004999' in open(path, 'rb').read()                      # the newest lines are in the live file
