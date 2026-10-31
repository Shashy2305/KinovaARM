#!/usr/bin/env python3
"""
Drive the dashboard jog with a 3Dconnexion SpaceMouse (Bluetooth or USB) WITHOUT touching robot bringup: this is a separate process that reads the mouse (/dev/input/event*) and sends the same
Cartesian jog commands the dashboard's Jog tab sends, so the jog's own protection applies to every command (table and whole-arm guards, joint limits, singularity check, 0.2 s of motion per command).

    /mnt/ros_workspace/venv/bin/python3 scripts/spacemouse_jog.py --probe [--save]   # guided gestures: learns which device axis is which, writes ~/.ros/spacemouse_map.json
    /mnt/ros_workspace/venv/bin/python3 scripts/spacemouse_jog.py                    # DRY RUN (default): prints what it WOULD send, sends nothing
    /mnt/ros_workspace/venv/bin/python3 scripts/spacemouse_jog.py --live             # sends. Needs: Jog ON in the dashboard (this script never switches it on) and a dead-man button HELD

Safety: nothing is sent unless --live, the dead-man button (either button by default) is held and the jog is on. Letting go of the button, a Bluetooth drop or Ctrl-C stops the commands, and the jog
stops the arm 0.2 s later. Translation OR rotation, never both. Speed is --speed (default 0.25 of the jog's 8 cm/s = 2 cm/s at full push) times the deflection. The hardware E-stop stays the real stop.
"""
import argparse
import fcntl
import json
import os
import select
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dashboard', 'backend'))
from app import spacemouse_map as sm  # noqa: E402

EV_KEY, EV_ABS = 1, 3
EVENT_SIZE = struct.calcsize('llHHi')
TICK_S = 0.0625                  # the jog drops commands closer than 0.06 s
FULL_SCALE_DEFAULT = 350.0

GESTURES = (      # (jog axis, what to do), described for a person standing at the robot base looking along +x (the direction of the table)
    ('x', 'PUSH the cap AWAY from you (toward the table)'),
    ('y', 'PUSH the cap to your RIGHT'),
    ('z', 'PULL the cap UP'),
    ('rx', 'TILT the cap so its RIGHT side goes UP (lean it to your left)'),
    ('ry', 'TILT the cap AWAY from you (nose down, top toward the table)'),
    ('rz', 'TWIST the cap COUNTER-clockwise as seen from above'),
)


def find_device(name='SpaceMouse'):
    base = '/sys/class/input'
    for e in sorted(os.listdir(base)):
        if not e.startswith('event'):
            continue
        try:
            with open(f'{base}/{e}/device/name') as f:
                if name.lower() in f.read().lower():
                    return f'/dev/input/{e}'
        except OSError:
            continue
    return None


class Device:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.raw = [0] * 6
        self.buttons = {}
        self.full = [FULL_SCALE_DEFAULT] * 6
        for a in range(6):
            try:
                info = fcntl.ioctl(self.fd, 0x80184540 + a, b'\0' * 24)            # EVIOCGABS: value, min, max, fuzz, flat, resolution
                _v, lo, hi = struct.unpack('iii', info[:12])
                if hi > 0:
                    self.full[a] = float(max(abs(lo), abs(hi)))
            except OSError:
                pass

    def poll(self, timeout=0.0):
        """Read what is pending; returns True when something changed. Raises OSError when the device is gone."""
        changed = False
        if select.select([self.fd], [], [], timeout)[0]:
            data = os.read(self.fd, EVENT_SIZE * 64)
            for i in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _s, _u, typ, code, val = struct.unpack('llHHi', data[i:i + EVENT_SIZE])
                if typ == EV_ABS and code < 6:
                    self.raw[code] = val
                    changed = True
                elif typ == EV_KEY:
                    self.buttons[code] = bool(val)
                    changed = True
        return changed

    def axes(self):
        return [sm.normalise(self.raw[i], self.full[i]) for i in range(6)]

    def close(self):
        os.close(self.fd)


def probe(dev, save):
    print('Guided probe: for each gesture, push/tilt the cap firmly for about 3 s when I say GO, then let it go. Stand at the robot base looking toward the table.')
    results = {}
    for name, text in GESTURES:
        input(f'\n[{name}] {text}\n  press Enter, then do it for 3 s... ')
        while dev.poll(0.05):
            pass
        peaks = [0.0] * 6
        sums = [0.0] * 6
        t0 = time.time()
        print('  GO')
        while time.time() - t0 < 3.0:
            dev.poll(0.05)
            ax = dev.axes()
            for i in range(6):
                if abs(ax[i]) > abs(peaks[i]):
                    peaks[i] = ax[i]
        order = sorted(range(6), key=lambda i: -abs(peaks[i]))
        best, second = order[0], order[1]
        clear = abs(peaks[best]) >= 0.25 and abs(peaks[best]) >= 1.6 * abs(peaks[second])
        print('  peaks per device axis:', [round(p, 2) for p in peaks], '->', f'device axis {best} sign {1 if peaks[best] > 0 else -1}' if clear else 'NOT CLEAR (too gentle, or two axes at once)')
        results[name] = (best, 1 if peaks[best] > 0 else -1) if clear else None
    try:
        mapping = sm.mapping_from_probe(results)
    except ValueError as e:
        print('\nprobe failed:', e, '- run it again, pushing one gesture at a time and a little harder.')
        return 1
    print('\nmapping:', json.dumps({k: list(v) for k, v in mapping.items()}))
    if save:
        sm.save_map(mapping)
        print('saved', sm.MAP_FILE)
    else:
        print('(not saved: add --save)')
    return 0


def run(args):
    dev_path = args.device or find_device()
    if not dev_path:
        print('no SpaceMouse found (is it connected? bluetoothctl devices Connected)')
        return 2
    try:
        dev = Device(dev_path)
    except OSError as e:
        print(f'cannot open {dev_path}: {e}')
        return 2
    if args.probe:
        return probe(dev, args.save)
    mapping = sm.load_map()
    live = bool(args.live)
    sock = None
    if live:
        if args.deadman == 'none':
            print('--live needs a dead-man button (--deadman any|left|right)')
            return 2
        from websockets.sync.client import connect
        sock = connect(args.url, open_timeout=5)
    print(f'SpaceMouse {dev_path}  mode: {"LIVE" if live else "DRY RUN (nothing is sent)"}  dead-man: {args.deadman}  speed {args.speed}  frame {args.frame}  map {mapping}')
    print('axis order printed: x y z rx ry rz.  Hold the dead-man button and move the cap.' + ('' if live else '  (--live to send)') + '  Ctrl-C to stop.')
    t_end = time.time() + args.seconds if args.seconds else None
    last_print, last_note, sent = 0.0, '', 0
    try:
        while t_end is None or time.time() < t_end:
            dev.poll(TICK_S)
            ax = dev.axes()
            held = args.deadman == 'none' or sm.deadman_ok(dev.buttons, args.deadman)
            v = sm.to_jog_vector(ax, mapping, args.deadzone, args.expo, args.dominant)
            moving = held and any(v)
            now = time.time()
            if moving and live:
                sock.send(json.dumps({'mode': 'cart', 'v': v, 'speed': args.speed, 'frame': args.frame}))
                try:
                    st = json.loads(sock.recv(timeout=1.0))
                except Exception:
                    st = {}
                sent += 1
                note = st.get('blocked') or ''
                if note and note != last_note:
                    print('  jog says:', note)
                last_note = note
            if now - last_print > 0.15 and (moving or any(abs(a) > 0.05 for a in ax)):
                last_print = now
                print('raw', [round(a, 2) for a in ax], 'buttons', {hex(k): v for k, v in dev.buttons.items() if v} or '-', '->', 'v', [round(c, 2) for c in v],
                      ('SENT' if live else 'would send') if moving else ('(dead-man not held)' if not held else '(inside dead zone)'))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print('device lost:', e)
    finally:
        if sock is not None:
            sock.close()
        dev.close()
    print(f'done. {sent} commands sent.' if live else 'done (dry run, nothing was sent).')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default=None, help='/dev/input/eventN (default: find "SpaceMouse")')
    p.add_argument('--probe', action='store_true', help='guided gestures to learn the axis mapping')
    p.add_argument('--save', action='store_true', help='with --probe: write ~/.ros/spacemouse_map.json')
    p.add_argument('--live', action='store_true', help='send commands (default: dry run)')
    p.add_argument('--url', default='ws://localhost:8000/api/ws/jog')
    p.add_argument('--speed', type=float, default=0.25, help='0..1 of the jog speed (default 0.25 = 2 cm/s at full push)')
    p.add_argument('--frame', choices=('base', 'tool'), default='base')
    p.add_argument('--deadman', choices=('any', 'left', 'right', 'none'), default='any', help='button that must be held to send (default: either)')
    p.add_argument('--deadzone', type=float, default=0.12)
    p.add_argument('--expo', type=float, default=2.0)
    p.add_argument('--dominant', action='store_true', help='only the single strongest axis moves (use for the first live tests)')
    p.add_argument('--seconds', type=float, default=0.0, help='stop after this long (0 = until Ctrl-C)')
    args = p.parse_args()
    if not 0.0 < args.speed <= 0.6:
        p.error('--speed must be in (0, 0.6]')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
