#!/usr/bin/env python3
"""Export the outcome logs (~/.ros/outcomes/*.jsonl) to CSV for use on another machine.

    python3 scripts/export_outcomes_csv.py --out /path/to/folder [--live-only] [--since 2026-11-01]

Writes, in --out:
  episodes.csv   one row per COMMAND (an episode): when it started and ended, the command, the model that planned it, ok / failed step, seconds, how many picks and places ran inside it and how many of
                 those succeeded, the plan as JSON, the code tag
  picks.csv      one row per pick attempt: object, position (x, y, z), method (top / side), attempt, heights, scene confidence, nearest neighbour, finger readings at the grip checks, ok / failed step / reason, seconds
  places.csv     one row per place attempt: object, requested target, strategy, carry / set-down heights, finger readings, ok / failed step / reason, seconds
  steps_long.csv time per named step of every pick / place (long format)
  grips_long.csv every finger-joint reading at every grip check (long format; 0 = open, ~0.35 = closed on a cup, ~0.69 = closed on air)
  events_long.csv the notes taken during a pick / place (grasp candidates, centring offsets, touch-downs ...), payload as JSON (long format)
Every row carries episode_id (commands.ts of the episode it belongs to), `live` (False = a dry run: the arm did not move) and the code tag columns code_git, code_dirty, code_cfg. Records written before
2026-11-05 have no code tag (empty columns).
A pick / place belongs to the command whose time window [ts - seconds, ts] contains it (the logs have no explicit link); records outside any command window (single manual picks) get episode_id empty.
"""
import argparse
import csv
import glob
import json
import os
import time
from datetime import datetime


def load(directory):
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(directory), 'outcomes-*.jsonl*'))):
        with open(f) as fh:
            for ln in fh:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
    rows.sort(key=lambda r: r.get('ts', 0))
    return rows


def code_cols(r):
    c = r.get('code') or {}
    return {'code_git': c.get('git'), 'code_dirty': c.get('dirty'), 'code_cfg': c.get('cfg')}


def scalar(v):
    return v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, separators=(',', ':'))


def episode_of(rec, commands):
    """The command whose [ts - seconds, ts] window holds this record (smallest window wins), as its ts; None when none does."""
    best = None
    for c in commands:
        lo, hi = c['ts'] - float(c.get('seconds') or 0) - 1.0, c['ts'] + 1.0
        if bool(c.get('live')) == bool(rec.get('live')) and lo <= rec['ts'] <= hi:
            if best is None or (hi - lo) < best[0]:
                best = (hi - lo, c['ts'])
    return best[1] if best else None


def write(path, rows, first=()):
    keys = list(first)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: scalar(r.get(k)) for k in keys})
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='~/.ros/outcomes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--live-only', action='store_true')
    ap.add_argument('--since', default='', help='YYYY-MM-DD')
    a = ap.parse_args()
    recs = load(a.dir)
    if a.live_only:
        recs = [r for r in recs if r.get('live')]
    if a.since:
        t0 = datetime.strptime(a.since, '%Y-%m-%d').timestamp()
        recs = [r for r in recs if r.get('ts', 0) >= t0]
    os.makedirs(a.out, exist_ok=True)
    commands = [r for r in recs if r.get('kind') == 'command']
    picks = [r for r in recs if r.get('kind') == 'pick']
    places = [r for r in recs if r.get('kind') == 'place']

    pick_rows, place_rows, steps, grips, events = [], [], [], [], []
    for kind, rs, out in (('pick', picks, pick_rows), ('place', places, place_rows)):
        for r in rs:
            ep = episode_of(r, commands)
            row = {'episode_id': ep, 'ts': r.get('ts'), 'iso': r.get('iso'), 'live': bool(r.get('live'))}
            for k, v in r.items():
                if k in ('ts', 'iso', 'live', 'kind', 'steps', 'grip', 'events', 'code', 'neighbours', 'xy', 'target', 'requested'):
                    continue
                row[k] = v
            if r.get('xy'):
                row['x'], row['y'] = r['xy'][0], r['xy'][1]
            if r.get('target') is not None:
                row['target_json'] = json.dumps(r['target'], separators=(',', ':'))
            if r.get('requested') is not None:
                row['requested_json'] = json.dumps(r['requested'], separators=(',', ':'))
            nb = r.get('neighbours') or []
            row['n_neighbours'] = len(nb)
            if nb:
                row['nearest_label'], row['nearest_m'], row['nearest_bearing_deg'] = nb[0].get('label'), nb[0].get('dist_m'), nb[0].get('bearing_deg')
            g = [x.get('finger') for x in (r.get('grip') or []) if x.get('finger') is not None]
            row['n_grip_checks'], row['finger_first'], row['finger_last'] = len(g), (g[0] if g else None), (g[-1] if g else None)
            row.update(code_cols(r))
            out.append(row)
            for name, sec in (r.get('steps') or {}).items():
                steps.append({'episode_id': ep, 'kind': kind, 'ts': r.get('ts'), 'live': bool(r.get('live')), 'step': name, 'seconds': sec})
            for x in (r.get('grip') or []):
                grips.append({'episode_id': ep, 'kind': kind, 'ts': r.get('ts'), 'live': bool(r.get('live')), 'at': x.get('at'), 'finger': x.get('finger')})
            for e in (r.get('events') or []):
                payload = {k: v for k, v in e.items() if k not in ('t', 'what')}
                events.append({'episode_id': ep, 'kind': kind, 'ts': r.get('ts'), 'live': bool(r.get('live')), 't_rel': e.get('t'), 'what': e.get('what'),
                               'payload_json': json.dumps(payload, separators=(',', ':'))})
    ep_rows = []
    for c in commands:
        inside_p = [p for p in pick_rows if p['episode_id'] == c['ts']]
        inside_l = [p for p in place_rows if p['episode_id'] == c['ts']]
        plan = c.get('plan') or []
        ep_rows.append({'episode_id': c['ts'], 'ts_start': round(c['ts'] - float(c.get('seconds') or 0), 3), 'ts_end': c['ts'], 'iso_end': c.get('iso'), 'live': bool(c.get('live')),
                        'command': c.get('command'), 'model': c.get('model'), 'ok': bool(c.get('ok')), 'failed_step': c.get('failed_step'), 'reason': c.get('reason'), 'seconds': c.get('seconds'),
                        'n_steps': len(plan), 'actions': ' '.join(str(s.get('action')) for s in plan if isinstance(s, dict)), 'n_picks': len(inside_p), 'n_picks_ok': sum(bool(p.get('ok')) for p in inside_p),
                        'n_places': len(inside_l), 'n_places_ok': sum(bool(p.get('ok')) for p in inside_l), 'plan_json': json.dumps(plan, separators=(',', ':')), **code_cols(c)})
    n = {'episodes.csv': write(os.path.join(a.out, 'episodes.csv'), ep_rows, ('episode_id', 'ts_start', 'ts_end', 'iso_end', 'live', 'command', 'model', 'ok', 'failed_step', 'reason', 'seconds')),
         'picks.csv': write(os.path.join(a.out, 'picks.csv'), pick_rows, ('episode_id', 'ts', 'iso', 'live', 'object_id', 'label', 'x', 'y', 'z', 'attempt', 'method', 'ok', 'failed_step', 'reason', 'seconds')),
         'places.csv': write(os.path.join(a.out, 'places.csv'), place_rows, ('episode_id', 'ts', 'iso', 'live', 'object_id', 'label', 'strategy', 'ok', 'failed_step', 'reason', 'seconds')),
         'steps_long.csv': write(os.path.join(a.out, 'steps_long.csv'), steps, ('episode_id', 'kind', 'ts', 'live', 'step', 'seconds')),
         'grips_long.csv': write(os.path.join(a.out, 'grips_long.csv'), grips, ('episode_id', 'kind', 'ts', 'live', 'at', 'finger')),
         'events_long.csv': write(os.path.join(a.out, 'events_long.csv'), events, ('episode_id', 'kind', 'ts', 'live', 't_rel', 'what', 'payload_json'))}
    for k, v in n.items():
        print(f'{k:16s} {v:6d} rows')


if __name__ == '__main__':
    main()
