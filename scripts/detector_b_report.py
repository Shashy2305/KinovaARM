#!/usr/bin/env python3
"""
detector_b_report.py -- what did the shadow-mode second detector say? Reads ~/.ros/detector_b/ (log_*.jsonl, stats.json), changes nothing.

    python3 scripts/detector_b_report.py [--days 3] [--last 15]

Prints per camera the verdict counts (agree / majority / uncertain / only_b / single), the most frequent A-vs-B-vs-C disagreements, and the latest ones with their crop file. Judge by what
actually happened afterwards (outcome log), not by which detector 'wins': the question is whether disagreement predicts a wrong label.
"""
import argparse
import collections
import glob
import json
import os
import time

DIR = os.path.expanduser('~/.ros/detector_b')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=3)
    ap.add_argument('--last', type=int, default=15)
    a = ap.parse_args()
    try:
        st = json.load(open(os.path.join(DIR, 'stats.json')))
        print(f'stats at {st["time"]} (verified bank: {st.get("bank")} samples)')
        for cam, c in st['stats'].items():
            print(f'  {cam:11s} ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
    except (OSError, ValueError):
        print('no stats.json yet: the node has not run for a minute')
    cutoff = time.time() - a.days * 86400
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, 'log_*.jsonl'))):
        if os.path.getmtime(f) < cutoff:
            continue
        for ln in open(f):
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    pairs = collections.Counter()
    for r in rows:
        for o in r['objects']:
            pairs[(r['camera'], o['verdict'], (o['a'] or {}).get('label'), (o['b'] or {}).get('label'), (o['c'] or {}).get('label'))] += 1
    print(f'\n{len(rows)} logged frames in the last {a.days} days; most frequent (camera, verdict, A, B, C):')
    for k, n in pairs.most_common(12):
        print(f'  {n:4d}  {k}')
    print('\nlatest:')
    for r in rows[-a.last:]:
        o = r['objects'][0]
        print(f'  {r["time"]} {r["camera"]:10s} {o["verdict"]:9s} A={(o["a"] or {}).get("label")} B={(o["b"] or {}).get("label")} C={(o["c"] or {}).get("label")}  crops/{r["crop"]}')


if __name__ == '__main__':
    main()
