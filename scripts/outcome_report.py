#!/usr/bin/env python3
"""Print the outcome-log summary:  python3 scripts/outcome_report.py [--all] [--recent N]

--all     include dry-run records as well (default: live runs only)
--recent  also list the last N failed picks/places with the reason
--csv     write one row per pick/place attempt to a CSV file (for a spreadsheet or a notebook)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'thesis_robot'))
from thesis_robot import outcome_log, outcome_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--all', action='store_true')
ap.add_argument('--recent', type=int, default=0)
ap.add_argument('--csv', default='')
ap.add_argument('--dir', default=outcome_log.DEFAULT_DIR)
args = ap.parse_args()
recs = outcome_log.read_records(args.dir)
print(f'{len(recs)} records in {args.dir}\n')
print(outcome_report.format_summary(outcome_report.summarize(recs, live_only=not args.all)))
if args.recent:
    print('\nRecent failures')
    bad = [r for r in recs if r.get('kind') in ('pick', 'place') and not r.get('ok') and (args.all or r.get('live', True))]
    for r in bad[-args.recent:]:
        print(f"  {r.get('iso')} {r['kind']} {r.get('label', '')} failed at {r.get('failed_step')}: {r.get('reason')}")

if args.csv:
    import csv
    rows = []
    for r in recs:
        if r.get('kind') not in ('pick', 'place') or not (args.all or r.get('live', True)):
            continue
        ev = {e.get('what'): e for e in (r.get('events') or [])}
        shape, cand = ev.get('shape', {}), ev.get('grasp_candidates', {})
        near = (r.get('neighbours') or [{}])[0]
        rows.append({
            'time': r.get('iso'), 'kind': r['kind'], 'label': r.get('label'), 'attempt': r.get('attempt'), 'ok': int(bool(r.get('ok'))),
            'failed_step': r.get('failed_step'), 'reason': r.get('reason'), 'strategy': r.get('strategy'), 'seconds': r.get('seconds'),
            'nearest_label': near.get('label'), 'nearest_m': near.get('dist_m'),
            'rho_deg': shape.get('rho'), 'width_mm': shape.get('width_mm'), 'table_width_mm': shape.get('table_width_mm'),
            'shape_class': shape.get('cls'), 'edge_support': shape.get('edge_support'),
            'grip_finger': (r.get('grip') or [{}])[-1].get('finger') if r.get('grip') else None,
            'candidates_offered': cand.get('n'), 'tilt_deg': (ev.get('tilt') or {}).get('tilt'),
            'unknown_near': len(r.get('unknown_near') or []), 'live': int(bool(r.get('live', True))),
        })
    with open(os.path.expanduser(args.csv), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['time'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows to {args.csv}')
