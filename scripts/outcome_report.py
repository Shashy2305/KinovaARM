#!/usr/bin/env python3
"""Print the outcome-log summary:  python3 scripts/outcome_report.py [--all] [--recent N]

--all     include dry-run records as well (default: live runs only)
--recent  also list the last N failed picks/places with the reason
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'thesis_robot'))
from thesis_robot import outcome_log, outcome_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--all', action='store_true')
ap.add_argument('--recent', type=int, default=0)
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
