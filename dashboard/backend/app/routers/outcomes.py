"""Read-only view of the outcome log (~/.ros/outcomes) and the self-labelled training samples, for the dashboard's Outcomes panel."""
import os
import sys

from fastapi import APIRouter

from .. import config

sys.path.insert(0, os.path.join(config.REPO_ROOT, 'src', 'thesis_robot'))
from thesis_robot import outcome_log, outcome_report, training_samples  # noqa: E402

router = APIRouter()


@router.get('/outcomes/summary')
def outcomes_summary(live_only: bool = True, recent: int = 6):
    records = outcome_log.read_records()
    summary = outcome_report.summarize(records, live_only=live_only)
    bad = [r for r in records if r.get('kind') in ('pick', 'place') and not r.get('ok') and (not live_only or r.get('live', True))]
    return {
        'n_records': len(records),
        'summary': {k: v for k, v in summary.items() if k != 'place_failures'},
        'recent_failures': [{'time': r.get('iso'), 'kind': r['kind'], 'label': r.get('label'), 'step': r.get('failed_step'),
                             'reason': r.get('reason')} for r in bad[-max(0, recent):]][::-1],
        'training_samples': len(training_samples.list_samples(training_samples.DEFAULT_DIR)),
    }
