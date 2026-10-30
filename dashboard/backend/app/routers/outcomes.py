"""Read-only view of the outcome log (~/.ros/outcomes) and the self-labelled training samples, for the dashboard's Outcomes panel."""
import os
import sys

from fastapi import APIRouter

from .. import config

sys.path.insert(0, os.path.join(config.REPO_ROOT, 'src', 'thesis_robot'))
from thesis_robot import grasp_options, outcome_log, outcome_report, training_samples  # noqa: E402

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


# The table area the map covers (base_link, metres): x forward, y sideways. The planner's workspace is a bit wider; this is where picks happen.
MAP_X = (0.10, 0.66)
MAP_Y = (-0.50, 0.50)
MAP_STEP = 0.05


@router.get('/outcomes/learning')
def outcomes_learning(label: str = 'cup'):
    """What the position-aware grasp learning (grasp_options.local_rate / side_allowed) believes for this class: a grid of local success rates per method over the table,
    the side grasp's allowed region (seed zone + trust region), and every past pick as a dot. Read-only; the same numbers the arm controller logs as "learned here"."""
    records = outcome_log.read_records()
    methods = ('top', 'side', 'lean')
    samples = {m: grasp_options.pick_samples(records, label, m) for m in methods}
    xs = [round(MAP_X[0] + i * MAP_STEP, 3) for i in range(int(round((MAP_X[1] - MAP_X[0]) / MAP_STEP)) + 1)]
    ys = [round(MAP_Y[0] + j * MAP_STEP, 3) for j in range(int(round((MAP_Y[1] - MAP_Y[0]) / MAP_STEP)) + 1)]
    grid = {m: [] for m in methods}
    side_ok = []
    for x in xs:
        rows = {m: [] for m in methods}
        row_ok = []
        for y in ys:
            for m in methods:
                p, n_eff, _s = grasp_options.local_rate(samples[m], (x, y))
                rows[m].append([round(p, 3), round(n_eff, 2)])
            row_ok.append(bool(grasp_options.side_allowed(label, (x, y), records)[0]))
        for m in methods:
            grid[m].append(rows[m])
        side_ok.append(row_ok)
    dots = [{'method': m, 'x': round(x, 3), 'y': round(y, 3), 'ok': ok} for m in methods for x, y, ok in samples[m] if x is not None]
    totals = {m: {'trials': len(samples[m]), 'successes': sum(1 for _x, _y, ok in samples[m] if ok)} for m in methods}
    labels = sorted({r.get('label') for r in records if r.get('kind') == 'pick' and r.get('label') and r.get('method')})
    return {'label': label, 'labels': labels or [label], 'x': xs, 'y': ys, 'grid': grid, 'side_allowed': side_ok, 'dots': dots, 'totals': totals,
            'seed_zone': {'x': list(grasp_options.SIDE_ZONE_X), 'y': list(grasp_options.SIDE_ZONE_Y)},
            'params': {'sigma_m': grasp_options.LEARN_SIGMA_M, 'trust_radius_m': grasp_options.TRUST_RADIUS_M, 'trust_min_rate': grasp_options.TRUST_MIN_RATE,
                       'block_rate': grasp_options.BLOCK_RATE}}
