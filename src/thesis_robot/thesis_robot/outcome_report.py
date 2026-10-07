"""
Summaries of the outcome log (pure Python): success by object, grip strategy and neighbour distance, what fails
and where, typical grip readings, and plain-language "lessons" that a planner prompt or a human can use.
"""
import statistics
from collections import Counter, defaultdict

NEIGHBOUR_BUCKETS = ((0.0, 0.12, 'neighbour < 12 cm'), (0.12, 0.18, 'neighbour 12-18 cm'), (0.18, 9.9, 'clear (>= 18 cm)'))
MIN_SAMPLES_FOR_LESSON = 2


def _rate(ok, n):
    return f'{ok}/{n}' + (f' ({100 * ok / n:.0f}%)' if n else '')


def _bucket(rec):
    near = (rec.get('neighbours') or [{}])[0].get('dist_m')
    if near is None:
        return 'clear (>= 18 cm)'
    for lo, hi, name in NEIGHBOUR_BUCKETS:
        if lo <= near < hi:
            return name
    return NEIGHBOUR_BUCKETS[-1][2]


def summarize(records, live_only=True):
    recs = [r for r in records if not live_only or r.get('live', True)]
    picks = [r for r in recs if r.get('kind') == 'pick']
    places = [r for r in recs if r.get('kind') == 'place']
    commands = [r for r in recs if r.get('kind') == 'command']

    by_label = defaultdict(lambda: [0, 0])
    by_strategy = defaultdict(lambda: [0, 0])
    by_label_strategy = defaultdict(lambda: [0, 0])
    by_bucket = defaultdict(lambda: [0, 0])
    fail_steps = defaultdict(Counter)
    grips = defaultdict(list)
    for r in picks:
        label, strat = r.get('label', '?'), r.get('strategy', 'full_open')
        for d, key in ((by_label, label), (by_strategy, strat), (by_label_strategy, (label, strat)),
                       (by_bucket, _bucket(r))):
            d[key][1] += 1
            d[key][0] += int(bool(r.get('ok')))
        if not r.get('ok'):
            fail_steps[label][r.get('failed_step') or '?'] += 1
        elif r.get('grip'):
            vals = [g['finger'] for g in r['grip'] if g.get('finger') is not None]
            if vals:
                grips[label].append(vals[-1])

    place_fail = Counter(r.get('failed_step') or r.get('reason') or '?' for r in places if not r.get('ok'))
    lessons = []
    for (label, strat), (ok, n) in sorted(by_label_strategy.items()):
        if n >= MIN_SAMPLES_FOR_LESSON and ok == 0:
            lessons.append(f'Picking the {label} with strategy "{strat}" failed {n}/{n} times.')
        elif n >= MIN_SAMPLES_FOR_LESSON and ok / n >= 0.9:
            lessons.append(f'Picking the {label} with strategy "{strat}" works ({ok}/{n}).')
    for name, (ok, n) in sorted(by_bucket.items()):
        if n >= MIN_SAMPLES_FOR_LESSON * 2 and ok / n < 0.5:
            lessons.append(f'Picks with {name} succeed only {ok}/{n}.')
    return {
        'picks': {'n': len(picks), 'ok': sum(1 for r in picks if r.get('ok'))},
        'places': {'n': len(places), 'ok': sum(1 for r in places if r.get('ok'))},
        'commands': {'n': len(commands), 'ok': sum(1 for r in commands if r.get('ok'))},
        'by_label': {k: tuple(v) for k, v in by_label.items()},
        'by_strategy': {k: tuple(v) for k, v in by_strategy.items()},
        'by_neighbour': {k: tuple(v) for k, v in by_bucket.items()},
        'fail_steps': {k: dict(v) for k, v in fail_steps.items()},
        'place_failures': dict(place_fail),
        'grip_median': {k: round(statistics.median(v), 3) for k, v in grips.items()},
        'lessons': lessons,
    }


def format_summary(s):
    lines = [f"picks    {_rate(s['picks']['ok'], s['picks']['n'])}",
             f"places   {_rate(s['places']['ok'], s['places']['n'])}",
             f"commands {_rate(s['commands']['ok'], s['commands']['n'])}", '']
    for title, key in (('Picks by object', 'by_label'), ('Picks by grip strategy', 'by_strategy'),
                       ('Picks by nearest neighbour', 'by_neighbour')):
        lines.append(title)
        for k, (ok, n) in sorted(s[key].items()):
            lines.append(f'  {k:<22} {_rate(ok, n)}')
        lines.append('')
    if s['fail_steps']:
        lines.append('Where picks fail')
        for label, c in sorted(s['fail_steps'].items()):
            lines.append(f"  {label}: " + ', '.join(f'{step} x{n}' for step, n in sorted(c.items(), key=lambda kv: -kv[1])))
        lines.append('')
    if s['grip_median']:
        lines.append('Typical finger reading when the grip held (closed on air reads ~0.69)')
        for label, v in sorted(s['grip_median'].items()):
            lines.append(f'  {label:<12} {v}')
        lines.append('')
    if s['lessons']:
        lines.append('Lessons')
        lines.extend(f'  - {t}' for t in s['lessons'])
    return '\n'.join(lines)
