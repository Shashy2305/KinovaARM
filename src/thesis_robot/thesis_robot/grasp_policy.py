"""
Grasp policy learned from the outcome log (pure Python, no ROS): which closing direction works on which object.

Every live pick records, in its 'shape' event, the direction the fingers closed along RELATIVE TO THE OBJECT'S LONG AXIS
(rho) and whether the pick ended with the object held (ok). Folded by symmetry (rho and 180-rho are the same grasp on a
symmetric object) into three bins:
    along the length (0-30 deg)   diagonal (30-60 deg)   across the short side (60-90 deg)
For each (object label, bin) we keep successes/trials. prior() turns that into a number in [0, 1] that multiplies the
shape-based score in grasp_planner.rank_grasps:
    prior = mean success (Beta(1,1) prior) + an exploration bonus that shrinks as trials accumulate (UCB-style),
so a direction that has failed repeatedly sinks, one that works rises, and a direction nobody has tried yet is not
ruled out by ignorance. It never overrides the hard constraints (fits between the fingers, no neighbour in the sweep).

This is deliberately small: with tens of trials, a three-bin table per object is what the data can support. It is the
honest version of "the robot learns from its mistakes": measured outcomes bias which of several SAFE grasps it tries.
"""
import math

BINS = ((0.0, 30.0, 'along its length'), (30.0, 60.0, 'diagonally'), (60.0, 90.01, 'across its short side'))
EXPLORE = 0.30


def fold(rho_deg):
    r = rho_deg % 180.0
    return min(r, 180.0 - r)


def bin_of(rho_deg):
    f = fold(rho_deg)
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= f < hi:
            return i
    return len(BINS) - 1


class GraspPolicy:
    def __init__(self, records=(), live_only=True):
        self.stats = {}                                    # (label, bin) -> [successes, trials]
        for r in records:
            if r.get('kind') != 'pick' or (live_only and not r.get('live', True)):
                continue
            shape = next((e for e in (r.get('events') or []) if e.get('what') == 'shape' and e.get('rho') is not None), None)
            if shape is None or not r.get('label'):
                continue
            st = self.stats.setdefault((r['label'], bin_of(shape['rho'])), [0, 0])
            st[1] += 1
            st[0] += int(bool(r.get('ok')))

    @classmethod
    def from_log(cls, directory=None):
        from thesis_robot import outcome_log as ol
        return cls(ol.read_records(directory or ol.DEFAULT_DIR))

    def counts(self, label, rho_deg):
        return tuple(self.stats.get((label, bin_of(rho_deg)), (0, 0)))

    def prior(self, label, rho_deg, explore=EXPLORE):
        s, n = self.counts(label, rho_deg)
        mean = (s + 1.0) / (n + 2.0)
        return min(1.0, mean + explore / math.sqrt(n + 1.0))

    def lessons(self, min_trials=2):
        out = []
        for (label, b), (s, n) in sorted(self.stats.items()):
            if n < min_trials:
                continue
            how = BINS[b][2]
            if s == 0:
                out.append(f'Gripping the {label} {how} has failed {n}/{n} times.')
            elif s == n:
                out.append(f'Gripping the {label} {how} has worked {s}/{n} times.')
            else:
                out.append(f'Gripping the {label} {how}: {s}/{n} worked.')
        return out
