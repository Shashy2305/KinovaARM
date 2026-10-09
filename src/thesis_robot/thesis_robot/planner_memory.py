"""
Planner memory (pure Python, no ROS): remembers which plans the language model gave for which kind of command and
how they turned out, and turns that into a SHORT block for the next prompt.

The model's context is 4096 tokens and the system prompt plus the scene already use about half of it, so the memory
is kept to at most one worked example and one warning (~350 characters).

How it generalises: object names in a command are replaced by slots, so "pick up the cup and put it next to the bowl"
and "pick up the mouse and put it next to the bottle" are the SAME template ("pick up the <A> and put it next to the
<B>"). A plan that worked for the first becomes a ready example for the second, with the current scene's object ids
filled in. The plan itself is stored as a skeleton (actions and which slot they touch), never coordinates.

What is NOT in here: physical grasp lessons (which closing angle worked on a mouse). Those change what the arm does,
not what the model should write, so they live in the controller (grasp_planner + the outcome log).
"""
import json
import os
import re
import threading
import time

from thesis_robot import command_grammar as cg

DEFAULT_DIR = os.path.expanduser('~/.ros/planner_memory')
MAX_BLOCK_CHARS = 420
MIN_TEMPLATE_SIMILARITY = 0.85

LABEL_SYNONYMS = {
    'cup': 'cup', 'mug': 'cup', 'bowl': 'bowl', 'bottle': 'bottle', 'mouse': 'mouse', 'phone': 'cell phone',
    'remote': 'remote', 'book': 'book', 'scissors': 'scissors', 'vase': 'vase',
}
STOPWORDS = {'the', 'a', 'an', 'please', 'it', 'that', 'this', 'to', 'and', 'then', 'of'}

# Canonical, always-valid examples (the same forms the system prompt describes): used until real ones exist.
SEEDS = [
    {'template': 'pick up the <A> and put it next to the <B>',
     'skeleton': [{'action': 'pick', 'slot': 'A'}, {'action': 'place', 'near': 'B'}]},
    {'template': 'pick up the <A>', 'skeleton': [{'action': 'pick', 'slot': 'A'}]},
    {'template': 'pick up the <A> and put it aside', 'skeleton': [{'action': 'pick', 'slot': 'A'}, {'action': 'place', 'aside': True}]},
]


# ── command <-> template ───────────────────────────────────────────────────────────────────────────────────────────
def templatize(command):
    """(template text, [labels in order of appearance]) with object names replaced by <A>, <B>, ..."""
    words = re.findall(r"[a-z]+", (command or '').lower())
    labels, out = [], []
    for w in words:
        lab = LABEL_SYNONYMS.get(w)
        if lab is not None:
            if lab not in labels:
                labels.append(lab)
            out.append(f'<{chr(65 + labels.index(lab))}>')
        else:
            out.append(w)
    return ' '.join(out), labels


def _tokens(template):
    return {w for w in template.split() if w not in STOPWORDS}


def similarity(t1, t2):
    a, b = _tokens(t1), _tokens(t2)
    return len(a & b) / len(a | b) if a | b else 0.0


# ── plan <-> skeleton ──────────────────────────────────────────────────────────────────────────────────────────────
def _label_of(obj_id, scene):
    o = (scene or {}).get(obj_id) or {}
    return o.get('label')


def skeletonize(plan, scene, labels):
    """Raw plan (the model's own steps, with object ids) -> [{'action', 'slot'|'near'|'aside'|'here'}], or None when
    the plan cannot be expressed in the command's slots (an object that is not one of the command's labels)."""
    slot_of = {lab: chr(65 + i) for i, lab in enumerate(labels)}
    out = []
    for st in plan:
        if not isinstance(st, dict):
            return None
        act = st.get('action')
        if act == 'pick':
            lab = _label_of(st.get('object_id'), scene)
            if lab not in slot_of:
                return None
            out.append({'action': 'pick', 'slot': slot_of[lab]})
        elif act == 'place':
            if st.get('here'):
                out.append({'action': 'place', 'here': True})
            elif st.get('near'):
                lab = _label_of(st.get('near'), scene)
                if lab not in slot_of:
                    return None
                sk = {'action': 'place', 'near': slot_of[lab]}
                if st.get('side') in ('left', 'right', 'front', 'behind'):
                    sk['side'] = st['side']
                out.append(sk)
            else:
                out.append({'action': 'place', 'aside': True})        # coordinates are never remembered
        elif act in ('go_home', 'open_gripper', 'close_gripper'):
            out.append({'action': act})
        else:
            return None
    return out or None


def _best_object(scene, label):
    best, best_c = None, -1.0
    for oid, o in (scene or {}).items():
        if isinstance(o, dict) and o.get('label') == label and o.get('reachable') and not o.get('stale'):
            c = float(o.get('confidence', 0) or 0)
            if c > best_c:
                best, best_c = oid, c
    return best


def instantiate(skeleton, labels, scene):
    """Skeleton + the new command's labels + the current scene -> concrete steps in the model's own format, or None
    if a needed object is not in the scene."""
    ids = {}
    for i, lab in enumerate(labels):
        oid = _best_object(scene, lab)
        if oid is None:
            return None
        ids[chr(65 + i)] = oid
    out = []
    for st in skeleton:
        if st['action'] == 'pick':
            if st['slot'] not in ids:
                return None
            out.append({'action': 'pick', 'object_id': ids[st['slot']], 'approach_z': 0.35})
        elif st['action'] == 'place':
            if st.get('here'):
                out.append({'action': 'place', 'here': True})
            elif st.get('near'):
                if st['near'] not in ids:
                    return None
                step = {'action': 'place', 'near': ids[st['near']]}
                if st.get('side'):
                    step['side'] = st['side']
                out.append(step)
            else:
                return None            # "aside" positions come from the planner's free-spot search, not from the model
        else:
            out.append({'action': st['action']})
    return out


# ── the memory ─────────────────────────────────────────────────────────────────────────────────────────────────────
class PlannerMemory:
    def __init__(self, directory=DEFAULT_DIR):
        self.dir = directory
        self.path = os.path.join(directory, 'memory.jsonl')
        self._lock = threading.Lock()
        self._inflight = None

    # storage
    def _append(self, rec):
        with self._lock:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.path, 'a') as f:
                f.write(json.dumps(rec, separators=(',', ':')) + '\n')

    def entries(self):
        out = []
        try:
            with open(self.path) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return out

    # recording
    def record_rejected(self, command, plan, scene, reason):
        tmpl, labels = templatize(command)
        self._append({'ts': time.time(), 'template': tmpl, 'result': 'rejected', 'reason': str(reason)[:120],
                      'skeleton': skeletonize(plan, scene, labels) if isinstance(plan, list) else None})

    def begin(self, command, plan, scene):
        """The plan was approved and published; its outcome arrives later through finish()."""
        tmpl, labels = templatize(command)
        sk = skeletonize(plan, scene, labels) if isinstance(plan, list) else None
        self._inflight = {'ts': time.time(), 'template': tmpl, 'skeleton': sk, 'labels': labels}

    def finish(self, status_text, live=None):
        """Called with the arm's status text after a begin(): COMPLETE -> worked, FAILED/ERROR -> failed."""
        inf, self._inflight = self._inflight, None
        if inf is None or not inf.get('skeleton'):
            return None
        s = (status_text or '').upper()
        if s.startswith('COMPLETE'):
            result = 'ok'
        elif s.startswith('FAILED') or s.startswith('ERROR'):
            result = 'failed'
        else:
            self._inflight = inf
            return None
        rec = {'ts': inf['ts'], 'template': inf['template'], 'skeleton': inf['skeleton'], 'result': result,
               'live': live, 'reason': None if result == 'ok' else status_text[:120]}
        self._append(rec)
        return rec

    # retrieval
    def _best(self, template, results, pool):
        best, score = None, 0.0
        for e in pool:
            if e.get('result') not in results:
                continue
            if 'ok' in results and not e.get('skeleton'):        # a worked example needs its steps; a warning needs only the reason
                continue
            s = similarity(template, e['template'])
            if s > score or (s == score and best is not None and e.get('live') and not best.get('live')):
                best, score = e, s
        return (best, score) if score >= MIN_TEMPLATE_SIMILARITY else (None, 0.0)

    def prompt_block(self, command, scene):
        """The text to add to the prompt for `command` (may be ''): one worked example from a similar command that
        worked, and one warning from a similar one that was rejected or failed. Never more than MAX_BLOCK_CHARS."""
        tmpl, labels = templatize(command)
        intent = cg.parse(command)
        if intent is not None and intent['kind'] in ('go_home', 'put_down'):
            # no objects involved: the structure is fixed, show it (the model once answered "go home" with a move_to)
            return 'A SIMILAR COMMAND THAT WORKED - use the same structure: ' + json.dumps(
                {'plan': cg.build_plan(intent, scene)}, separators=(',', ':'))
        if not labels:
            return ''
        pool = self.entries()
        for sd in SEEDS:
            pool.append({'template': sd['template'], 'skeleton': sd['skeleton'], 'result': 'ok', 'live': None})
        lines = []
        ex, _ = self._best(tmpl, ('ok',), pool)
        if ex is not None:
            steps = instantiate(ex['skeleton'], labels, scene)
            if steps:
                line = 'A SIMILAR COMMAND THAT WORKED - use the same structure: ' + json.dumps({'plan': steps}, separators=(',', ':'))
                if len(line) <= MAX_BLOCK_CHARS - 150:          # never cut a JSON example in half: skip it instead
                    lines.append(line)
        bad, _ = self._best(tmpl, ('rejected', 'failed'), self.entries())
        if bad is not None and bad.get('reason'):
            lines.append(f'A SIMILAR COMMAND WENT WRONG BEFORE ({bad["reason"][:90]}) - avoid that.')
        return '\n'.join(lines)
