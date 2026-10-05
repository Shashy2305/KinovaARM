export type StageId = 'listen' | 'transcribe' | 'plan' | 'execute'
export type StageState = 'skip' | 'pending' | 'active' | 'done' | 'failed'

export interface StatusEvent { seq: number; key: string; text: string; t: number; cseq?: number }

export interface Stage {
  state: StageState
  t0?: number    // server time (s) the stage started
  t1?: number    // server time (s) it ended
  rt0?: number   // browser time (ms) it started, for the live timer
}

export interface Run {
  command: string
  voice: boolean
  stages: Record<StageId, Stage>
  failure?: string
  steps: { index: number; total: number; action: string }[]
}

export const STAGE_ORDER: StageId[] = ['listen', 'transcribe', 'plan', 'execute']

export function freshRun(voice: boolean): Run {
  return {
    command: '',
    voice,
    stages: {
      listen: { state: voice ? 'pending' : 'skip' },
      transcribe: { state: voice ? 'pending' : 'skip' },
      plan: { state: 'pending' },
      execute: { state: 'pending' },
    },
    steps: [],
    failure: undefined,
  }
}

function clone(r: Run): Run {
  return {
    ...r,
    stages: Object.fromEntries(STAGE_ORDER.map((s) => [s, { ...r.stages[s] }])) as Run['stages'],
    steps: [...r.steps],
  }
}

function begin(r: Run, id: StageId, ev: StatusEvent, rt: number) {
  for (const s of STAGE_ORDER) {
    if (s !== id && r.stages[s].state === 'active') r.stages[s] = { ...r.stages[s], state: 'done', t1: ev.t }
  }
  r.stages[id] = { state: 'active', t0: ev.t, t1: undefined, rt0: rt }
}

function finish(r: Run, id: StageId, state: 'done' | 'failed', ev: StatusEvent) {
  const st = r.stages[id]
  r.stages[id] = { ...st, state, t0: st.t0 ?? ev.t, t1: ev.t }
}

function parseStep(text: string) {
  const m = text.match(/Step (\d+)\/(\d+):\s*(.+)/)
  return m ? { index: Number(m[1]), total: Number(m[2]), action: m[3] } : null
}

/** Fold ONE status message into the current run. Pure: returns a new run
 * (or the same object when the message does not concern the pipeline). */
export function applyEvent(run: Run | null, ev: StatusEvent, rt: number): Run | null {
  const text = ev.text ?? ''
  const up = text.toUpperCase()

  if (ev.key === 'audio_status') {
    if (up.startsWith('RECORDING')) {
      if (run?.voice && run.stages.listen.state === 'active') return run
      const r = freshRun(true)
      begin(r, 'listen', ev, rt)
      return r
    }
    if (!run?.voice) return run
    const r = clone(run)
    if (up.startsWith('TRANSCRIBING')) {
      finish(r, 'listen', 'done', ev)
      begin(r, 'transcribe', ev, rt)
    } else if (up.startsWith('HEARD')) {
      finish(r, 'transcribe', 'done', ev)
      r.command = text.slice(text.indexOf(':') + 1).trim()
    } else if (up.startsWith('NO SPEECH')) {
      finish(r, 'transcribe', 'failed', ev)
      r.failure = 'No speech recognised'
    } else {
      return run
    }
    return r
  }

  if (ev.key === 'planner_status') {
    if (up.startsWith('PLANNING')) {
      const keepVoice = !!run?.voice && run.stages.transcribe.state === 'done' && run.stages.plan.state === 'pending'
      const r = keepVoice ? clone(run!) : freshRun(false)
      r.command = text.slice(text.indexOf(':') + 1).trim() || r.command
      begin(r, 'plan', ev, rt)
      return r
    }
    if (!run) return run
    if (up.startsWith('EXECUTING')) {
      const r = clone(run)
      finish(r, 'plan', 'done', ev)
      if (r.stages.execute.state === 'pending') begin(r, 'execute', ev, rt)
      return r
    }
    if (up.startsWith('REJECTED') || up.startsWith('ERROR')) {
      const r = clone(run)
      finish(r, 'plan', 'failed', ev)
      r.failure = text.replace(/^(rejected|error):?\s*/i, '')
      return r
    }
    return run
  }

  if (ev.key === 'arm_status') {
    if (!run) return run
    const low = text.toLowerCase()
    const r = clone(run)
    if (low.startsWith('executing')) {
      if (r.stages.plan.state === 'active') finish(r, 'plan', 'done', ev)
      if (r.stages.execute.state === 'pending') begin(r, 'execute', ev, rt)
    } else if (low.startsWith('complete')) {
      finish(r, 'execute', 'done', ev)
    } else if (low.startsWith('failed') || low.startsWith('blocked') || low.startsWith('safety')) {
      if (r.stages.execute.state === 'pending') begin(r, 'execute', ev, rt)
      finish(r, 'execute', 'failed', ev)
      r.failure = text
    } else if (low.startsWith('ready')) {
      if (r.stages.execute.state !== 'active') return run
      finish(r, 'execute', 'done', ev)
    } else {
      return run
    }
    return r
  }

  if (ev.key === 'pick_place_status' && run) {
    const s = parseStep(text)
    if (!s) return run
    const r = clone(run)
    r.steps = [...r.steps.filter((x) => x.index !== s.index), s].sort((a, b) => a.index - b.index)
    return r
  }

  return run
}

export function stageSeconds(s: Stage, nowMs: number): number {
  if (s.t0 !== undefined && s.t1 !== undefined) return Math.max(0, s.t1 - s.t0)
  if (s.rt0 !== undefined) return Math.max(0, (nowMs - s.rt0) / 1000)
  return 0
}
