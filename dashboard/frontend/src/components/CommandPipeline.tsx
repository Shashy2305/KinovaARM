import { useEffect, useRef, useState } from 'react'
import type { RosStatus } from '../lib/types'

type Stage = 'idle' | 'planning' | 'executing' | 'done' | 'failed'

interface StepState {
  index: number
  total: number
  action: string
  done: boolean
}

function parseStep(text: string): { index: number; total: number; action: string } | null {
  // arm_controller_node.py logs "Step i/N: action" to /pick_place_status
  const m = text.match(/Step (\d+)\/(\d+):\s*(.+)/)
  if (!m) return null
  return { index: Number(m[1]), total: Number(m[2]), action: m[3] }
}

/** Visualizes the real pipeline a command goes through, driven entirely by
 * /planner_status and /pick_place_status text -- no separate "plan" topic
 * exists today, so step-by-step state is reconstructed by watching how
 * those two status strings change over time, not read from one snapshot. */
export function CommandPipeline({ ros }: { ros: RosStatus | null }) {
  const [stage, setStage] = useState<Stage>('idle')
  const [steps, setSteps] = useState<StepState[]>([])
  const lastPlanner = useRef<string | null>(null)
  const lastPickPlace = useRef<string | null>(null)

  useEffect(() => {
    const planner = ros?.planner_status ?? null
    if (planner && planner !== lastPlanner.current) {
      lastPlanner.current = planner
      const lower = planner.toLowerCase()
      if (lower.startsWith('planning')) {
        setStage('planning')
        setSteps([])
      } else if (lower.startsWith('rejected') || lower.includes('error')) {
        setStage('failed')
      } else if (lower.startsWith('executing')) {
        setStage('executing')
      } else if (lower.includes('complete')) {
        setStage('done')
      } else if (lower.startsWith('ready') && stage === 'executing') {
        // arm_controller republishes READY after finishing a plan either way
        setStage((s) => (s === 'executing' ? 'done' : s))
      }
    }
  }, [ros?.planner_status, stage])

  useEffect(() => {
    const pp = ros?.pick_place_status ?? null
    if (pp && pp !== lastPickPlace.current) {
      lastPickPlace.current = pp
      const parsed = parseStep(pp)
      if (parsed) {
        setSteps((prev) => {
          const next: StepState[] = []
          for (let i = 1; i <= parsed.total; i++) {
            const existing = prev.find((s) => s.index === i)
            next.push({
              index: i,
              total: parsed.total,
              action: i === parsed.index ? parsed.action : existing?.action ?? '…',
              done: i < parsed.index || existing?.done || false,
            })
          }
          return next
        })
      }
      if (pp.toUpperCase().includes('FAILED')) setStage('failed')
    }
  }, [ros?.pick_place_status])

  if (stage === 'idle') return null

  const STAGE_META: Record<Stage, { label: string; tone: string }> = {
    idle: { label: '', tone: '' },
    planning: { label: 'Thinking…', tone: 'text-(--color-amber)' },
    executing: { label: 'Executing', tone: 'text-(--color-brand)' },
    done: { label: 'Complete', tone: 'text-(--color-green)' },
    failed: { label: 'Failed / Rejected', tone: 'text-(--color-red)' },
  }

  return (
    <div className="rounded-xl bg-black/20 p-4 space-y-3 border border-(--color-border)">
      <div className="flex items-center gap-2.5">
        {(stage === 'planning' || stage === 'executing') && (
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-60" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-current" />
          </span>
        )}
        <span className={`text-sm font-semibold ${STAGE_META[stage].tone}`}>{STAGE_META[stage].label}</span>
      </div>

      {steps.length > 0 && (
        <ol className="space-y-1.5">
          {steps.map((s) => {
            const isCurrent = !s.done && steps.every((x) => x.index >= s.index || x.done)
            return (
              <li key={s.index} className="flex items-center gap-2.5 text-sm">
                <span
                  className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 transition-colors ${
                    s.done
                      ? 'bg-(--color-green)/20 text-(--color-green)'
                      : isCurrent
                        ? 'bg-(--color-brand)/25 text-(--color-brand)'
                        : 'bg-white/5 text-(--color-text-faint)'
                  }`}
                >
                  {s.done ? '✓' : s.index}
                </span>
                <span className={s.done ? 'text-(--color-text-dim) line-through decoration-(--color-text-faint)' : isCurrent ? 'text-(--color-text)' : 'text-(--color-text-faint)'}>
                  {s.action}
                </span>
                {isCurrent && (
                  <span className="relative flex h-1.5 w-1.5 ml-auto">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-(--color-brand) opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-(--color-brand)" />
                  </span>
                )}
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
