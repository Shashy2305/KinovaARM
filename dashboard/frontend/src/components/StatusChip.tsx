type Tone = 'ok' | 'warn' | 'bad' | 'neutral'

const TONE_COLOR: Record<Tone, string> = {
  ok: 'text-(--color-green)',
  warn: 'text-(--color-amber)',
  bad: 'text-(--color-red)',
  neutral: 'text-(--color-text-dim)',
}

export function StatusChip({ label, tone, pulse = false }: { label: string; tone: Tone; pulse?: boolean }) {
  return (
    <span className={`status-chip ${TONE_COLOR[tone]}`}>
      <span className={`led ${pulse ? 'led-blink' : ''}`} />
      {label}
    </span>
  )
}

export function toneForProcessStatus(status: string): Tone {
  if (status === 'running' || status === 'running_external') return 'ok'
  if (status === 'running_other_user') return 'warn'
  if (status === 'conflict') return 'bad'
  return 'neutral'
}

export function toneForCalibration(status: string | undefined): Tone {
  if (status === 'ok') return 'ok'
  if (status === 'needs_recalibration' || status === 'no_signal') return 'bad'
  return 'neutral'
}

export type { Tone }
