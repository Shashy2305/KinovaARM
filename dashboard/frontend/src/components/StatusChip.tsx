type Tone = 'ok' | 'warn' | 'bad' | 'neutral'

const TONE_STYLES: Record<Tone, string> = {
  ok: 'text-(--color-green) border-(--color-green)/35 bg-(--color-green)/10',
  warn: 'text-(--color-amber) border-(--color-amber)/35 bg-(--color-amber)/10',
  bad: 'text-(--color-red) border-(--color-red)/35 bg-(--color-red)/10',
  neutral: 'text-(--color-text-dim) border-(--color-border-bright) bg-white/[0.02]',
}

export function StatusChip({
  label,
  tone,
  pulse = false,
}: {
  label: string
  tone: Tone
  pulse?: boolean
}) {
  return (
    <span className={`status-chip ${TONE_STYLES[tone]}`}>
      {pulse && <span className="live-dot" style={{ background: 'currentColor' }} />}
      {label}
    </span>
  )
}

export function toneForProcessStatus(status: string): Tone {
  if (status === 'running' || status === 'running_external') return 'ok'
  if (status === 'conflict') return 'bad'
  return 'neutral'
}

export function toneForCalibration(status: string | undefined): Tone {
  if (status === 'ok') return 'ok'
  if (status === 'needs_recalibration') return 'bad'
  return 'neutral'
}
