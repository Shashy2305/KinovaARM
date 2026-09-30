import type { StatusPayload } from '../lib/types'
import { StatusChip, toneForCalibration } from './StatusChip'

export function TopStatusBar({
  status,
  connected,
}: {
  status: StatusPayload | null
  connected: boolean
}) {
  const calib = status?.ros.camera_calibration_status
  const armStatus = status?.ros.arm_status ?? 'unknown'
  const isLive = armStatus.toLowerCase().includes('live')
  const robotUp = status?.processes.robot_bringup?.status !== 'stopped'

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-(--color-border) bg-(--color-panel)/80 backdrop-blur sticky top-0 z-20">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-(--color-cyan) to-(--color-cyan-dim) flex items-center justify-center font-mono font-bold text-(--color-bg) text-sm shadow-[0_0_20px_rgba(34,211,238,0.35)]">
          K
        </div>
        <div>
          <div className="font-semibold tracking-tight text-base leading-none">KinovaARM</div>
          <div className="text-xs text-(--color-text-dim) mt-0.5 font-mono">Mission Control</div>
        </div>
      </div>

      <div className="flex items-center gap-2.5 flex-wrap justify-end">
        <StatusChip
          label={connected ? 'Live Link' : 'Reconnecting'}
          tone={connected ? 'ok' : 'warn'}
          pulse={connected}
        />
        <StatusChip label={robotUp ? 'Robot Up' : 'Robot Down'} tone={robotUp ? 'ok' : 'neutral'} />
        <StatusChip label={`OAK-D ${calib?.oakd ?? '—'}`} tone={toneForCalibration(calib?.oakd)} />
        <StatusChip
          label={`RealSense ${calib?.realsense ?? '—'}`}
          tone={toneForCalibration(calib?.realsense)}
        />
        <StatusChip label={`Wrist ${calib?.wrist ?? '—'}`} tone={toneForCalibration(calib?.wrist)} />
        <StatusChip
          label={isLive ? 'ARM: LIVE' : 'ARM: DRY RUN'}
          tone={isLive ? 'bad' : 'warn'}
          pulse={isLive}
        />
      </div>
    </header>
  )
}
