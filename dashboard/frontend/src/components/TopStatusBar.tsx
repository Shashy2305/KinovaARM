import type { StatusPayload } from '../lib/types'
import { StatusChip, toneForCalibration } from './StatusChip'

function titleCase(s: string) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

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
    <header className="flex items-center justify-between px-8 py-5 border-b border-(--color-border) bg-(--color-bg)/70 backdrop-blur-xl sticky top-0 z-20">
      <div className="flex items-center gap-3.5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-(--color-brand-from) to-(--color-brand-to) flex items-center justify-center font-bold text-white text-base shadow-[0_6px_24px_-6px_rgba(139,92,246,0.7)]">
          K
        </div>
        <div>
          <div className="font-semibold tracking-tight text-[1.05rem] leading-none">KinovaARM</div>
          <div className="text-[0.8rem] text-(--color-text-dim) mt-1">Pick &amp; Place Control</div>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap justify-end">
        <StatusChip label={connected ? 'Connected' : 'Reconnecting'} tone={connected ? 'ok' : 'warn'} pulse={connected} />
        <StatusChip label={robotUp ? 'Robot Online' : 'Robot Offline'} tone={robotUp ? 'ok' : 'neutral'} />
        <StatusChip label={`OAK-D · ${titleCase(calib?.oakd ?? 'unknown')}`} tone={toneForCalibration(calib?.oakd)} />
        <StatusChip
          label={`RealSense · ${titleCase(calib?.realsense ?? 'unknown')}`}
          tone={toneForCalibration(calib?.realsense)}
        />
        <StatusChip label={`Wrist · ${titleCase(calib?.wrist ?? 'unknown')}`} tone={toneForCalibration(calib?.wrist)} />
        <StatusChip label={isLive ? 'Arm: Live' : 'Arm: Dry Run'} tone={isLive ? 'bad' : 'warn'} pulse={isLive} />
      </div>
    </header>
  )
}
