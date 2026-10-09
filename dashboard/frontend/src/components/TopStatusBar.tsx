import type { StatusPayload } from '../lib/types'
import { useTableStatus } from '../lib/useTableStatus'
import type { Tone } from './StatusChip'

const COLOR: Record<Tone, string> = {
  ok: 'var(--color-green)', warn: 'var(--color-amber)', bad: 'var(--color-red)', neutral: 'var(--color-text-faint)',
}

function Cell({ label, value, tone, blink = false, wide = false }: {
  label: string; value: string; tone: Tone; blink?: boolean; wide?: boolean
}) {
  return (
    <div className={`px-3.5 py-2 border-r border-(--color-border) ${wide ? 'min-w-40' : 'min-w-28'}`}>
      <div className="label">{label}</div>
      <div className="flex items-center gap-1.5 mt-0.5 font-mono text-[0.8rem] whitespace-nowrap" style={{ color: COLOR[tone] }}>
        <span className={`led ${blink ? 'led-blink' : ''}`} />
        {value}
      </div>
    </div>
  )
}

export function TopStatusBar({ status, connected }: { status: StatusPayload | null; connected: boolean }) {
  const table = useTableStatus()
  const calib = status?.ros.camera_calibration_status
  const armStatus = status?.ros.arm_status ?? ''
  const isLive = armStatus.toLowerCase().includes('live')
  const robot = status?.processes.robot_bringup?.status
  const robotUp = !!robot && robot !== 'stopped'
  const noJoints = robotUp && status?.ros.joint_state_publishers === 0     // driver process alive but nothing publishes the arm's joint states

  const cams = ['oakd', 'realsense', 'realsense2', 'wrist']
  const camOff = cams.filter((c) => calib?.[c] === 'disabled')     // switched off on purpose: neither ok nor a fault
  const camOk = cams.filter((c) => calib?.[c] === 'ok').length
  const camBad = cams.some((c) => calib?.[c] === 'needs_recalibration')
  const camDown = cams.filter((c) => calib?.[c] === 'no_signal')

  const audio = (status?.ros.audio_status ?? '').toUpperCase()
  const voice = !status?.processes.audio_node || status.processes.audio_node.status === 'stopped'
    ? { v: 'off', t: 'neutral' as Tone }
    : audio.startsWith('RECORDING') ? { v: 'recording', t: 'bad' as Tone }
      : audio.startsWith('LISTENING') ? { v: 'hands-free', t: 'ok' as Tone }
        : audio.startsWith('LOADING') ? { v: 'loading', t: 'warn' as Tone }
          : { v: 'ready', t: 'ok' as Tone }

  const tableCell = !table
    ? { v: 'unknown', t: 'neutral' as Tone }
    : table.saved
      ? { v: `top z ${table.saved.table_top_z.toFixed(2)}`, t: 'ok' as Tone }
      : { v: 'not recorded', t: 'bad' as Tone }

  return (
    <header className="sticky top-0 z-20 bg-(--color-bg) border-b border-(--color-border)">
      {isLive && <div className="hazard h-1.5" />}
      <div className="flex flex-wrap items-stretch border-l border-(--color-border)">
        <Cell label="Link" value={connected ? 'connected' : 'reconnecting'} tone={connected ? 'ok' : 'warn'} blink={!connected} />
        <Cell label="Robot" value={noJoints ? 'NO JOINT STATES - restart bringup' : robotUp ? 'bringup up' : 'bringup off'} tone={noJoints ? 'bad' : robotUp ? 'ok' : 'neutral'} wide={noJoints} blink={noJoints} />
        <Cell label="Arm" value={isLive ? 'LIVE — MOVES' : 'dry run'} tone={isLive ? 'bad' : 'warn'} blink={isLive} />
        <Cell label="Table" value={tableCell.v} tone={tableCell.t} wide />
        <Cell label="Cameras" value={camDown.length ? `no signal: ${camDown.join(', ')}` : `${camOk}/${cams.length - camOff.length} ok${camOff.length ? ` · ${camOff.join(', ')} off` : ''}${camBad ? ' · recal' : ''}`} tone={camBad || camDown.length ? 'bad' : camOk === cams.length - camOff.length ? 'ok' : 'warn'} wide={camDown.length > 0 || camOff.length > 0} />
        <Cell label="Voice" value={voice.v} tone={voice.t} blink={voice.v === 'recording'} />
      </div>
    </header>
  )
}
