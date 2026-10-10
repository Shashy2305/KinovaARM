import type { HardwareHolder } from '../lib/types'

// Another account on this shared lab machine is running the robot driver or holding a camera. Our nodes cannot use that
// hardware until they stop it (the dashboard never touches other users' processes), so say who and what.
export function HardwareBanner({ holders }: { holders: HardwareHolder[] | undefined }) {
  if (!holders || holders.length === 0) return null
  const byUser: Record<string, string[]> = {}
  for (const h of holders) (byUser[h.user] ??= []).push(h.what)
  return (
    <div className="px-5 py-2 border-b border-(--color-amber)/40 bg-(--color-amber)/10 text-xs">
      <span className="text-(--color-amber) font-semibold">Hardware in use by another user. </span>
      {Object.entries(byUser).map(([user, what]) => (
        <span key={user} className="mr-3"><span className="mono">{user}</span> is running {what.join('; ')}.</span>
      ))}
      <span className="text-(--color-text-dim)">Cameras or the arm that show no signal below are probably held by them. Ask them to stop; this dashboard never stops another user's processes.</span>
    </div>
  )
}
