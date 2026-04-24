/* Radial gauge component — used for EAR, PERCLOS, Confidence */
export default function GaugeChart({ value = 0, min = 0, max = 1, label = '', unit = '', danger = false, warning = false, size = 120 }) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)))
  const angle = pct * 270 - 135   // sweep from -135° to +135°
  const R = size / 2 - 10
  const cx = size / 2
  const cy = size / 2 + 10

  const polarToXY = (deg, r) => {
    const rad = (deg - 90) * Math.PI / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  const arcPath = (startDeg, endDeg, r) => {
    const start = polarToXY(startDeg, r)
    const end = polarToXY(endDeg, r)
    const large = endDeg - startDeg > 180 ? 1 : 0
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`
  }

  const color = danger ? 'var(--red)' : warning ? 'var(--amber)' : 'var(--cyan)'
  const needleTip = polarToXY(angle, R - 4)
  const needleBase1 = polarToXY(angle - 90, 4)
  const needleBase2 = polarToXY(angle + 90, 4)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width={size} height={size * 0.85} viewBox={`0 0 ${size} ${size}`}>
        {/* Track */}
        <path d={arcPath(-135, 135, R)} fill="none"
          stroke="var(--border)" strokeWidth={6} strokeLinecap="round" />

        {/* Fill */}
        {pct > 0 && (
          <path d={arcPath(-135, angle, R)} fill="none"
            stroke={color} strokeWidth={6} strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
        )}

        {/* Needle */}
        <polygon
          points={`${needleTip.x},${needleTip.y} ${needleBase1.x},${needleBase1.y} ${needleBase2.x},${needleBase2.y}`}
          fill={color} opacity={0.9}
          style={{ filter: `drop-shadow(0 0 3px ${color})` }} />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={4} fill="var(--bg-elevated)"
          stroke={color} strokeWidth={2} />

        {/* Value text */}
        <text x={cx} y={cy - 22} textAnchor="middle"
          fontFamily="var(--font-mono)" fontWeight="700"
          fontSize={size * 0.14} fill="var(--text-primary)">
          {typeof value === 'number' ? value.toFixed(2) : '--'}
        </text>
        <text x={cx} y={cy - 8} textAnchor="middle"
          fontFamily="var(--font-mono)" fontSize={8}
          fill="var(--text-muted)" letterSpacing="0.1em">
          {unit}
        </text>
      </svg>
      <div className="label" style={{ textAlign: 'center' }}>{label}</div>
    </div>
  )
}
