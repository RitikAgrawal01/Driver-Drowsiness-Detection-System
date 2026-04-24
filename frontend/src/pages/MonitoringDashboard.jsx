import { useEffect, useState } from 'react'
import { ExternalLink, AlertTriangle, CheckCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from 'recharts'
import { useSessionStore } from '../store/sessionStore'

const GRAFANA = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3001'
const PROM    = import.meta.env.VITE_PROMETHEUS_URL || 'http://localhost:9090'

const FEATURE_COLORS = {
  ear_mean: '#00D4FF', ear_min: '#0099BB', ear_std: '#66E5FF',
  perclos: '#FF3B3B', mar_mean: '#F59E0B', mar_max: '#B45309',
  head_pitch_mean: '#10B981', head_yaw_mean: '#34D399', head_roll_mean: '#6EE7B7',
}

export default function MonitoringDashboard() {
  const { driftHistory, featureDriftScores, overallDriftScore, latencyHistory, confidenceHistory } = useSessionStore()
  const isDrifting = overallDriftScore > 0.15
  const [grafanaLoaded, setGrafanaLoaded] = useState(false)

  const featureDriftData = Object.entries(featureDriftScores).map(([k, v]) => ({
    name: k.replace('_mean', '').replace('head_', ''),
    value: Number(v.toFixed(4)),
    color: FEATURE_COLORS[k] || '#7A9AB5',
  })).sort((a, b) => b.value - a.value)

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, letterSpacing: '0.05em' }}>
            Monitoring Dashboard
          </h1>
          <div className="label" style={{ marginTop: 4 }}>Prometheus · Grafana · Real-time drift detection</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <a href={GRAFANA} target="_blank" rel="noreferrer" style={{
            padding: '8px 14px', background: 'var(--bg-elevated)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius)',
            color: 'var(--amber)', fontSize: 11, textDecoration: 'none',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <ExternalLink size={12} /> Grafana ↗
          </a>
          <a href={`${PROM}/alerts`} target="_blank" rel="noreferrer" style={{
            padding: '8px 14px', background: 'var(--bg-elevated)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius)',
            color: 'var(--cyan)', fontSize: 11, textDecoration: 'none',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <ExternalLink size={12} /> Prometheus ↗
          </a>
        </div>
      </div>

      {/* Drift alert banner */}
      {isDrifting && (
        <div style={{
          marginBottom: 20, padding: '14px 20px',
          background: 'rgba(255,59,59,0.08)',
          border: '1px solid rgba(255,59,59,0.4)', borderRadius: 'var(--radius-lg)',
          display: 'flex', alignItems: 'center', gap: 12,
          animation: 'fadeSlideUp 0.3s ease',
        }}>
          <AlertTriangle size={18} color="var(--red)" />
          <div>
            <div style={{ color: 'var(--red)', fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13 }}>
              DATA DRIFT DETECTED — Score: {overallDriftScore.toFixed(4)} (threshold: 0.15)
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: 11, marginTop: 2 }}>
              Feature distributions have shifted from training baseline. Consider triggering dag_retrain in Airflow.
            </div>
          </div>
        </div>
      )}

      {/* ── Grafana embed ── */}
      <div className="card" style={{ marginBottom: 20, padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13 }}>Grafana Live Dashboard</div>
            <div className="label" style={{ marginTop: 2 }}>Embedded from localhost:3001 — refreshes every 10s</div>
          </div>
          {!grafanaLoaded && (
            <span className="badge badge-amber">Connecting...</span>
          )}
        </div>
        <iframe
          src={`${GRAFANA}/d/ddd-main/driver-drowsiness-detection-live-monitor?orgId=1&refresh=10s&kiosk=tv`}
          style={{ width: '100%', height: 420, border: 'none', background: 'var(--bg-base)' }}
          title="Grafana Dashboard"
          onLoad={() => setGrafanaLoaded(true)}
        />
        {!grafanaLoaded && (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
            If Grafana is not running, start it with: <span style={{ color: 'var(--cyan)' }}>docker-compose up -d grafana</span>
            <br />Then open <a href={GRAFANA} target="_blank" rel="noreferrer" style={{ color: 'var(--cyan)' }}>{GRAFANA}</a> (admin / admin)
          </div>
        )}
      </div>

      {/* ── Live charts from current session ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 20 }}>

        {/* Drift over time */}
        <div className="card">
          <div className="label" style={{ marginBottom: 10 }}>Overall Drift Score — Live</div>
          <div style={{
            fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 28,
            color: isDrifting ? 'var(--red)' : 'var(--text-primary)',
            lineHeight: 1, marginBottom: 10,
          }}>
            {overallDriftScore.toFixed(4)}
          </div>
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={driftHistory}>
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 0.3]} hide />
              <ReferenceLine y={0.15} stroke="rgba(255,59,59,0.5)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="v" dot={false} strokeWidth={2}
                stroke={isDrifting ? 'var(--red)' : 'var(--cyan)'}
                isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
            {isDrifting
              ? <><AlertTriangle size={11} color="var(--red)" /><span style={{ fontSize: 10, color: 'var(--red)' }}>Above threshold</span></>
              : <><CheckCircle size={11} color="var(--green)" /><span style={{ fontSize: 10, color: 'var(--green)' }}>Within baseline</span></>
            }
          </div>
        </div>

        {/* Latency */}
        <div className="card">
          <div className="label" style={{ marginBottom: 10 }}>Inference Latency (ms)</div>
          <div style={{
            fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 28,
            color: 'var(--text-primary)', lineHeight: 1, marginBottom: 10,
          }}>
            {latencyHistory.length ? latencyHistory[latencyHistory.length - 1].v.toFixed(1) : '--'}
            <span style={{ fontSize: 13, color: 'var(--text-muted)', marginLeft: 4 }}>ms</span>
          </div>
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={latencyHistory}>
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 250]} hide />
              <ReferenceLine y={200} stroke="rgba(255,59,59,0.5)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="v" dot={false} strokeWidth={2}
                stroke="var(--purple)" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            SLA threshold: 200ms
          </div>
        </div>

        {/* Confidence */}
        <div className="card">
          <div className="label" style={{ marginBottom: 10 }}>Prediction Confidence</div>
          <div style={{
            fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 28,
            color: 'var(--text-primary)', lineHeight: 1, marginBottom: 10,
          }}>
            {confidenceHistory.length
              ? (confidenceHistory[confidenceHistory.length - 1].v * 100).toFixed(1)
              : '--'}
            <span style={{ fontSize: 13, color: 'var(--text-muted)', marginLeft: 2 }}>%</span>
          </div>
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={confidenceHistory}>
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 1]} hide />
              <ReferenceLine y={0.7} stroke="rgba(245,158,11,0.5)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="v" dot={false} strokeWidth={2}
                stroke="var(--amber)" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Alert threshold: 70%
          </div>
        </div>
      </div>

      {/* Per-feature drift */}
      <div className="card">
        <div className="label" style={{ marginBottom: 14 }}>Per-Feature Drift Scores (KL Divergence vs Training Baseline)</div>
        {featureDriftData.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            Start a monitoring session to see live drift scores.
            Or run: <span style={{ color: 'var(--cyan)' }}>python tools/simulate_drift.py --mode lighting --count 300</span>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
            {featureDriftData.map(({ name, value, color }) => (
              <div key={name} style={{
                padding: '10px 14px', background: 'var(--bg-elevated)',
                border: `1px solid ${value > 0.15 ? 'rgba(255,59,59,0.3)' : 'var(--border)'}`,
                borderRadius: 'var(--radius)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{name}</span>
                  <span style={{
                    fontSize: 11, fontFamily: 'var(--font-display)', fontWeight: 700,
                    color: value > 0.15 ? 'var(--red)' : value > 0.08 ? 'var(--amber)' : 'var(--green)',
                  }}>{value.toFixed(4)}</span>
                </div>
                <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.min(100, value / 0.3 * 100)}%`,
                    background: value > 0.15 ? 'var(--red)' : value > 0.08 ? 'var(--amber)' : color,
                    borderRadius: 2, transition: 'width 0.5s ease',
                  }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
