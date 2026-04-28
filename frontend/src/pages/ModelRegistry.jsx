import { useEffect, useState } from 'react'
import { ExternalLink, Trophy } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { getMLflowRuns } from '../services/api' // <-- IMPORTING THE NEW API

const MLFLOW = import.meta.env.VITE_MLFLOW_URL || 'http://127.0.0.1:5000'
const stageColor = { Production: 'var(--green)', Staging: 'var(--amber)', Archived: 'var(--text-muted)' }
const MODEL_COLORS = { XGBoost: 'var(--cyan)', SVM: 'var(--purple)', Unknown: 'var(--amber)' }

export default function ModelRegistry() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    getMLflowRuns().then(data => {
      // Filter out empty/failed runs that don't have metrics yet
      const validRuns = data.filter(r => r.metrics && Object.keys(r.metrics).length > 0)
      setRuns(validRuns)
      setSelected(validRuns[0] || null)
      setLoading(false)
    })
  }, [])

  // Find the current production model
  const winner = runs.find(r => r.stage === 'Production') || runs[0]

  // Prepare data for the Bar Chart comparing models
  const comparisonData = [
    { metric: 'F1 Score', ...Object.fromEntries(runs.map(r => [r.model_type, r.metrics.f1_weighted || 0])) },
    { metric: 'Accuracy', ...Object.fromEntries(runs.map(r => [r.model_type, r.metrics.accuracy || 0])) },
    { metric: 'AUC-ROC',  ...Object.fromEntries(runs.map(r => [r.model_type, r.metrics.auc_roc || 0])) },
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, letterSpacing: '0.05em' }}>Model Registry</h1>
          <div className="label" style={{ marginTop: 4 }}>MLflow experiments · drowsiness_detection</div>
        </div>
        <a href={MLFLOW} target="_blank" rel="noreferrer" style={{
          padding: '8px 14px', background: 'var(--bg-elevated)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius)',
          color: 'var(--cyan)', fontSize: 11, textDecoration: 'none',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <ExternalLink size={12} /> Open MLflow UI ↗
        </a>
      </div>

      {/* Production model banner */}
      {winner && !loading && (
        <div style={{
          marginBottom: 20, padding: '16px 20px',
          background: 'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(0,212,255,0.04))',
          border: '1px solid rgba(16,185,129,0.3)', borderRadius: 'var(--radius-lg)',
          display: 'flex', alignItems: 'center', gap: 16,
          animation: 'fadeSlideUp 0.3s ease',
        }}>
          <div style={{
            width: 44, height: 44, borderRadius: '50%',
            background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.4)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <Trophy size={20} color="var(--green)" />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14, color: 'var(--green)' }}>
              Production Model — {winner.model_type}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
              F1: {winner.metrics.f1_weighted?.toFixed(4)} · AUC-ROC: {winner.metrics.auc_roc?.toFixed(4)}
            </div>
          </div>
          <span className="badge badge-green">PRODUCTION</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 20 }}>

        {/* ── Run list ── */}
        <div className="card" style={{ height: 'fit-content' }}>
          <div className="label" style={{ marginBottom: 12 }}>Experiment Runs</div>
          {loading ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading from MLflow...</div>
          ) : runs.length === 0 ? (
             <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No successful runs found.</div>
          ) : (
            runs.map(run => {
              const isSel = selected?.run_id === run.run_id
              const color = MODEL_COLORS[run.model_type] || 'var(--text-muted)'
              return (
                <div key={run.run_id} onClick={() => setSelected(run)}
                  style={{
                    padding: '12px', borderRadius: 'var(--radius)', marginBottom: 6,
                    cursor: 'pointer',
                    background: isSel ? 'rgba(0,212,255,0.07)' : 'var(--bg-elevated)',
                    border: `1px solid ${isSel ? 'rgba(0,212,255,0.25)' : 'var(--border)'}`,
                    transition: 'all var(--transition)',
                  }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                        {run.run_name}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                        {new Date(run.start_time).toLocaleString()}
                      </div>
                    </div>
                    {run.stage && (
                      <span style={{
                        fontSize: 9, padding: '2px 7px', borderRadius: 999,
                        background: `${stageColor[run.stage] || '#444'}22`,
                        color: stageColor[run.stage] || '#fff',
                        border: `1px solid ${stageColor[run.stage] || '#444'}44`,
                        letterSpacing: '0.1em',
                      }}>
                        {run.stage}
                      </span>
                    )}
                  </div>
                  {run.metrics.f1_weighted && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                      <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                        F1: <span style={{ color }}>{run.metrics.f1_weighted?.toFixed(3)}</span>
                      </span>
                      <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                        AUC: <span style={{ color }}>{run.metrics.auc_roc?.toFixed(3)}</span>
                      </span>
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* ── Detail + comparison ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Selected run detail */}
          {selected && (
            <div className="card">
              <div className="label" style={{ marginBottom: 14 }}>Run Detail — {selected.run_name}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Metrics</div>
                  {Object.entries(selected.metrics).map(([k, v]) => (
                    <div key={k} style={{
                      display: 'flex', justifyContent: 'space-between',
                      padding: '5px 0', borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ fontSize: 11, fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {typeof v === 'number' ? v.toFixed(4) : v}
                      </span>
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Parameters</div>
                  {Object.entries(selected.params).map(([k, v]) => (
                    <div key={k} style={{
                      display: 'flex', justifyContent: 'space-between',
                      padding: '5px 0', borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                      <span style={{ fontSize: 11, color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Model comparison chart */}
          {runs.length >= 2 && (
            <div className="card">
              <div className="label" style={{ marginBottom: 14 }}>Model Comparison — XGBoost vs SVM</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={comparisonData} barGap={4}>
                  <XAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0.8, 1.0]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 4, fontSize: 11 }}
                    formatter={(v, name) => [v?.toFixed(4), name]}
                  />
                  {runs.map(r => (
                    <Bar key={r.run_id} dataKey={r.model_type} fill={MODEL_COLORS[r.model_type]} radius={[3,3,0,0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}