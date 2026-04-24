import { useEffect, useState } from 'react'
import { ExternalLink, Trophy, Cpu, TrendingUp } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts'

const MLFLOW = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000'

// Mock MLflow data — replaced by real API calls when MLflow is running
const MOCK_EXPERIMENTS = [
  {
    run_id: 'abc123',
    run_name: 'xgboost_training',
    model_type: 'XGBoost',
    status: 'FINISHED',
    start_time: new Date(Date.now() - 7200000).toISOString(),
    metrics: { f1_weighted: 0.912, accuracy: 0.924, auc_roc: 0.963, latency_p95_ms: 4.2 },
    params: { n_estimators: '200', max_depth: '6', learning_rate: '0.1' },
    stage: 'Production',
  },
  {
    run_id: 'def456',
    run_name: 'svm_training',
    model_type: 'SVM',
    status: 'FINISHED',
    start_time: new Date(Date.now() - 3600000).toISOString(),
    metrics: { f1_weighted: 0.883, accuracy: 0.891, auc_roc: 0.941, latency_p95_ms: 22.7 },
    params: { kernel: 'rbf', C: '1.0', gamma: 'scale' },
    stage: 'Archived',
  },
  {
    run_id: 'ghi789',
    run_name: 'model_evaluation',
    model_type: 'Evaluation',
    status: 'FINISHED',
    start_time: new Date(Date.now() - 1800000).toISOString(),
    metrics: { xgb_f1_weighted: 0.912, svm_f1_weighted: 0.883, winner_f1: 0.912 },
    params: { winner: 'XGBoost', promoted_version: '1' },
    stage: null,
  },
]

async function fetchMLflowRuns() {
  try {
    const res = await fetch(`${MLFLOW}/api/2.0/mlflow/runs/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ experiment_ids: ['1'], max_results: 20 }),
    })
    if (!res.ok) throw new Error()
    const data = await res.json()
    return (data.runs || []).map(r => ({
      run_id: r.info.run_id,
      run_name: r.info.run_name || r.data?.tags?.['mlflow.runName'] || 'unnamed',
      model_type: r.data?.params?.model_type || 'Unknown',
      status: r.info.status,
      start_time: new Date(r.info.start_time).toISOString(),
      metrics: r.data?.metrics || {},
      params: r.data?.params || {},
      stage: null,
    }))
  } catch {
    return MOCK_EXPERIMENTS
  }
}

const stageColor = { Production: 'var(--green)', Staging: 'var(--amber)', Archived: 'var(--text-muted)' }

export default function ModelRegistry() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    fetchMLflowRuns().then(data => {
      setRuns(data)
      setSelected(data[0] || null)
      setLoading(false)
    })
  }, [])

  const trainingRuns = runs.filter(r => r.run_name !== 'model_evaluation')
  const winner = trainingRuns.find(r => r.stage === 'Production') || trainingRuns[0]

  // Comparison bar chart data
  const comparisonData = [
    { metric: 'F1 Score', ...Object.fromEntries(trainingRuns.map(r => [r.model_type, r.metrics.f1_weighted])) },
    { metric: 'Accuracy', ...Object.fromEntries(trainingRuns.map(r => [r.model_type, r.metrics.accuracy])) },
    { metric: 'AUC-ROC',  ...Object.fromEntries(trainingRuns.map(r => [r.model_type, r.metrics.auc_roc])) },
  ]

  const MODEL_COLORS = { XGBoost: 'var(--cyan)', SVM: 'var(--purple)', Evaluation: 'var(--amber)' }

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
      {winner && (
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
              F1: {winner.metrics.f1_weighted?.toFixed(4)} · AUC-ROC: {winner.metrics.auc_roc?.toFixed(4)} · P95 Latency: {winner.metrics.latency_p95_ms?.toFixed(1)}ms
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
                        background: `${stageColor[run.stage]}22`,
                        color: stageColor[run.stage],
                        border: `1px solid ${stageColor[run.stage]}44`,
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
          {trainingRuns.length >= 2 && (
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
                  {trainingRuns.map(r => (
                    <Bar key={r.model_type} dataKey={r.model_type} fill={MODEL_COLORS[r.model_type]} radius={[3,3,0,0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
                {trainingRuns.map(r => (
                  <div key={r.model_type} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: MODEL_COLORS[r.model_type] }} />
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{r.model_type}</span>
                    {r.stage === 'Production' && <span style={{ fontSize: 9, color: 'var(--green)' }}>★ Production</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Acceptance criteria */}
          <div className="card">
            <div className="label" style={{ marginBottom: 12 }}>Acceptance Criteria</div>
            {winner && [
              { label: 'F1 ≥ 0.85', pass: winner.metrics.f1_weighted >= 0.85, value: winner.metrics.f1_weighted?.toFixed(4) },
              { label: 'AUC-ROC ≥ 0.90', pass: winner.metrics.auc_roc >= 0.90, value: winner.metrics.auc_roc?.toFixed(4) },
              { label: 'Latency P95 ≤ 200ms', pass: winner.metrics.latency_p95_ms <= 200, value: `${winner.metrics.latency_p95_ms?.toFixed(1)}ms` },
            ].map(({ label, pass, value }) => (
              <div key={label} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '8px 0', borderBottom: '1px solid var(--border)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: pass ? 'var(--green)' : 'var(--red)' }} />
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
                </div>
                <span style={{
                  fontSize: 12, fontFamily: 'var(--font-display)', fontWeight: 700,
                  color: pass ? 'var(--green)' : 'var(--red)',
                }}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
