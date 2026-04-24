import { useEffect, useState } from 'react'
import { RefreshCw, CheckCircle, XCircle, Clock, Play, ChevronRight } from 'lucide-react'
import { getAirflowDagRuns, getAirflowTaskInstances } from '../services/api'

const DAG_STAGES = [
  { id: 't0_validate_raw_data',   label: 'Validate Raw Data',    tool: 'Python',    color: '#06B6D4' },
  { id: 't1_extract_frames',      label: 'Extract Frames',       tool: 'OpenCV',    color: '#06B6D4' },
  { id: 't2_extract_landmarks',   label: 'Extract Landmarks',    tool: 'MediaPipe', color: '#06B6D4' },
  { id: 't3_feature_engineering', label: 'Feature Engineering',  tool: 'PySpark',   color: '#10B981' },
  { id: 't4_split_data',          label: 'Split Data',           tool: 'sklearn',   color: '#10B981' },
  { id: 't5_dvc_add_and_push',    label: 'DVC Push',             tool: 'DVC',       color: '#A78BFA' },
  { id: 't6_pipeline_summary',    label: 'Pipeline Summary',     tool: 'Airflow',   color: '#A78BFA' },
]

const stateColor = { success: '#10B981', running: '#F59E0B', failed: '#FF3B3B', queued: '#7A9AB5', skipped: '#3A5068', upstream_failed: '#FF3B3B' }
const stateIcon = { success: CheckCircle, running: Clock, failed: XCircle, queued: Clock, skipped: Clock }

function StatusDot({ state, size = 10 }) {
  const color = stateColor[state] || '#3A5068'
  const Icon = stateIcon[state] || Clock
  return <Icon size={size} color={color} />
}

export default function PipelineConsole() {
  const [dagRuns, setDagRuns] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const load = async () => {
    setRefreshing(true)
    try {
      const data = await getAirflowDagRuns('ddd_data_pipeline')
      const runs = data.dag_runs || []
      setDagRuns(runs)
      const latest = runs[0]
      if (latest && !selectedRun) {
        setSelectedRun(latest)
        const taskData = await getAirflowTaskInstances('ddd_data_pipeline', latest.dag_run_id)
        setTasks(taskData.task_instances || [])
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => { load() }, [])

  const selectRun = async (run) => {
    setSelectedRun(run)
    try {
      const data = await getAirflowTaskInstances('ddd_data_pipeline', run.dag_run_id)
      setTasks(data.task_instances || [])
    } catch { setTasks([]) }
  }

  const taskMap = Object.fromEntries(tasks.map(t => [t.task_id, t]))

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, letterSpacing: '0.05em' }}>Pipeline Console</h1>
          <div className="label" style={{ marginTop: 4 }}>Airflow DAG — ddd_data_pipeline · DVC · PySpark</div>
        </div>
        <button onClick={load} disabled={refreshing} style={{
          display: 'flex', alignItems: 'center', gap: 7,
          padding: '8px 16px', background: 'var(--bg-elevated)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius)',
          color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 11,
        }}>
          <RefreshCw size={13} style={{ animation: refreshing ? 'sweep 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      {/* ── DAG Visual ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="label" style={{ marginBottom: 16 }}>Pipeline DAG — Task Flow</div>

        {/* Horizontal flow */}
        <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 0, minWidth: 'max-content' }}>
            {DAG_STAGES.map((stage, i) => {
              const task = taskMap[stage.id]
              const state = task?.state || 'queued'
              const color = stateColor[state] || '#3A5068'
              const isLast = i === DAG_STAGES.length - 1
              return (
                <div key={stage.id} style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    gap: 8, padding: '12px 14px',
                    background: state === 'running'
                      ? `rgba(245,158,11,0.08)` : `rgba(${color.includes('#10') ? '16,185,129' : color.includes('#06') ? '6,182,212' : color.includes('#A7') ? '139,92,246' : '255,59,59'},0.06)`,
                    border: `1px solid ${color}44`,
                    borderRadius: 'var(--radius)',
                    minWidth: 110,
                    transition: 'all 0.3s ease',
                  }}>
                    <StatusDot state={state} size={14} />
                    <div style={{
                      fontSize: 11, fontFamily: 'var(--font-mono)',
                      color: state === 'success' ? 'var(--text-primary)' : 'var(--text-secondary)',
                      textAlign: 'center', lineHeight: 1.3,
                    }}>
                      {stage.label}
                    </div>
                    <div style={{
                      fontSize: 9, padding: '2px 6px',
                      background: `${stage.color}22`,
                      border: `1px solid ${stage.color}44`,
                      borderRadius: 999, color: stage.color,
                      letterSpacing: '0.1em',
                    }}>
                      {stage.tool}
                    </div>
                    {task?.duration && (
                      <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                        {task.duration.toFixed(1)}s
                      </div>
                    )}
                  </div>
                  {!isLast && (
                    <ChevronRight size={16} color="var(--border-bright)" style={{ margin: '0 4px', flexShrink: 0 }} />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 20 }}>

        {/* ── Run history ── */}
        <div className="card" style={{ height: 'fit-content' }}>
          <div className="label" style={{ marginBottom: 12 }}>Recent Runs</div>
          {loading ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading...</div>
          ) : dagRuns.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              No runs found.<br />Trigger a run from Airflow UI at<br />
              <span style={{ color: 'var(--cyan)' }}>localhost:8080</span>
            </div>
          ) : (
            dagRuns.map(run => {
              const isSelected = selectedRun?.dag_run_id === run.dag_run_id
              const color = stateColor[run.state] || 'var(--text-muted)'
              return (
                <div key={run.dag_run_id} onClick={() => selectRun(run)}
                  style={{
                    padding: '10px 12px', borderRadius: 'var(--radius)',
                    marginBottom: 6, cursor: 'pointer',
                    background: isSelected ? 'rgba(0,212,255,0.07)' : 'var(--bg-elevated)',
                    border: `1px solid ${isSelected ? 'rgba(0,212,255,0.25)' : 'var(--border)'}`,
                    transition: 'all var(--transition)',
                  }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <StatusDot state={run.state} size={12} />
                    <span style={{ fontSize: 9, color, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{run.state}</span>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(run.execution_date).toLocaleDateString()}
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                    {run.dag_run_id.split('__').pop()?.slice(0, 16)}
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* ── Task detail ── */}
        <div className="card">
          <div className="label" style={{ marginBottom: 16 }}>
            Task Details — {selectedRun?.dag_run_id || 'Select a run'}
          </div>
          {tasks.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No task data available.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {DAG_STAGES.map(stage => {
                const task = taskMap[stage.id]
                const state = task?.state || 'queued'
                const color = stateColor[state] || '#3A5068'
                return (
                  <div key={stage.id} style={{
                    display: 'flex', alignItems: 'center', gap: 14,
                    padding: '10px 14px', borderRadius: 'var(--radius)',
                    background: 'var(--bg-elevated)',
                    border: `1px solid ${state === 'success' ? '#10B98122' : state === 'failed' ? '#FF3B3B22' : 'var(--border)'}`,
                  }}>
                    <StatusDot state={state} size={13} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 12, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {stage.label}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>
                        {stage.id}
                      </div>
                    </div>
                    <div style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 999,
                      background: `${color}22`, color, border: `1px solid ${color}44`,
                      letterSpacing: '0.1em', textTransform: 'uppercase',
                    }}>
                      {state}
                    </div>
                    {task?.duration && (
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', minWidth: 50, textAlign: 'right' }}>
                        {task.duration.toFixed(1)}s
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {/* Links to external UIs */}
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border)', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[
              { label: 'Open Airflow UI', url: 'http://localhost:8080', color: 'var(--amber)' },
              { label: 'View DVC DAG', url: '#', color: 'var(--cyan)' },
            ].map(({ label, url, color }) => (
              <a key={label} href={url} target="_blank" rel="noreferrer" style={{
                padding: '7px 14px', background: 'var(--bg-base)',
                border: `1px solid ${color}44`, borderRadius: 'var(--radius)',
                color, fontSize: 11, textDecoration: 'none',
                fontFamily: 'var(--font-mono)', letterSpacing: '0.05em',
                transition: 'all var(--transition)',
              }}>
                {label} ↗
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
