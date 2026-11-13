// ============================================================================
// LiveMonitor.jsx — Real-time webcam monitoring page
// ============================================================================
//
// CHANGES FROM PREV VERSION:
//
//   BUG FIX 1 — Replaced createWebSocket + inline Promise pattern with
//     the new openWebSocket() from api.js. This eliminates ALL race conditions
//     and gives a clear error message if WS fails or times out.
//
//   BUG FIX 2 — Removed the old commented-out Promise block so there's no
//     confusion about which code is active.
//
//   BUG FIX 3 — Import now includes openWebSocket instead of createWebSocket.
//
// ============================================================================

import { useEffect, useRef, useState, useCallback } from 'react'
import { Camera, CameraOff, AlertTriangle, CheckCircle, Volume2, VolumeX } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { useWebcam } from '../hooks/useWebcam'
import { useSessionStore } from '../store/sessionStore'
import { openWebSocket, getDrift } from '../services/api'
import GaugeChart from '../components/charts/GaugeChart'

const MAX_WIDTH = 640  // downscale before JPEG encoding (keeps frames well under the server's size limit)

// Display for every backend state. Errors and a missing face are never shown as "alert".
const STATE_UI = {
  idle:       { label: 'STANDBY',           color: 'var(--text-muted)' },
  warming_up: { label: 'CALIBRATING',       color: 'var(--text-muted)' },
  alert:      { label: 'ALERT',             color: 'var(--green)' },
  drowsy:     { label: 'DROWSY',            color: 'var(--red)' },
  no_face:    { label: 'NO FACE — CHECK',   color: 'var(--amber)' },
  degraded:   { label: 'LOW QUALITY INPUT', color: 'var(--amber)' },
  error:      { label: 'SYSTEM ERROR',      color: 'var(--amber)' },
}

export default function LiveMonitor() {
  const { videoRef, canvasRef, isOn, error: camError, start: startCam, stop: stopCam } = useWebcam()
  const store = useSessionStore()
  const intervalRef = useRef(null)
  const [sound, setSound] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const alertedRef = useRef(false)

  const isDrowsy   = store.currentState === 'drowsy'
  const driftTimerRef = useRef(null)

  // Play audio alert on drowsy
  useEffect(() => {
    if (isDrowsy && sound && !alertedRef.current) {
      alertedRef.current = true
      try {
        const ctx  = new AudioContext()
        const osc  = ctx.createOscillator()
        const gain = ctx.createGain()
        osc.connect(gain); gain.connect(ctx.destination)
        osc.frequency.setValueAtTime(880, ctx.currentTime)
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3)
        gain.gain.setValueAtTime(0.4, ctx.currentTime)
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6)
        osc.start(); osc.stop(ctx.currentTime + 0.6)
      } catch { /* AudioContext may be blocked by browser policy */ }
    }
    if (!isDrowsy) alertedRef.current = false
  }, [isDrowsy, sound])

  const handleStart = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // 1. Camera first — if this fails we never touch the backend
      await startCam()
      store.reset()

      // 2. WebSocket. Backpressure: at most ONE frame in flight; the server answers every frame.
      let ready = false, inFlight = false, sentAt = 0
      const onMessage = (msg) => {
        if (msg.type === 'init_ack') ready = true
        if ((msg.type === 'status' || msg.type === 'error') && inFlight) {
          inFlight = false
          store.recordLatency(performance.now() - sentAt)
        }
        store.handleWsMessage(msg)
      }
      const ws = await openWebSocket(onMessage, () => { if (useSessionStore.getState().isActive) handleStop() })
      ws.send(JSON.stringify({ type: 'init' }))  // the server decides window length and fps
      store.setWs(ws)

      // 3. Capture loop at the server's target fps, with a monotonic timestamp per frame.
      const tick = () => {
        if (ws.readyState !== WebSocket.OPEN || !ready || inFlight) return
        const video = videoRef.current, canvas = canvasRef.current
        if (!video || !canvas || video.readyState !== 4) return
        const scale = Math.min(1, MAX_WIDTH / (video.videoWidth || MAX_WIDTH))
        canvas.width = Math.round((video.videoWidth || 640) * scale)
        canvas.height = Math.round((video.videoHeight || 480) * scale)
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
        const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
        inFlight = true
        sentAt = performance.now()
        ws.send(JSON.stringify({ type: 'frame', ts: sentAt, image_b64: b64 }))
      }
      const fps = useSessionStore.getState().targetFps || 10
      intervalRef.current = setInterval(tick, Math.round(1000 / fps))
      driftTimerRef.current = setInterval(() => getDrift().then(store.setDrift).catch(() => {}), 5000)

    } catch (e) {
      setError(e.message)
      stopCam()
      store.clearSession()
    } finally {
      setLoading(false)
    }
  }, [startCam, stopCam, store])

  const handleStop = useCallback(async () => {
    clearInterval(intervalRef.current)
    clearInterval(driftTimerRef.current)
    const ws = useSessionStore.getState().ws
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'close' }))  // server replies with a summary and closes the session
      ws.close()
    }
    store.setWs(null)
    stopCam()
    store.clearSession()
  }, [store, stopCam])

  // Cleanup on unmount
  useEffect(() => () => {
    clearInterval(intervalRef.current)
    clearInterval(driftTimerRef.current)
    useSessionStore.getState().ws?.close()
  }, [])

  const ui = STATE_UI[store.currentState] || STATE_UI.idle
  const stateColor = ui.color
  const stateLabel = store.currentState === 'warming_up'
    ? `CALIBRATING ${Math.round((store.warmup || 0) * 100)}%`
    : ui.label

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22, letterSpacing: '0.05em' }}>
            Live Monitor
          </h1>
          <div className="label" style={{ marginTop: 4 }}>Real-time drowsiness detection via webcam</div>
        </div>
        <button onClick={() => setSound(s => !s)} style={{
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius)', padding: '8px 14px',
          color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', gap: 6,
        }}>
          {sound ? <Volume2 size={14} /> : <VolumeX size={14} />}
          <span style={{ fontSize: 11 }}>{sound ? 'Sound On' : 'Sound Off'}</span>
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 20 }}>

        {/* ── Left: Webcam + state ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Webcam */}
          <div className="card" style={{ padding: 0, overflow: 'hidden', position: 'relative', aspectRatio: '4/3' }}>
            <video ref={videoRef} autoPlay playsInline muted
              style={{
                width: '100%', height: '100%', objectFit: 'cover',
                transform: 'scaleX(-1)',
                display: isOn ? 'block' : 'none',
              }} />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Overlay when camera is off */}
            {!isOn && (
              <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                background: 'var(--bg-card)', gap: 12,
              }}>
                <CameraOff size={40} color="var(--text-muted)" />
                <div className="label">Camera inactive</div>
                {camError && (
                  <div style={{ color: 'var(--red)', fontSize: 11, maxWidth: 280, textAlign: 'center' }}>
                    {camError}
                  </div>
                )}
              </div>
            )}

            {/* State badge overlay */}
            {isOn && (
              <div style={{
                position: 'absolute', top: 16, left: 16,
                padding: '6px 14px',
                background: isDrowsy ? 'rgba(255,59,59,0.25)' : 'rgba(0,0,0,0.6)',
                backdropFilter: 'blur(8px)',
                border: `1px solid ${stateColor}`,
                borderRadius: 999,
                fontFamily: 'var(--font-display)', fontWeight: 700,
                fontSize: 12, letterSpacing: '0.15em',
                color: stateColor,
                animation: isDrowsy ? 'blink 1s ease infinite' : 'none',
              }}>
                {stateLabel}
              </div>
            )}

            {/* Frame counter */}
            {isOn && (
              <div style={{
                position: 'absolute', bottom: 16, right: 16,
                fontSize: 10, fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)', background: 'rgba(0,0,0,0.6)',
                padding: '3px 8px', borderRadius: 'var(--radius)',
              }}>
                {store.framesProcessed.toLocaleString()} frames
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: 12 }}>
            {!store.isActive ? (
              <button onClick={handleStart} disabled={loading} style={{
                flex: 1, padding: '13px',
                background: 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05))',
                border: '1px solid rgba(0,212,255,0.4)', borderRadius: 'var(--radius)',
                color: 'var(--cyan)', fontFamily: 'var(--font-display)', fontWeight: 700,
                fontSize: 13, letterSpacing: '0.1em', cursor: loading ? 'wait' : 'pointer',
                transition: 'all var(--transition)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              }}>
                <Camera size={15} />
                {loading ? 'STARTING...' : 'START MONITORING'}
              </button>
            ) : (
              <button onClick={handleStop} style={{
                flex: 1, padding: '13px',
                background: 'rgba(255,59,59,0.08)',
                border: '1px solid rgba(255,59,59,0.3)', borderRadius: 'var(--radius)',
                color: 'var(--red)', fontFamily: 'var(--font-display)', fontWeight: 700,
                fontSize: 13, letterSpacing: '0.1em', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              }}>
                <CameraOff size={15} />
                STOP SESSION
              </button>
            )}
          </div>

          {error && (
            <div style={{
              padding: '10px 14px',
              background: 'rgba(255,59,59,0.1)',
              border: '1px solid rgba(255,59,59,0.3)',
              borderRadius: 'var(--radius)',
              color: 'var(--red)', fontSize: 12,
            }}>
              {error}
            </div>
          )}

          {/* EAR Time Series */}
          <div className="card">
            <div className="label" style={{ marginBottom: 12 }}>Eye Aspect Ratio — Live</div>
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={store.earHistory}>
                <XAxis dataKey="t" hide />
                <YAxis domain={[0, 0.5]} hide />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 4, fontSize: 11 }}
                  formatter={(v) => [v?.toFixed(3), 'EAR']}
                  labelFormatter={() => ''}
                />
                <ReferenceLine y={0.2} stroke="rgba(255,59,59,0.5)" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="v" dot={false} strokeWidth={2}
                  stroke="var(--cyan)" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              Window-mean EAR per prediction · red line = eye-closed threshold (EAR &lt; 0.20)
            </div>
          </div>
        </div>

        {/* ── Right: Gauges + alerts ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Gauges */}
          <div className="card">
            <div className="label" style={{ marginBottom: 16 }}>Feature Readings</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, justifyItems: 'center' }}>
              <GaugeChart
                value={store.currentEAR ?? 0}
                min={0} max={0.5}
                label="EAR" unit="ratio"
                danger={store.currentEAR !== null && store.currentEAR < 0.25}
                size={130}
              />
              <GaugeChart
                value={store.currentPERCLOS ?? 0}
                min={0} max={1}
                label="PERCLOS" unit="%"
                danger={store.currentPERCLOS !== null && store.currentPERCLOS > 0.8}
                warning={store.currentPERCLOS !== null && store.currentPERCLOS > 0.5}
                size={130}
              />
              <GaugeChart
                value={Math.max(0, Math.min(1, store.currentScore ?? 0))}
                min={0} max={1}
                label="Model score" unit={store.threshold != null ? `thr ${store.threshold.toFixed(2)}` : 'score'}
                danger={store.currentState === 'drowsy'}
                size={130}
              />
              <GaugeChart
                value={store.currentMAR ?? 0}
                min={0} max={1}
                label="MAR" unit="ratio"
                warning={store.currentMAR !== null && store.currentMAR > 0.6}
                size={130}
              />
            </div>
          </div>

          {/* Session stats */}
          <div className="card">
            <div className="label" style={{ marginBottom: 12 }}>Session Stats</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                { label: 'Frames',      value: store.framesProcessed.toLocaleString() },
                { label: 'Alerts',      value: store.alertsTriggered, highlight: store.alertsTriggered > 0 },
                { label: 'Drift (PSI)', value: store.overallDriftScore >= 0 ? store.overallDriftScore.toFixed(3) : 'n/a' },
                { label: 'Input FPS',   value: (store.fps ?? 0).toFixed(1), highlight: store.currentState === 'degraded' },
              ].map(({ label, value, highlight }) => (
                <div key={label} style={{
                  background: 'var(--bg-elevated)', padding: '10px 12px',
                  borderRadius: 'var(--radius)', border: '1px solid var(--border)',
                }}>
                  <div className="label" style={{ marginBottom: 4 }}>{label}</div>
                  <div style={{
                    fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18,
                    color: highlight ? 'var(--red)' : 'var(--text-primary)',
                  }}>{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Alert log */}
          <div className="card" style={{ flex: 1 }}>
            <div className="label" style={{ marginBottom: 12 }}>Alert Log</div>
            {store.alertLog.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
                <CheckCircle size={14} color="var(--green)" />
                No alerts — driver is alert
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 200, overflowY: 'auto' }}>
                {store.alertLog.map(alert => (
                  <div key={alert.id} style={{
                    padding: '8px 12px', borderRadius: 'var(--radius)',
                    background: 'rgba(255,59,59,0.08)',
                    border: '1px solid rgba(255,59,59,0.2)',
                    animation: 'fadeSlideUp 0.3s ease',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--red)', fontSize: 11 }}>
                        <AlertTriangle size={11} /> DROWSY
                      </span>
                      <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>{alert.time}</span>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                      Model score: {alert.score?.toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
