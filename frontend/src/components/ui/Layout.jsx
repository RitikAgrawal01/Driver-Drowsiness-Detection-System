import { NavLink, Outlet } from 'react-router-dom'
import { Activity, LayoutDashboard, Cpu, GitBranch } from 'lucide-react'
import { useSessionStore } from '../../store/sessionStore'

const NAV = [
  { to: '/',          icon: Activity,       label: 'Live Monitor'  },
  { to: '/pipeline',  icon: GitBranch,      label: 'Pipeline'      },
  { to: '/monitoring',icon: LayoutDashboard, label: 'Monitoring'   },
  { to: '/models',    icon: Cpu,            label: 'Model Registry'},
]

export default function Layout() {
  const { isActive, currentState, alertsTriggered } = useSessionStore()
  const isDrowsy = isActive && currentState === 'drowsy'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-base)' }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 220, flexShrink: 0,
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        position: 'sticky', top: 0, height: '100vh',
      }}>

        {/* Logo */}
        <div style={{ padding: '24px 20px 20px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 34, height: 34,
              background: isDrowsy
                ? 'radial-gradient(circle, rgba(255,59,59,0.4) 0%, transparent 70%)'
                : 'radial-gradient(circle, rgba(0,212,255,0.3) 0%, transparent 70%)',
              border: `1px solid ${isDrowsy ? 'var(--red)' : 'var(--cyan)'}`,
              borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.4s ease',
              flexShrink: 0,
            }}>
              <div style={{
                width: 10, height: 10, borderRadius: '50%',
                background: isDrowsy ? 'var(--red)' : 'var(--cyan)',
                boxShadow: isDrowsy
                  ? '0 0 8px var(--red)'
                  : '0 0 8px var(--cyan)',
                animation: isActive ? 'blink 1.5s ease infinite' : 'none',
              }} />
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 13, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>DDD</div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', letterSpacing: '0.15em' }}>DROWSINESS SYSTEM</div>
            </div>
          </div>

          {/* System status pill */}
          <div style={{ marginTop: 14 }}>
            <span className={`badge ${isActive ? (isDrowsy ? 'badge-red' : 'badge-green') : 'badge-cyan'}`}>
              <span style={{
                width: 5, height: 5, borderRadius: '50%',
                background: 'currentColor',
                animation: isActive ? 'blink 1.2s ease infinite' : 'none',
              }} />
              {isActive ? (isDrowsy ? 'DROWSY DETECTED' : 'MONITORING') : 'STANDBY'}
            </span>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: '16px 12px', flex: 1 }}>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} end={to === '/'} style={({ isActive }) => ({
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '10px 12px', borderRadius: 'var(--radius)',
              marginBottom: 4,
              textDecoration: 'none',
              transition: 'all var(--transition)',
              background: isActive ? 'rgba(0,212,255,0.08)' : 'transparent',
              border: `1px solid ${isActive ? 'rgba(0,212,255,0.2)' : 'transparent'}`,
              color: isActive ? 'var(--cyan)' : 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11, letterSpacing: '0.08em',
            })}>
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Alert counter */}
        {isActive && (
          <div style={{
            margin: '0 12px 16px',
            padding: '12px',
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            border: `1px solid ${alertsTriggered > 0 ? 'rgba(255,59,59,0.3)' : 'var(--border)'}`,
          }}>
            <div className="label" style={{ marginBottom: 4 }}>Session Alerts</div>
            <div style={{
              fontFamily: 'var(--font-display)', fontWeight: 800,
              fontSize: 28, color: alertsTriggered > 0 ? 'var(--red)' : 'var(--text-secondary)',
              lineHeight: 1,
            }}>{alertsTriggered}</div>
          </div>
        )}

        {/* Footer */}
        <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
            MLOps Pipeline v1.0<br />
            MediaPipe · XGBoost · FastAPI
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main style={{ flex: 1, overflow: 'auto', minWidth: 0 }}>
        {/* Drowsy alert banner */}
        {isDrowsy && (
          <div style={{
            background: 'linear-gradient(90deg, rgba(255,59,59,0.2), rgba(255,59,59,0.05))',
            borderBottom: '1px solid rgba(255,59,59,0.4)',
            padding: '10px 24px',
            display: 'flex', alignItems: 'center', gap: 12,
            animation: 'fadeSlideUp 0.3s ease',
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: 'var(--red)', boxShadow: '0 0 12px var(--red)',
              animation: 'blink 0.8s ease infinite', flexShrink: 0,
            }} />
            <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--red)', fontSize: 13, letterSpacing: '0.1em' }}>
              DROWSINESS DETECTED — Please take a break immediately
            </span>
          </div>
        )}
        <Outlet />
      </main>
    </div>
  )
}
