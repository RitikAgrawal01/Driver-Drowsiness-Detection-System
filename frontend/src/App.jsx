import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/ui/Layout'
import LiveMonitor from './pages/LiveMonitor'
import PipelineConsole from './pages/PipelineConsole'
import MonitoringDashboard from './pages/MonitoringDashboard'
import ModelRegistry from './pages/ModelRegistry'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<LiveMonitor />} />
          <Route path="pipeline" element={<PipelineConsole />} />
          <Route path="monitoring" element={<MonitoringDashboard />} />
          <Route path="models" element={<ModelRegistry />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
