import { useState } from "react";

const phases = [
  {
    id: 0,
    title: "Project Scaffold",
    subtitle: "Folder Structure & Environment Setup",
    color: "#00D4FF",
    accent: "#003D4D",
    duration: "Day 1",
    score: "Foundation for all points",
    tasks: [
      "Create monorepo folder structure: /frontend, /backend, /model_server, /airflow, /monitoring, /data, /docs, /tests",
      "Initialize Git repository + .gitignore + Git LFS config",
      "Install DVC and initialize: dvc init",
      "Create root docker-compose.yml skeleton (5 services: frontend, backend, model_server, prometheus, grafana)",
      "Create requirements.txt / pyproject.toml for each service",
      "Set up Python virtual environments per service",
      "Install core tools: FastAPI, MLflow, DVC, Airflow, PySpark, MediaPipe, XGBoost, scikit-learn, Prometheus client",
      "Create .env template file for all configurable parameters",
    ],
    deliverable: "Git repo initialized, folder structure ready, all dependencies installable",
  },
  {
    id: 1,
    title: "Architecture & Design Docs",
    subtitle: "HLD, LLD, Architecture Diagram",
    color: "#FFB800",
    accent: "#3D2D00",
    duration: "Day 1–2",
    score: "Software Engineering: Design Principle [2 pts]",
    tasks: [
      "Draw Architecture Diagram: React ↔ FastAPI ↔ Model Server ↔ MLflow ↔ Prometheus/Grafana",
      "Write HLD document: system overview, component responsibilities, data flow narrative",
      "Write LLD document: all REST API endpoints with request/response JSON schemas",
      "Define API contracts: /predict, /health, /ready, /metrics, /start-session, /stop-session",
      "Document design decisions: why XGBoost+SVM, why MediaPipe, why loose coupling",
      "Create UML-style component diagram",
    ],
    deliverable: "docs/architecture.png, docs/HLD.md, docs/LLD.md with full API specs",
  },
  {
    id: 2,
    title: "Data Pipeline",
    subtitle: "Airflow + PySpark Data Engineering",
    color: "#FF6B35",
    accent: "#3D1800",
    duration: "Day 2–4",
    score: "MLOps: Data Engineering [2 pts]",
    tasks: [
      "Download/prepare dataset: UTA-RLDD or custom recorded video clips (drowsy vs alert)",
      "Write Airflow DAG: dag_data_pipeline.py with tasks for frame extraction → landmark processing → feature engineering",
      "Task 1 (Airflow): Extract frames from video at configurable FPS using OpenCV",
      "Task 2 (Airflow): Run MediaPipe on each frame → extract 468 landmarks → save as CSV",
      "Task 3 (PySpark): Compute EAR, PERCLOS, MAR, Head Pose over sliding window → feature CSV",
      "Task 4 (Airflow): DVC add data/raw, data/landmarks, data/features → dvc push",
      "Compute drift baselines: mean/variance/distribution of all features (save as baseline.json)",
      "Implement data validation: schema checks, missing value checks as an Airflow task",
      "Measure and log pipeline throughput (frames/sec, time per stage)",
    ],
    deliverable: "Working Airflow DAG visible in UI, DVC-versioned feature CSV, baseline.json",
  },
  {
    id: 3,
    title: "Source Control & CI",
    subtitle: "DVC DAG, Git LFS, Versioning",
    color: "#A855F7",
    accent: "#2D0050",
    duration: "Day 4–5",
    score: "MLOps: Source Control & CI [2 pts]",
    tasks: [
      "Configure DVC remote (local path or NFS): dvc remote add",
      "Create DVC pipeline stages in dvc.yaml: data_ingest → feature_eng → train → evaluate",
      "Run dvc dag to visualize and screenshot the DAG for documentation",
      "Use Git LFS for large files (videos, model weights): git lfs track '*.h5' '*.pkl'",
      "Version raw data, features, and models at every stage with dvc add + dvc push",
      "Configure GitHub Actions (or local pre-commit hook) to run dvc repro on push",
      "Ensure every experiment is reproducible via Git commit hash + MLflow run ID",
      "Write CI workflow: lint → unit tests → dvc repro → mlflow log",
    ],
    deliverable: "dvc.yaml with full pipeline, DVC DAG screenshot, CI workflow file",
  },
  {
    id: 4,
    title: "Model Development",
    subtitle: "XGBoost & SVM with MLflow Tracking",
    color: "#10B981",
    accent: "#003D28",
    duration: "Day 5–7",
    score: "MLOps: Experiment Tracking [2 pts]",
    tasks: [
      "Create model training scripts: train_xgboost.py, train_svm.py",
      "Implement MLflow experiment tracking: mlflow.start_run() wrapping all training",
      "Log parameters: hyperparameters, feature window size, model type",
      "Log metrics: accuracy, F1-score, precision, recall, AUC-ROC, inference latency",
      "Log artifacts: trained model (.pkl), confusion matrix plot, feature importance plot",
      "Implement MLflow Autolog + manual logging for additional info (dataset hash, DVC commit)",
      "Use MLflow Model Registry to register best model as 'production'",
      "Optimize models for local hardware: XGBoost tree depth tuning, SVM kernel selection",
      "Compare models in MLflow UI → document best model choice with rationale",
      "Create evaluate.py that loads from MLflow registry and outputs evaluation report",
    ],
    deliverable: "MLflow UI showing 2+ experiments, registered production model, evaluation report",
  },
  {
    id: 5,
    title: "Model Server & FastAPI Backend",
    subtitle: "Model Serving + REST API",
    color: "#F43F5E",
    accent: "#3D0012",
    duration: "Day 7–9",
    score: "MLOps: Software Packaging [4 pts] + Testing [1 pt]",
    tasks: [
      "Create model_server/: load model from MLflow registry, expose /predict endpoint",
      "Input: JSON with EAR, PERCLOS, MAR, head_pose features; Output: {state, confidence}",
      "Implement /health and /ready endpoints in model server",
      "Create backend/main.py with FastAPI: endpoints for /start-session, /stop-session, /predict, /status",
      "Backend receives webcam frames → runs MediaPipe → computes features → calls model server",
      "Implement WebSocket endpoint for real-time frame streaming from frontend",
      "Add comprehensive exception handling + logging (Python logging module) throughout",
      "Follow PEP8 / Black formatting for all Python code",
      "Add inline code comments on every non-trivial function",
      "Write unit tests: test_features.py (EAR/MAR calc), test_api.py (endpoint responses)",
      "Create test plan, test cases doc, and test report with pass/fail counts",
      "Define acceptance criteria: latency < 200ms, F1 > 0.85",
    ],
    deliverable: "FastAPI docs at /docs, model server running, unit tests passing, test report",
  },
  {
    id: 6,
    title: "Monitoring Stack",
    subtitle: "Prometheus Instrumentation + Grafana",
    color: "#F97316",
    accent: "#3D1500",
    duration: "Day 9–10",
    score: "MLOps: Exporter Instrumentation & Visualization [2 pts]",
    tasks: [
      "Add prometheus_client to FastAPI backend: expose /metrics endpoint",
      "Instrument inference latency: Histogram('inference_latency_seconds')",
      "Instrument prediction confidence distribution: Histogram('prediction_confidence')",
      "Instrument request count, error rate, active sessions: Counter + Gauge",
      "Instrument data drift: compare incoming feature stats vs baseline.json → Gauge('feature_drift_score')",
      "Configure prometheus.yml to scrape backend /metrics and model_server /metrics",
      "Build Grafana dashboard: inference latency panel, confidence distribution, drift score, error rate",
      "Configure Prometheus alert rules: error_rate > 5%, drift_score > threshold",
      "Add monitoring for Airflow pipeline: task success/failure metrics",
      "Screenshot all Grafana dashboards for documentation",
    ],
    deliverable: "Grafana dashboard live with 4+ panels, Prometheus scraping all services, alerts configured",
  },
  {
    id: 7,
    title: "React Frontend",
    subtitle: "Web UI + ML Pipeline Visualization",
    color: "#06B6D4",
    accent: "#003040",
    duration: "Day 10–12",
    score: "Demonstration: UI/UX [6 pts] + Pipeline Viz [4 pts]",
    tasks: [
      "Bootstrap React app with Vite: npm create vite@latest frontend -- --template react",
      "Page 1 — Live Monitor: webcam feed, EAR/MAR gauges, drowsiness alert banner, session controls",
      "Page 2 — Pipeline Console: Airflow DAG visualization (embed Airflow UI or custom viz), pipeline run history, task status",
      "Page 3 — Monitoring Dashboard: embed Grafana panels via iframe OR recreate charts with recharts",
      "Page 4 — Model Registry: list MLflow experiments, compare metrics, current production model info",
      "Design: dark automotive theme, red alert colors, clean gauges — intuitive for non-technical users",
      "Implement WebSocket client to receive real-time predictions from backend",
      "Add responsive design: works on desktop and tablet",
      "Add audio alert for drowsiness detection",
      "Error states: camera not found, backend offline → friendly messages",
      "Write User Manual: docs/user_manual.md with screenshots for non-technical users",
    ],
    deliverable: "React app running on port 3000, all 4 pages functional, user manual written",
  },
  {
    id: 8,
    title: "Dockerization",
    subtitle: "Multi-Container Docker Compose",
    color: "#8B5CF6",
    accent: "#200050",
    duration: "Day 12–13",
    score: "MLOps: Software Packaging [4 pts] — Docker + Compose",
    tasks: [
      "Write Dockerfile for backend (FastAPI): multi-stage build, non-root user",
      "Write Dockerfile for model_server: copies MLflow model artifacts into image",
      "Write Dockerfile for frontend (React): nginx-based production build",
      "Write Dockerfile for Airflow (custom image with PySpark + MediaPipe)",
      "Update docker-compose.yml: 5 services — frontend, backend, model_server, prometheus, grafana",
      "Add healthcheck directives to each service in docker-compose.yml",
      "Use named volumes for MLflow artifacts, DVC cache, Airflow logs, Prometheus data",
      "Use environment variables from .env file for all configurable values",
      "Test full stack: docker-compose up --build → verify all services healthy",
      "Confirm frontend → backend → model_server chain works end-to-end in containers",
    ],
    deliverable: "docker-compose up brings everything live; all containers healthy",
  },
  {
    id: 9,
    title: "Automated Retraining",
    subtitle: "Performance Decay Trigger Pipeline",
    color: "#EC4899",
    accent: "#3D0025",
    duration: "Day 13–14",
    score: "Bonus robustness + Viva defense points",
    tasks: [
      "Create Airflow DAG: dag_retrain.py triggered when Prometheus drift/error alert fires",
      "Retraining pipeline: pull latest data (DVC) → retrain → evaluate → compare vs production",
      "Auto-promote to MLflow registry if new model beats production F1 by > 2%",
      "Implement feedback loop logging: log ground truth labels when available",
      "Test retraining trigger manually and document the flow",
    ],
    deliverable: "Retraining DAG visible in Airflow, documented trigger logic",
  },
  {
    id: 10,
    title: "Documentation & Final Polish",
    subtitle: "All Docs + Demo Prep",
    color: "#14B8A6",
    accent: "#003330",
    duration: "Day 14–15",
    score: "Viva [8 pts] + Documentation requirements",
    tasks: [
      "Finalize docs/architecture.png with all components labeled",
      "Finalize docs/HLD.md: design choices, rationale, component interactions",
      "Finalize docs/LLD.md: every API endpoint with exact JSON I/O schemas",
      "Finalize docs/test_plan.md + docs/test_report.md with pass/fail count",
      "Finalize docs/user_manual.md with annotated screenshots",
      "Prepare viva answers: be ready to explain every tool choice (why Airflow, why DVC, why MLflow)",
      "Prepare demo script: 5-min walkthrough covering UI → pipeline viz → monitoring → model registry",
      "Run full end-to-end test: start docker-compose, open browser, run drowsiness detection session",
      "List any incomplete items honestly with plausible explanation (for viva)",
      "Git tag final release: git tag v1.0.0",
    ],
    deliverable: "All 5 required docs complete, demo rehearsed, repo tagged",
  },
];

const scoreBreakdown = [
  { category: "Web App UI/UX", max: 6, covered: "Phase 7" },
  { category: "ML Pipeline Visualization", max: 4, covered: "Phase 7" },
  { category: "Design Principles", max: 2, covered: "Phase 1" },
  { category: "Implementation", max: 2, covered: "Phases 5, 7" },
  { category: "Testing", max: 1, covered: "Phase 5" },
  { category: "Data Engineering", max: 2, covered: "Phase 2" },
  { category: "Source Control & CI", max: 2, covered: "Phase 3" },
  { category: "Experiment Tracking", max: 2, covered: "Phase 4" },
  { category: "Monitoring & Visualization", max: 2, covered: "Phase 6" },
  { category: "Software Packaging", max: 4, covered: "Phases 5, 8" },
  { category: "Viva", max: 8, covered: "Phase 10" },
];

const totalMax = scoreBreakdown.reduce((s, i) => s + i.max, 0);

export default function Roadmap() {
  const [activePhase, setActivePhase] = useState(null);
  const [tab, setTab] = useState("roadmap");

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0A0C10",
      fontFamily: "'Courier New', 'Lucida Console', monospace",
      color: "#E2E8F0",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #1E2A3A",
        padding: "24px 32px",
        background: "linear-gradient(135deg, #0A0C10 0%, #0D1520 100%)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "4px" }}>
          <div style={{
            width: "10px", height: "10px", borderRadius: "50%",
            background: "#FF4444", boxShadow: "0 0 8px #FF4444",
            animation: "pulse 2s infinite"
          }} />
          <span style={{ color: "#64748B", fontSize: "11px", letterSpacing: "3px", textTransform: "uppercase" }}>
            SYSTEM ACTIVE
          </span>
        </div>
        <h1 style={{
          fontSize: "clamp(18px, 3vw, 28px)",
          fontWeight: "bold",
          color: "#00D4FF",
          letterSpacing: "1px",
          margin: "0 0 4px 0",
          textShadow: "0 0 20px rgba(0,212,255,0.3)"
        }}>
          Driver Drowsiness Detection — Project Roadmap
        </h1>
        <p style={{ color: "#475569", fontSize: "12px", margin: 0 }}>
          End-to-End MLOps Pipeline · {phases.length} Phases · {totalMax} Points Total
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "0", borderBottom: "1px solid #1E2A3A", padding: "0 32px" }}>
        {["roadmap", "scores", "stack"].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "14px 24px",
            background: "none",
            border: "none",
            borderBottom: tab === t ? "2px solid #00D4FF" : "2px solid transparent",
            color: tab === t ? "#00D4FF" : "#475569",
            cursor: "pointer",
            fontSize: "12px",
            letterSpacing: "2px",
            textTransform: "uppercase",
            fontFamily: "inherit",
            transition: "all 0.2s",
          }}>
            {t}
          </button>
        ))}
      </div>

      {/* Roadmap Tab */}
      {tab === "roadmap" && (
        <div style={{ padding: "32px" }}>
          {/* Timeline bar */}
          <div style={{
            display: "flex",
            gap: "4px",
            marginBottom: "32px",
            background: "#0D1520",
            padding: "16px",
            borderRadius: "8px",
            border: "1px solid #1E2A3A",
            flexWrap: "wrap",
          }}>
            <span style={{ color: "#475569", fontSize: "11px", width: "100%", marginBottom: "8px", letterSpacing: "2px" }}>
              TIMELINE — 15 DAYS
            </span>
            {phases.map(p => (
              <div key={p.id} onClick={() => setActivePhase(activePhase === p.id ? null : p.id)}
                style={{
                  flex: 1, minWidth: "60px",
                  height: "28px",
                  background: activePhase === p.id ? p.color : p.accent,
                  border: `1px solid ${p.color}`,
                  borderRadius: "4px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "10px",
                  color: activePhase === p.id ? "#000" : p.color,
                  fontWeight: "bold",
                  transition: "all 0.2s",
                }}>
                P{p.id}
              </div>
            ))}
          </div>

          {/* Phase Cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {phases.map(phase => (
              <div key={phase.id}
                onClick={() => setActivePhase(activePhase === phase.id ? null : phase.id)}
                style={{
                  background: activePhase === phase.id
                    ? `linear-gradient(135deg, ${phase.accent}88, #0D1520)`
                    : "#0D1520",
                  border: `1px solid ${activePhase === phase.id ? phase.color : "#1E2A3A"}`,
                  borderRadius: "8px",
                  padding: "20px 24px",
                  cursor: "pointer",
                  transition: "all 0.25s",
                  boxShadow: activePhase === phase.id ? `0 0 20px ${phase.color}22` : "none",
                }}>
                {/* Header row */}
                <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
                  <div style={{
                    width: "36px", height: "36px", borderRadius: "6px",
                    background: phase.accent,
                    border: `1px solid ${phase.color}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: phase.color, fontWeight: "bold", fontSize: "13px",
                    flexShrink: 0,
                  }}>
                    {phase.id}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: "10px", flexWrap: "wrap" }}>
                      <span style={{ color: phase.color, fontWeight: "bold", fontSize: "15px" }}>
                        {phase.title}
                      </span>
                      <span style={{ color: "#475569", fontSize: "12px" }}>
                        {phase.subtitle}
                      </span>
                    </div>
                    <div style={{ color: "#334155", fontSize: "11px", marginTop: "2px" }}>
                      {phase.score}
                    </div>
                  </div>
                  <div style={{
                    color: "#475569", fontSize: "11px",
                    background: "#111827", padding: "4px 10px", borderRadius: "4px",
                    border: "1px solid #1E2A3A", whiteSpace: "nowrap",
                  }}>
                    {phase.duration}
                  </div>
                  <div style={{
                    color: activePhase === phase.id ? phase.color : "#334155",
                    fontSize: "16px",
                    transition: "transform 0.2s",
                    transform: activePhase === phase.id ? "rotate(90deg)" : "none",
                  }}>›</div>
                </div>

                {/* Expanded content */}
                {activePhase === phase.id && (
                  <div style={{ marginTop: "20px", paddingTop: "20px", borderTop: `1px solid ${phase.color}33` }}>
                    <div style={{ marginBottom: "16px" }}>
                      <div style={{ color: "#475569", fontSize: "10px", letterSpacing: "2px", marginBottom: "10px" }}>
                        TASKS
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                        {phase.tasks.map((task, i) => (
                          <div key={i} style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
                            <div style={{
                              width: "20px", height: "20px", borderRadius: "4px",
                              border: `1px solid ${phase.color}66`,
                              background: phase.accent,
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: "9px", color: phase.color, flexShrink: 0, marginTop: "1px",
                            }}>
                              {String(i + 1).padStart(2, "0")}
                            </div>
                            <span style={{ color: "#94A3B8", fontSize: "13px", lineHeight: "1.5" }}>
                              {task}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div style={{
                      background: `${phase.accent}66`,
                      border: `1px solid ${phase.color}44`,
                      borderRadius: "6px",
                      padding: "12px 16px",
                    }}>
                      <span style={{ color: phase.color, fontSize: "10px", letterSpacing: "2px" }}>
                        ✓ DELIVERABLE:{" "}
                      </span>
                      <span style={{ color: "#94A3B8", fontSize: "12px" }}>
                        {phase.deliverable}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scores Tab */}
      {tab === "scores" && (
        <div style={{ padding: "32px" }}>
          <div style={{
            background: "#0D1520", border: "1px solid #1E2A3A",
            borderRadius: "8px", padding: "24px", marginBottom: "24px",
          }}>
            <div style={{ color: "#475569", fontSize: "11px", letterSpacing: "2px", marginBottom: "16px" }}>
              MARK ALLOCATION
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {scoreBreakdown.map((item, i) => (
                <div key={i}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <span style={{ color: "#94A3B8", fontSize: "13px" }}>{item.category}</span>
                    <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                      <span style={{ color: "#475569", fontSize: "11px" }}>{item.covered}</span>
                      <span style={{ color: "#00D4FF", fontSize: "13px", fontWeight: "bold", minWidth: "40px", textAlign: "right" }}>
                        {item.max} pts
                      </span>
                    </div>
                  </div>
                  <div style={{ height: "4px", background: "#1E2A3A", borderRadius: "2px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${(item.max / totalMax) * 100 * 3}%`,
                      maxWidth: "100%",
                      background: `hsl(${(i * 30) % 360}, 70%, 55%)`,
                      borderRadius: "2px",
                      transition: "width 0.5s",
                    }} />
                  </div>
                </div>
              ))}
            </div>
            <div style={{
              marginTop: "24px", paddingTop: "16px",
              borderTop: "1px solid #1E2A3A",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ color: "#64748B", fontSize: "13px" }}>TOTAL MARKS</span>
              <span style={{ color: "#00D4FF", fontSize: "24px", fontWeight: "bold" }}>
                {totalMax} pts
              </span>
            </div>
          </div>

          <div style={{
            background: "#0D1520", border: "1px solid #1E2A3A",
            borderRadius: "8px", padding: "24px",
          }}>
            <div style={{ color: "#475569", fontSize: "11px", letterSpacing: "2px", marginBottom: "16px" }}>
              REQUIRED DOCUMENTATION CHECKLIST
            </div>
            {[
              "Architecture diagram with block explanations",
              "High-level design (HLD) document with design rationale",
              "Low-level design (LLD) document with all API endpoint I/O specs",
              "Test plan & test cases document",
              "User manual for non-technical users (with screenshots)",
            ].map((doc, i) => (
              <div key={i} style={{
                display: "flex", gap: "10px", alignItems: "center",
                padding: "10px 0",
                borderBottom: i < 4 ? "1px solid #111827" : "none",
              }}>
                <div style={{
                  width: "18px", height: "18px", border: "1px solid #00D4FF",
                  borderRadius: "3px", flexShrink: 0,
                }} />
                <span style={{ color: "#94A3B8", fontSize: "13px" }}>{doc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stack Tab */}
      {tab === "stack" && (
        <div style={{ padding: "32px" }}>
          {[
            {
              label: "DATA ENGINEERING", color: "#FF6B35",
              items: ["Apache Airflow (DAG orchestration)", "PySpark (feature engineering)", "MediaPipe (landmark extraction)", "OpenCV (frame extraction)"]
            },
            {
              label: "VERSION CONTROL", color: "#A855F7",
              items: ["Git + Git LFS (code & large files)", "DVC (data, features, model versioning)", "DVC pipeline (dvc.yaml DAG)"]
            },
            {
              label: "ML & EXPERIMENT TRACKING", color: "#10B981",
              items: ["XGBoost (primary classifier)", "SVM (comparison model)", "MLflow (experiment tracking + registry)", "scikit-learn (metrics, preprocessing)"]
            },
            {
              label: "BACKEND & SERVING", color: "#F43F5E",
              items: ["FastAPI (main API + WebSocket)", "Model Server (MLflow-loaded model)", "Prometheus client (instrumentation)", "Python logging (structured logs)"]
            },
            {
              label: "MONITORING", color: "#F97316",
              items: ["Prometheus (metrics collection)", "Grafana (real-time dashboards)", "Custom drift detector (vs baseline.json)", "Alert rules (error rate > 5%, drift threshold)"]
            },
            {
              label: "FRONTEND", color: "#06B6D4",
              items: ["React + Vite (web application)", "WebSocket client (live predictions)", "recharts (metric visualizations)", "Responsive dark automotive UI"]
            },
            {
              label: "PACKAGING & CI", color: "#8B5CF6",
              items: ["Docker (per-service Dockerfiles)", "Docker Compose (5-service stack)", "GitHub Actions / DVC CI pipeline", "MLflow MLprojects (env parity)"]
            },
          ].map((group, i) => (
            <div key={i} style={{
              background: "#0D1520", border: "1px solid #1E2A3A",
              borderRadius: "8px", padding: "20px", marginBottom: "12px",
            }}>
              <div style={{ color: group.color, fontSize: "11px", letterSpacing: "2px", marginBottom: "12px" }}>
                {group.label}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                {group.items.map((item, j) => (
                  <span key={j} style={{
                    background: "#111827", border: `1px solid ${group.color}44`,
                    borderRadius: "4px", padding: "6px 12px",
                    color: "#94A3B8", fontSize: "12px",
                  }}>
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0A0C10; }
        ::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 3px; }
      `}</style>
    </div>
  );
}
