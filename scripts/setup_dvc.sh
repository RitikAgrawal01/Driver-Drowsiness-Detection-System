#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup_dvc.sh
# One-time DVC + Git LFS setup script for WSL2.
# Run this ONCE after cloning the repo on a new machine.
#
# Usage:
#   chmod +x scripts/setup_dvc.sh
#   ./scripts/setup_dvc.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "══════════════════════════════════════════════════════"
echo " DDD Project — DVC + Git LFS Setup"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Step 1: Git LFS ───────────────────────────────────────────────────────────
echo "→ Installing Git LFS hooks..."
git lfs install
echo "  ✓ Git LFS installed"

# ── Step 2: DVC init (safe to run if already initialised) ─────────────────────
echo "→ Checking DVC initialisation..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "  ✓ DVC initialised"
else
    echo "  ✓ DVC already initialised (.dvc/ exists)"
fi

# ── Step 3: DVC remote ────────────────────────────────────────────────────────
echo "→ Setting up DVC local remote..."

DVC_REMOTE_PATH="$HOME/dvc_remote_ddd"
mkdir -p "$DVC_REMOTE_PATH"

# Add remote (ignore error if already exists)
dvc remote add -d localremote "$DVC_REMOTE_PATH" 2>/dev/null || \
    dvc remote modify localremote url "$DVC_REMOTE_PATH"

echo "  ✓ DVC remote: $DVC_REMOTE_PATH"

# ── Step 4: DVC autostage ─────────────────────────────────────────────────────
echo "→ Enabling DVC autostage..."
dvc config core.autostage true
echo "  ✓ DVC autostage enabled (dvc add automatically runs git add)"

# ── Step 5: Create required directories ───────────────────────────────────────
echo "→ Creating data directories..."
mkdir -p data/raw/drowsy data/raw/alert
mkdir -p data/frames/drowsy data/frames/alert
mkdir -p data/landmarks data/features data/processed
mkdir -p models reports logs/backend logs/model_server
echo "  ✓ Directory structure ready"

# ── Step 6: .env setup ────────────────────────────────────────────────────────
echo "→ Checking .env file..."
if [ ! -f ".env" ]; then
    cp .env.template .env
    # Generate Fernet key for Airflow
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "")
    if [ -n "$FERNET_KEY" ]; then
        sed -i "s/AIRFLOW__CORE__FERNET_KEY=.*/AIRFLOW__CORE__FERNET_KEY=$FERNET_KEY/" .env
        echo "  ✓ .env created with auto-generated Fernet key"
    else
        echo "  ✓ .env created from template (set AIRFLOW__CORE__FERNET_KEY manually)"
    fi
else
    echo "  ✓ .env already exists"
fi

# ── Step 7: Commit DVC config if changed ──────────────────────────────────────
echo "→ Committing DVC config..."
git add .dvc/config .dvc/.gitignore 2>/dev/null || true
git diff --staged --quiet || git commit -m "chore: configure DVC remote (localremote)" 2>/dev/null || true
echo "  ✓ DVC config committed"

# ── Step 8: Verify ────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo " Verification"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Git LFS version : $(git lfs version 2>/dev/null || echo 'not found')"
echo "DVC version     : $(dvc version 2>/dev/null | head -1 || echo 'not found')"
echo "DVC remote      : $(dvc remote list 2>/dev/null || echo 'none')"
echo ""
echo "Pipeline stages in dvc.yaml:"
python3 -c "
import yaml
with open('dvc.yaml') as f:
    p = yaml.safe_load(f)
for s in p.get('stages', {}):
    print(f'  ✓ {s}')
" 2>/dev/null || echo "  (dvc.yaml not readable)"
echo ""
echo "══════════════════════════════════════════════════════"
echo " Setup complete! Next steps:"
echo ""
echo "  1. Download dataset into data/raw/drowsy/ and data/raw/alert/"
echo "  2. Run pipeline: dvc repro"
echo "  3. Check DAG:    dvc dag"
echo "  4. Push data:    dvc push"
echo "══════════════════════════════════════════════════════"
echo ""
