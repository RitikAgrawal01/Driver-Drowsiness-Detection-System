"""
Plain-Python helpers for the DAGs (no Airflow import, so they are unit-tested in CI).

The DAGs do not re-implement the pipeline: every data/model step is `dvc repro`, so DVC's
dvc.yaml + params.yaml stay the single definition of what runs and with which settings.
"""

import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(os.getenv("DDD_REPO", "/opt/project"))
VIDEO_RE = re.compile(r"(alert|drowsy)(\d+)\.(mp4|mov)$", re.IGNORECASE)


def sh(cmd: str, cwd: Path = REPO) -> str:
    """Run a shell command; raise (fail the task) on a non-zero exit. No `|| true` anywhere."""
    r = subprocess.run(["bash", "-c", f"set -euo pipefail; {cmd}"], cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-20000:])
    if r.returncode != 0:
        print(r.stderr[-20000:])
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.stdout


def fingerprint(path: Path, chunk: int = 8 << 20) -> str:
    """Cheap content fingerprint: size + md5 of the first and last 8 MB (catches copied files)."""
    size = path.stat().st_size
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def validate_raw_videos(raw_dir: Path) -> dict:
    """Names, pairing (alertN + drowsyN per participant) and duplicate content (v1 had drowsy8 == alert8)."""
    files = sorted(p for p in raw_dir.glob("*/*") if p.is_file() and not p.name.startswith("."))
    problems, pairs, prints = [], defaultdict(set), defaultdict(list)
    for p in files:
        m = VIDEO_RE.fullmatch(p.name)
        if not m or p.parent.name != m.group(1).lower():
            problems.append(f"unexpected file {p.relative_to(raw_dir)}")
            continue
        pairs[int(m.group(2))].add(m.group(1).lower())
        prints[fingerprint(p)].append(p.name)
    for part, kinds in sorted(pairs.items()):
        if kinds != {"alert", "drowsy"}:
            problems.append(f"participant {part} has only {sorted(kinds)}")
    for names in prints.values():
        if len(names) > 1:
            problems.append(f"identical content: {names}")
    summary = {"videos": len(files), "participants": len(pairs), "problems": problems}
    if problems:
        raise ValueError(json.dumps(summary, indent=2))
    return summary


def stale_stages(repo: Path = REPO) -> list[str]:
    """Stages `dvc status` reports as changed (empty list = everything up to date)."""
    out = sh("dvc status --json", cwd=repo)
    return sorted(json.loads(out or "{}").keys())


def should_retrain(conf: dict | None, run_type: str, stale: list[str]) -> tuple[bool, str]:
    """Manual / alert-triggered runs always proceed; scheduled runs only if something changed."""
    conf = conf or {}
    if conf.get("retrain_reason"):
        return True, f"triggered: {conf['retrain_reason']}"
    if run_type == "manual":
        return True, "manual run"
    if stale:
        return True, f"stale stages: {stale}"
    return False, "scheduled run and nothing changed (dvc status clean)"
