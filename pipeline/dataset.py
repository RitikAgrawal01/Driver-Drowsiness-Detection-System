"""Paths and video naming for the pipeline."""

import re
from pathlib import Path

from ddd_core.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
LANDMARKS_DIR = DATA_DIR / "landmarks_10fps"
FRAME_FEATURES = DATA_DIR / "frame_features.parquet"
WINDOWS = DATA_DIR / "windows.parquet"


def parse_video_name(path: Path) -> tuple[str, int, int]:
    """alert3.MOV -> ('alert3', participant 3, label 0); drowsy3.mov -> label 1.

    alertN and drowsyN are the same RLDD Fold-1 participant N (KSS 0 and KSS 10 videos).
    """
    m = re.fullmatch(r"(alert|drowsy)(\d+)", path.stem)
    if not m:
        raise ValueError(f"Unexpected video name: {path.name}")
    return path.stem, int(m.group(2)), 1 if m.group(1) == "drowsy" else 0


def list_videos() -> list[Path]:
    vids = sorted(p for p in RAW_DIR.glob("*/*") if p.suffix.lower() in {".mp4", ".mov"})
    if len(vids) != 24:
        raise RuntimeError(f"Expected 24 videos in {RAW_DIR}, found {len(vids)}")
    return vids
