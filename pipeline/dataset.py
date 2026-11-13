"""Paths, folds and video naming for the pipeline.

Raw videos live per RLDD fold in data/raw/fold<N>/{alert,drowsy}/ (usually a symlink to an
external drive). Which folds are used is set ONLY by params.yaml -> data.folds; the sealed
test fold (data.sealed_test_fold) is refused everywhere until the final test.
"""

import re
from pathlib import Path

from ddd_core.config import REPO_ROOT, load_params

RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
LANDMARKS_DIR = DATA_DIR / "landmarks_10fps"
FRAME_FEATURES = DATA_DIR / "frame_features.parquet"
WINDOWS = DATA_DIR / "windows.parquet"


class SealedFoldError(RuntimeError):
    pass


def active_folds(params: dict | None = None) -> list[int]:
    """Folds used for training/evaluation; raises if the sealed test fold is among them."""
    data = (params or load_params())["data"]
    folds = [int(f) for f in data["folds"]]
    if int(data["sealed_test_fold"]) in folds:
        raise SealedFoldError(f"fold {data['sealed_test_fold']} is the sealed test fold; remove it from data.folds")
    return folds


def check_fold_allowed(fold: int, params: dict | None = None) -> None:
    sealed = int((params or load_params())["data"]["sealed_test_fold"])
    if int(fold) == sealed:
        raise SealedFoldError(f"fold {fold} is sealed (final test only)")


def parse_video_name(path: Path) -> tuple[str, int, int]:
    """alert13.mp4 -> ('alert13', participant 13, label 0); drowsy13.mp4 -> label 1.

    alertN and drowsyN are the same RLDD participant N (KSS 0 and KSS 10 videos).
    """
    m = re.fullmatch(r"(alert|drowsy)(\d+)", path.stem)
    if not m:
        raise ValueError(f"Unexpected video name: {path.name}")
    return path.stem, int(m.group(2)), 1 if m.group(1) == "drowsy" else 0


def list_videos(fold: int) -> list[Path]:
    """Videos of one fold; every participant must have exactly one alert and one drowsy video."""
    check_fold_allowed(fold)
    root = RAW_DIR / f"fold{fold}"
    vids = sorted(p for p in root.glob("*/*") if p.suffix.lower() in {".mp4", ".mov"})
    if not vids:
        raise RuntimeError(f"No videos in {root} (is the external drive connected / the symlink in place?)")
    pairs: dict[int, list[int]] = {}
    for v in vids:
        _, participant, label = parse_video_name(v)
        pairs.setdefault(participant, []).append(label)
    bad = {p: labels for p, labels in pairs.items() if sorted(labels) != [0, 1]}
    if bad:
        raise RuntimeError(f"fold {fold}: participants without exactly one alert + one drowsy video: {bad}")
    return vids
