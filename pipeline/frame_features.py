"""
Stage 2: per-frame EAR, MAR and head pose, using ddd_core.geometry.frame_features —
the exact function the live backend calls.

Reads the folds listed in params.yaml -> data.folds (never the sealed test fold).
Output: data/frame_features.parquet (no-face frames kept with face=0 and NaN features)
"""

import pandas as pd

from ddd_core.geometry import USED_LANDMARKS, frame_features
from pipeline.dataset import FRAME_FEATURES, LANDMARKS_DIR, active_folds


def landmark_files() -> list:
    files = []
    for fold in active_folds():
        found = sorted((LANDMARKS_DIR / f"fold{fold}").glob("*.csv"))
        if not found:
            raise RuntimeError(f"no landmarks for fold {fold}; run `dvc repro extract_landmarks@{fold}`")
        files += [(fold, f) for f in found]
    return files


def main():
    parts = []
    for fold, f in landmark_files():
        df = pd.read_csv(f)
        feats = []
        for r in df.itertuples(index=False):
            r = r._asdict()
            pts = {i: (r[f"x{i}"], r[f"y{i}"]) for i in USED_LANDMARKS} if r["face"] == 1 else None
            feats.append(frame_features(pts, int(r["width"]), int(r["height"])))
        out = pd.concat(
            [df[["video", "participant", "label", "t"]].reset_index(drop=True), pd.DataFrame(feats)], axis=1
        )
        out.insert(0, "fold", fold)
        parts.append(out)
        print(
            f"{f.stem:>9}: {len(out)} frames, median EAR {out['ear'].median():.3f}, "
            f"median pitch/yaw/roll {out[['pitch', 'yaw', 'roll']].median().round(1).tolist()}",
            flush=True,
        )

    frames = pd.concat(parts, ignore_index=True).sort_values(["video", "t"])
    frames.to_parquet(FRAME_FEATURES, index=False)
    print(f"saved {len(frames)} frames from {len(parts)} videos")


if __name__ == "__main__":
    main()
