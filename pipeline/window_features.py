"""
Stage 3: time-based sliding windows (15 / 30 / 60 s, 2 s stride) -> 12 features per window,
via ddd_core.windows.window_features — the same function the live backend calls.

Outputs: data/windows.parquet, reports/window_summary.json
"""

import json

import numpy as np
import pandas as pd

from ddd_core.config import load_params
from ddd_core.windows import ALL_FEATURES, window_features
from pipeline.dataset import FRAME_FEATURES, REPORTS_DIR, WINDOWS


def main():
    params = load_params()
    cfg, fps = params["features"], params["extract"]["target_fps"]
    frames = pd.read_parquet(FRAME_FEATURES)

    out, summary = [], {"per_window_length": {}}
    for length in cfg["window_lengths_s"]:
        kept = dropped = 0
        for video, v in frames.groupby("video", sort=True):
            v = v.sort_values("t")
            cols = {c: v[c].to_numpy() for c in ["t", "face", "ear", "mar", "pitch", "yaw", "roll"]}
            t = cols["t"]
            s = t[0]
            while s + length <= t[-1] + 0.5 / fps:
                lo, hi = np.searchsorted(t, [s, s + length])
                w = window_features(
                    *(cols[c][lo:hi] for c in ["t", "face", "ear", "mar", "pitch", "yaw", "roll"]),
                    length_s=float(length),
                    cfg=cfg,
                    fps=fps,
                )
                if w is None:
                    dropped += 1
                else:
                    w.update(
                        video=video,
                        participant=int(v["participant"].iloc[0]),
                        label=int(v["label"].iloc[0]),
                        window_s=int(length),
                        start_s=round(float(s), 2),
                    )
                    out.append(w)
                    kept += 1
                s += cfg["stride_s"]
        summary["per_window_length"][str(length)] = {"windows": kept, "dropped_low_face": dropped}

    df = pd.DataFrame(out)[["video", "participant", "label", "window_s", "start_s", "face_ratio"] + ALL_FEATURES]
    df.to_parquet(WINDOWS, index=False)

    counts = df.groupby(["window_s", "participant", "label"]).size().unstack("label", fill_value=0)
    summary["per_participant"] = {
        f"L{L}_p{p}": {"alert": int(r[0]), "drowsy": int(r[1])} for (L, p), r in counts.iterrows()
    }
    with open(REPORTS_DIR / "window_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["per_window_length"], indent=2))


if __name__ == "__main__":
    main()
