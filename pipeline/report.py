"""Stage 7: write reports/RESULTS.md and reports/per_participant_auc.png from the stage outputs."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ddd_core.config import load_params  # noqa: E402
from pipeline.dataset import REPORTS_DIR as R  # noqa: E402
from pipeline.dataset import active_folds  # noqa: E402


def load(name):
    return json.loads((R / name).read_text())


def main():
    params = load_params()
    main_set = params["features"]["main_set"]
    folds = active_folds(params)
    ext = [load(f"extraction/fold{f}.json") for f in folds]
    frames, faces = sum(e["frames"] for e in ext), sum(e["faces"] for e in ext)
    win, loso = load("window_summary.json"), load("loso_metrics.json")
    leaky, gate, final = load("leaky_vs_loso.json"), load("gate.json"), load("final_model.json")
    loso.pop("_meta", None)
    per_part = pd.read_csv(R / "loso_per_participant.csv")
    n_people = per_part["participant"].nunique()

    L = ["# Results: subject-wise evaluation (v2)", ""]
    L += [
        "## Data",
        "",
        f"- UTA-RLDD folds {folds}: {n_people} participants, {2 * n_people} videos "
        "(alert = KSS 0, drowsy = KSS 10). "
        f"Fold {params['data']['sealed_test_fold']} is sealed for the final test and was not used.",
        f"- {params['extract']['target_fps']} fps by timestamp, rotation metadata applied: {frames:,} frames, "
        f"face found in {faces:,} ({100 * faces / max(frames, 1):.2f} %).",
        "- Windows (2 s stride, face in >= 80 % of expected frames): "
        + ", ".join(
            f"{k} s: {v['windows']:,} (dropped {v['dropped_low_face']})" for k, v in win["per_window_length"].items()
        ),
        "",
    ]

    L += [
        "## Nested leave-one-participant-out",
        "",
        "Drowsy = positive class. **Pooled** = all 12 held-out participants' windows together. "
        "**Per participant** = metric on each held-out person, then mean ± std. "
        "**Videos** = majority vote of window predictions per video (24 videos).",
        "",
        "| Model | Features | Precision | Recall | F1 | ROC-AUC | PR-AUC | ROC-AUC / participant | "
        "F1 / participant | Worst recall | Videos | Window chosen (s: folds) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in loso.items():
        model, fs = key.split("__")
        p, mu, sd = m["pooled"], m["per_participant_mean"], m["per_participant_std"]
        chosen = ", ".join(f"{k}: {v}" for k, v in sorted(m["chosen_window_s"].items(), key=lambda x: int(x[0])))
        L.append(
            f"| {model} | {fs} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['roc_auc']:.3f} | "
            f"{p['pr_auc']:.3f} | {mu['roc_auc']:.3f} ± {sd['roc_auc']:.3f} | {mu['f1']:.3f} ± {sd['f1']:.3f} | "
            f"{m['worst_participant_recall']:.3f} | {m['video_level_correct']} | {chosen} |"
        )
    L += [""]

    L += [
        "## How much did the v1 random split inflate the score?",
        "",
        f"Same v2 features, {leaky['window_s']} s windows, v1 hyperparameters, threshold 0.5. Only the split differs.",
        "",
        "| Model | Random split F1 | Random split ROC-AUC | LOSO F1 | LOSO ROC-AUC | F1 inflation |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["xgboost", "svm"]:
        r, lo = leaky[name]["random_split"], leaky[name]["loso"]
        L.append(
            f"| {name} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {lo['f1']:.3f} | {lo['roc_auc']:.3f} | "
            f"{leaky[name]['f1_inflation']:+.3f} |"
        )
    L += [
        "",
        "v1 reported XGBoost F1 0.934 / ROC-AUC 0.994 (random split, 2 fps, sideways frames, duplicate video).",
        "",
    ]

    L += [
        "## Gate (pre-registered rules)",
        "",
        f"- ML candidate: **{gate['ml_candidate']}** (best per-participant ROC-AUC on {main_set}); "
        f"baseline: {gate['baseline_run']}.",
    ]
    for name, c in gate["ml_checks"].items():
        detail = ", ".join(f"{k}={v}" for k, v in c.items() if k != "passed")
        L.append(f"  - {name}: {'pass' if c['passed'] else 'FAIL'} ({detail})")
    L.append(f"- ML gate: **{'PASSED' if gate['ml_gate_passed'] else 'FAILED'}**")
    if "baseline_checks" in gate:
        L.append("- Deployment policy `deploy_baseline_if_ml_fails` (added after the first run; gate rules unchanged):")
        for name, c in gate["baseline_checks"].items():
            detail = ", ".join(f"{k}={v}" for k, v in c.items() if k != "passed")
            L.append(f"  - baseline {name}: {'pass' if c['passed'] else 'FAIL'} ({detail})")
    dep = gate.get("deploy")
    L += [f"- Deployed: **{dep['model'] if dep else 'nothing'}**" + (f" — {dep['reason']}" if dep else ""), ""]

    L += [
        f"## Final model (fit on all {n_people} participants)",
        "",
        f"- {final['model']} on {final['feature_set']}, {final['window_s']} s windows, config {final['config']}, "
        f"trees {final['n_estimators']}, threshold {final['threshold']:.4f} (from grouped out-of-fold scores).",
        f"- MLflow: `{final['registered_model']}` version {final['version']}, alias `@{final['alias']}`, "
        f"run `{final['mlflow_run_id']}`.",
        "- Expected performance on a NEW person = the nested LOSO numbers above, not a training score.",
        "",
    ]

    shown = dep["run_key"] if dep else gate["ml_candidate"]
    champ = per_part[per_part["run"] == shown].sort_values("participant")
    L += [
        f"## Per participant ({shown})",
        "",
        "| Participant | Window (s) | Windows | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in champ.iterrows():
        L.append(
            f"| {int(r['participant'])} | {int(r['window_s'])} | {int(r['n_windows'])} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} |"
        )
    L += [
        "",
        "## Caveats",
        "",
        "- Labels are per video (self-reported KSS), not per moment: a 'drowsy' video contains alert moments.",
        f"- {n_people} participants is still small; the spread across people matters as much as the mean.",
        "- The deployed model family was picked by comparing outer LOSO results (mild optimism). "
        "Everything inside a fold (window, hyperparameters, trees, threshold) was chosen without the test person.",
        "- RLDD was recorded at a desk, not while driving.",
        "",
    ]
    (R / "RESULTS.md").write_text("\n".join(L))

    runs = [k for k in loso if k.endswith(f"__{main_set}") or k.startswith("rule__")]
    pivot = per_part[per_part["run"].isin(runs)].pivot(index="participant", columns="run", values="roc_auc")
    ax = pivot.plot(kind="bar", figsize=(11, 4.5), width=0.8)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("ROC-AUC on held-out participant")
    ax.set_xlabel("Participant")
    ax.set_title("Nested LOSO: ROC-AUC per held-out participant")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(R / "per_participant_auc.png", dpi=130)
    print((R / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
