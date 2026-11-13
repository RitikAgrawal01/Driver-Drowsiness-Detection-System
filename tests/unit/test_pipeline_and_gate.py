"""Model selection (grouped CV), the pyfunc wrapper and the gate's decisions."""

import json

import numpy as np
import pandas as pd
import pytest

from ddd_core.windows import FEATURE_SETS
from pipeline import gate as gate_mod
from pipeline.evaluation import summarize, window_metrics
from pipeline.models import best_f1_threshold, fit_final, score, select
from pipeline.pyfunc_model import DrowsinessModel

FEATS = FEATURE_SETS["v2_12"]


def grouped_data(n_participants=8, per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_participants + 1):
        for label in (0, 1):
            for k in range(per_class):
                r = {f: rng.normal(0, 1) for f in FEATS}
                r["perclos"] = float(np.clip(0.05 + 0.25 * label + rng.normal(0, 0.05), 0, 1))
                r.update(video=f"{'drowsy' if label else 'alert'}{p}", participant=p, label=label, start_s=2 * k)
                rows.append(r)
    return pd.DataFrame(rows)


def test_best_f1_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 1, 1])
    assert 0.2 < best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9])) <= 0.8


@pytest.mark.parametrize("model", ["rule", "logreg", "xgboost"])
def test_select_and_fit_on_grouped_data(model, params):
    df = grouped_data()
    feats = ["perclos"] if model == "rule" else FEATS
    choice = select(model, {15: df, 30: df}, feats, params["models"], n_splits=4)
    assert choice["window_s"] in (15, 30) and 0.5 < choice["inner_auc"] <= 1.0
    est = fit_final(model, choice, df, feats, params["models"])
    s = score(model, est, df[feats])
    assert len(s) == len(df)
    if model == "xgboost":
        assert choice["n_estimators"] >= 1


def test_pyfunc_wrapper_rule_and_threshold():
    m = DrowsinessModel(None, "rule", ["perclos"], threshold=0.1, window_s=60)
    out = m.predict(None, pd.DataFrame({"perclos": [0.05, 0.2]}))
    assert out["drowsy"].tolist() == [0, 1] and out["score"].tolist() == [0.05, 0.2]


def test_summarize_per_participant_and_video_vote():
    pred = pd.DataFrame(
        {
            "video": ["alert1", "alert1", "drowsy1", "drowsy1"],
            "participant": 1,
            "label": [0, 0, 1, 1],
            "window_s": 15,
            "score": [0.1, 0.3, 0.4, 0.9],
            "pred": [0, 0, 0, 1],
        }
    )
    s, per = summarize(pred)
    assert s["pooled"]["recall"] == 0.5
    assert s["video_level_correct"] == "2/2"  # drowsy1: 1 of 2 windows drowsy -> a tie counts as drowsy
    assert per.iloc[0]["roc_auc"] == 1.0
    assert window_metrics(np.array([0, 1]), np.array([0.2, 0.8]), np.array([0, 1]))["f1"] == 1.0


# ── gate ─────────────────────────────────────────────────────────────────
def run_metrics(pp_auc, pooled_auc):
    return {"pooled": {"roc_auc": pooled_auc, "f1": 0.7}, "per_participant_mean": {"roc_auc": pp_auc}}


def run_gate(tmp_path, monkeypatch, ml_auc, rule_auc, champ):
    metrics = {
        "rule__perclos_only": run_metrics(rule_auc, rule_auc),
        "svm__v2_12": run_metrics(ml_auc, ml_auc),
        "xgboost__v2_12": run_metrics(ml_auc - 0.05, ml_auc),
        "svm__v2_12_no_pose": run_metrics(0.99, 0.99),
        "_meta": {},
    }
    (tmp_path / "loso_metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(gate_mod, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(gate_mod, "current_champion", champ)
    try:
        gate_mod.main()
        code = 0
    except SystemExit as e:
        code = e.code
    return code, json.loads((tmp_path / "gate.json").read_text())


def no_champion(name, alias):
    return {"status": "none"}


def test_gate_deploys_ml_model_when_it_beats_the_rule(tmp_path, monkeypatch):
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.85, rule_auc=0.8, champ=no_champion)
    assert code == 0 and g["ml_gate_passed"] and g["deploy"]["model"] == "svm"
    assert g["ml_candidate"] == "svm__v2_12"  # main feature set only, never the ablation


def test_gate_falls_back_to_rule_when_ml_loses(tmp_path, monkeypatch):
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.75, rule_auc=0.8, champ=no_champion)
    assert code == 0 and not g["ml_gate_passed"] and g["deploy"]["model"] == "rule"


def test_gate_fails_when_nothing_is_good_enough(tmp_path, monkeypatch):
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.6, rule_auc=0.65, champ=no_champion)
    assert code == 1 and g["deploy"] is None


def test_gate_fails_closed_when_mlflow_is_unreachable(tmp_path, monkeypatch):
    def unreachable(name, alias):
        raise ConnectionError("mlflow down")

    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.9, rule_auc=0.8, champ=unreachable)
    assert code == 1 and not g["passed"]  # v1 treated "MLflow down" as prod F1 = 0 and promoted


def champion(run_key, stored_auc=0.9):
    return lambda name, alias: {"status": "ok", "version": 5, "pp_auc": stored_auc, "run_key": run_key}


def test_gate_compares_with_the_champion_rescored_on_current_data(tmp_path, monkeypatch):
    """Champion family = svm no-pose; in THIS run it scores 0.99, so the candidate (0.85) loses."""
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.85, rule_auc=0.8, champ=champion("svm__v2_12_no_pose", 0.5))
    check = g["ml_checks"]["beats_champion"]
    assert not check["passed"] and check["champion_pp_auc_on_current_data"] == 0.99  # not the stored 0.5


def test_gate_ignores_a_stale_stored_score_from_smaller_data(tmp_path, monkeypatch):
    """The rule was champion with 0.81 on 12 people; on more people the rule scores 0.75. Same family -> ok."""
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.7, rule_auc=0.75, champ=champion("rule__perclos_only", 0.81))
    assert code == 0 and g["deploy"]["model"] == "rule"
    assert "re-trained" in g["baseline_checks"]["beats_champion"]["note"]


def test_gate_falls_back_to_stored_score_when_champion_family_missing(tmp_path, monkeypatch):
    code, g = run_gate(tmp_path, monkeypatch, ml_auc=0.85, rule_auc=0.8, champ=champion("knn__v9", 0.9))
    assert code == 1 and "stored" in g["ml_checks"]["beats_champion"]["note"]
