"""
Stage 5: evaluation gate. Fails the pipeline (exit 1) when nothing may be deployed, and fails
closed: if the current champion cannot be checked, nothing is promoted.

ML candidate = the ML model on the main feature set with the best mean per-participant ROC-AUC
under nested LOSO. Pre-registered rules (params.yaml -> gate):
  1. candidate mean per-participant ROC-AUC > the PERCLOS rule's
  2. candidate pooled LOSO ROC-AUC >= min_pooled_roc_auc
  3. candidate beats the registered champion by > min_gain, with the champion's model family
     RE-SCORED on the current data in this run (a stored score from a smaller dataset is not
     comparable); same family as the champion = re-trained champion, allowed.
     (no champion yet -> first model, explicitly allowed; MLflow unreachable -> FAIL)
Deployment policy: if the ML candidate fails and deploy_baseline_if_ml_fails is true, the
PERCLOS rule is deployed instead, provided it passes rules 2 and 3 itself.
Output: reports/gate.json (`deploy` = what train_final registers).
"""

import json
import sys

from ddd_core.config import load_params
from pipeline.dataset import REPORTS_DIR
from pipeline.evaluation import save_json


def current_champion(name: str, alias: str) -> dict:
    """{'status': 'none'} | {'status': 'ok', ...}; raises if MLflow is unreachable."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e.error_code) or "not found" in str(e).lower():
            return {"status": "none"}
        raise
    run = client.get_run(mv.run_id)
    return {
        "status": "ok",
        "version": int(mv.version),
        "pp_auc": run.data.metrics.get("loso_pp_mean_roc_auc"),
        "loso_md5": run.data.tags.get("loso_metrics_md5"),
        "run_key": run.data.tags.get("loso_run_key"),
    }


def champion_check(run_key: str, pp_auc: float, champ: dict | None, metrics: dict, min_gain: float) -> dict:
    """Compare the candidate with the current champion ON THE SAME DATA AND PROTOCOL.

    The champion's stored LOSO score may come from a different (smaller) dataset, so it is not
    comparable. train_evaluate re-evaluates every model family on the current data, so the
    champion's family is re-scored here; the stored score is only a fallback.
    """
    if champ is None:
        return {"passed": False, "note": "cannot read champion from MLflow (fail closed)"}
    if champ["status"] == "none":
        return {"passed": True, "note": "no registered champion yet (first model)"}
    champ_key = champ.get("run_key")
    if champ_key == run_key:
        return {
            "passed": True,
            "note": f"same model family as champion v{champ['version']}, re-trained on current data",
        }
    if champ_key in metrics:
        rescored = metrics[champ_key]["per_participant_mean"]["roc_auc"]
        return {
            "champion_version": champ["version"],
            "champion_family": champ_key,
            "champion_pp_auc_on_current_data": rescored,
            "required_gain": min_gain,
            "passed": pp_auc > rescored + min_gain,
        }
    return {
        "champion_version": champ["version"],
        "champion_pp_auc_stored": champ["pp_auc"],
        "note": "champion family not evaluated in this run; compared with its stored score (different data)",
        "required_gain": min_gain,
        "passed": champ["pp_auc"] is not None and pp_auc > champ["pp_auc"] + min_gain,
    }


def main():
    params = load_params()
    g, ml, main_set = params["gate"], params["mlflow"], params["features"]["main_set"]
    metrics_path = REPORTS_DIR / "loso_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.pop("_meta", None)

    candidates = {k: v for k, v in metrics.items() if k.endswith(f"__{main_set}")}
    cand_key = max(candidates, key=lambda k: candidates[k]["per_participant_mean"]["roc_auc"])
    cand = candidates[cand_key]
    base_key = next(k for k in metrics if k.startswith(f"{g['must_beat_baseline']}__"))
    base = metrics[base_key]
    cand_auc, base_auc = cand["per_participant_mean"]["roc_auc"], base["per_participant_mean"]["roc_auc"]

    try:
        champ = current_champion(ml["model_name"], ml["alias"])
    except Exception as e:
        print(f"cannot read champion: {e}", file=sys.stderr)
        champ = None

    ml_checks = {
        "beats_baseline": {"candidate_pp_auc": cand_auc, "baseline_pp_auc": base_auc, "passed": cand_auc > base_auc},
        "min_pooled_auc": {
            "candidate_pooled_auc": cand["pooled"]["roc_auc"],
            "required": g["min_pooled_roc_auc"],
            "passed": cand["pooled"]["roc_auc"] >= g["min_pooled_roc_auc"],
        },
        "beats_champion": champion_check(cand_key, cand_auc, champ, metrics, g["min_gain_over_champion"]),
    }
    ml_passed = all(c["passed"] for c in ml_checks.values())
    result = {"ml_candidate": cand_key, "baseline_run": base_key, "ml_checks": ml_checks, "ml_gate_passed": ml_passed}

    deploy = None
    if ml_passed:
        deploy = {
            "model": cand_key.split("__")[0],
            "feature_set": main_set,
            "run_key": cand_key,
            "reason": "ML candidate passed the gate",
        }
    elif g.get("deploy_baseline_if_ml_fails"):
        base_checks = {
            "min_pooled_auc": {
                "baseline_pooled_auc": base["pooled"]["roc_auc"],
                "required": g["min_pooled_roc_auc"],
                "passed": base["pooled"]["roc_auc"] >= g["min_pooled_roc_auc"],
            },
            "beats_champion": champion_check(base_key, base_auc, champ, metrics, g["min_gain_over_champion"]),
        }
        result["baseline_checks"] = base_checks
        if all(c["passed"] for c in base_checks.values()):
            deploy = {
                "model": "rule",
                "feature_set": "perclos_only",
                "run_key": base_key,
                "reason": "ML candidate failed; deploying the PERCLOS rule baseline (policy)",
            }

    result["deploy"] = deploy
    result["passed"] = deploy is not None
    save_json(result, REPORTS_DIR / "gate.json")
    print(json.dumps(result, indent=2))
    if deploy is None:
        print("GATE FAILED: nothing will be promoted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
