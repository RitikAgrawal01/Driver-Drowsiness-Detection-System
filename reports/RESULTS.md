# Results: subject-wise evaluation (v2)

## Data

- 12 participants (UTA-RLDD Fold 1, participants 01-12), 24 videos: alert = KSS 0, drowsy = KSS 10. The duplicated drowsy8 file was replaced with the original.
- 10 fps by timestamp, rotation metadata applied: 143,735 frames, face found in 143,680 (99.96 %).
- Windows (2 s stride, face in >= 80 % of expected frames): 15 s: 7,019 (dropped 0), 30 s: 6,841 (dropped 0), 60 s: 6,481 (dropped 0)

## Nested leave-one-participant-out

Drowsy = positive class. **Pooled** = all 12 held-out participants' windows together. **Per participant** = metric on each held-out person, then mean ± std. **Videos** = majority vote of window predictions per video (24 videos).

| Model | Features | Precision | Recall | F1 | ROC-AUC | PR-AUC | ROC-AUC / participant | F1 / participant | Worst recall | Videos | Window chosen (s: folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | perclos_only | 0.603 | 0.908 | 0.725 | 0.759 | 0.784 | 0.811 ± 0.198 | 0.732 ± 0.099 | 0.698 | 18/24 | 60: 12 |
| logreg | v1_9_fixed | 0.490 | 0.807 | 0.610 | 0.680 | 0.730 | 0.719 ± 0.257 | 0.584 ± 0.162 | 0.077 | 12/24 | 60: 12 |
| logreg | v2_12 | 0.469 | 0.910 | 0.619 | 0.632 | 0.688 | 0.690 ± 0.273 | 0.612 ± 0.105 | 0.389 | 11/24 | 60: 12 |
| logreg | v2_12_no_pose | 0.538 | 0.798 | 0.643 | 0.680 | 0.735 | 0.744 ± 0.237 | 0.647 ± 0.102 | 0.502 | 16/24 | 60: 12 |
| svm | v1_9_fixed | 0.505 | 0.850 | 0.633 | 0.729 | 0.770 | 0.776 ± 0.238 | 0.618 ± 0.143 | 0.183 | 12/24 | 60: 12 |
| svm | v2_12 | 0.491 | 0.867 | 0.627 | 0.694 | 0.738 | 0.767 ± 0.262 | 0.618 ± 0.100 | 0.256 | 13/24 | 30: 1, 60: 11 |
| svm | v2_12_no_pose | 0.582 | 0.865 | 0.696 | 0.683 | 0.709 | 0.757 ± 0.213 | 0.700 ± 0.132 | 0.482 | 15/24 | 60: 12 |
| xgboost | v1_9_fixed | 0.555 | 0.672 | 0.608 | 0.657 | 0.676 | 0.661 ± 0.252 | 0.574 ± 0.231 | 0.000 | 14/24 | 15: 1, 30: 2, 60: 9 |
| xgboost | v2_12 | 0.563 | 0.725 | 0.634 | 0.681 | 0.678 | 0.686 ± 0.251 | 0.610 ± 0.213 | 0.000 | 14/24 | 15: 1, 30: 2, 60: 9 |
| xgboost | v2_12_no_pose | 0.554 | 0.800 | 0.654 | 0.662 | 0.657 | 0.744 ± 0.146 | 0.651 ± 0.110 | 0.278 | 15/24 | 30: 1, 60: 11 |

## How much did the v1 random split inflate the score?

Same v2 features, 15 s windows, v1 hyperparameters, threshold 0.5. Only the split differs.

| Model | Random split F1 | Random split ROC-AUC | LOSO F1 | LOSO ROC-AUC | F1 inflation |
|---|---|---|---|---|---|
| xgboost | 0.990 | 1.000 | 0.572 | 0.600 | +0.418 |
| svm | 0.874 | 0.946 | 0.576 | 0.577 | +0.298 |

v1 reported XGBoost F1 0.934 / ROC-AUC 0.994 (random split, 2 fps, sideways frames, duplicate video).

## Gate (pre-registered rules)

- ML candidate: **svm__v2_12** (best per-participant ROC-AUC on v2_12); baseline: rule__perclos_only.
  - beats_baseline: FAIL (candidate_pp_auc=0.7671, baseline_pp_auc=0.8112)
  - min_pooled_auc: FAIL (candidate_pooled_auc=0.6938, required=0.7)
  - beats_champion: FAIL (champion_version=3, champion_pp_auc=0.8112, required_gain=0.0)
- ML gate: **FAILED**
- Deployment policy `deploy_baseline_if_ml_fails` (added after the first run; gate rules unchanged):
  - baseline min_pooled_auc: pass (baseline_pooled_auc=0.7586, required=0.7)
  - baseline beats_champion: pass (note=reproduces champion v3 exactly)
- Deployed: **rule** — ML candidate failed; deploying the PERCLOS rule baseline (policy)

## Final model (fit on all 12 participants)

- rule on perclos_only, 60 s windows, config {}, trees None, threshold 0.0191 (from grouped out-of-fold scores).
- MLflow: `drowsiness_classifier_v2` version 4, alias `@champion`, run `8ac10dad91b4469594e448827988aafb`.
- Expected performance on a NEW person = the nested LOSO numbers above, not a training score.

## Per participant (rule__perclos_only)

| Participant | Window (s) | Windows | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | 60 | 543 | 0.544 | 0.893 | 0.676 | 0.650 |
| 2 | 60 | 548 | 0.850 | 0.787 | 0.818 | 0.893 |
| 3 | 60 | 596 | 0.473 | 1.000 | 0.642 | 1.000 |
| 4 | 60 | 556 | 0.773 | 0.835 | 0.803 | 0.868 |
| 5 | 60 | 548 | 0.719 | 0.801 | 0.757 | 0.827 |
| 6 | 60 | 461 | 0.412 | 0.913 | 0.568 | 0.433 |
| 7 | 60 | 543 | 0.517 | 1.000 | 0.682 | 0.948 |
| 8 | 60 | 543 | 0.728 | 1.000 | 0.842 | 0.997 |
| 9 | 60 | 578 | 0.532 | 0.997 | 0.694 | 0.480 |
| 10 | 60 | 527 | 0.708 | 0.698 | 0.703 | 0.650 |
| 11 | 60 | 566 | 0.900 | 0.989 | 0.942 | 0.988 |
| 12 | 60 | 472 | 0.491 | 1.000 | 0.659 | 1.000 |

## Caveats

- Labels are per video (self-reported KSS), not per moment: a 'drowsy' video contains alert moments.
- 12 participants is small; the spread across people matters as much as the mean.
- The deployed model family was picked by comparing outer LOSO results (mild optimism). Everything inside a fold (window, hyperparameters, trees, threshold) was chosen without the test person.
- RLDD was recorded at a desk, not while driving.
