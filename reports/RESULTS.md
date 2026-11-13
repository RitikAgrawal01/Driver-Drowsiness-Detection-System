# Results: subject-wise evaluation (v2)

## Data

- UTA-RLDD folds [1, 2]: 24 participants, 48 videos (alert = KSS 0, drowsy = KSS 10). Fold 4 is sealed for the final test and was not used.
- 10 fps by timestamp, rotation metadata applied: 293,332 frames, face found in 293,099 (99.92 %).
- Windows (2 s stride, face in >= 80 % of expected frames): 15 s: 14,328 (dropped 7), 30 s: 13,978 (dropped 0), 60 s: 13,258 (dropped 0)

## Nested leave-one-participant-out

Drowsy = positive class. **Pooled** = all 12 held-out participants' windows together. **Per participant** = metric on each held-out person, then mean ± std. **Videos** = majority vote of window predictions per video (24 videos).

| Model | Features | Precision | Recall | F1 | ROC-AUC | PR-AUC | ROC-AUC / participant | F1 / participant | Worst recall | Videos | Window chosen (s: folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | perclos_only | 0.618 | 0.918 | 0.739 | 0.775 | 0.788 | 0.859 ± 0.235 | 0.753 ± 0.109 | 0.642 | 35/48 | 60: 24 |
| logreg | v1_9_fixed | 0.546 | 0.850 | 0.665 | 0.697 | 0.721 | 0.786 ± 0.244 | 0.646 ± 0.163 | 0.000 | 27/48 | 60: 24 |
| logreg | v2_12 | 0.521 | 0.845 | 0.644 | 0.693 | 0.728 | 0.737 ± 0.254 | 0.633 ± 0.109 | 0.169 | 26/48 | 60: 24 |
| logreg | v2_12_no_pose | 0.585 | 0.859 | 0.696 | 0.727 | 0.761 | 0.772 ± 0.256 | 0.697 ± 0.144 | 0.324 | 32/48 | 15: 2, 60: 22 |
| svm | v1_9_fixed | 0.534 | 0.873 | 0.663 | 0.634 | 0.645 | 0.663 ± 0.334 | 0.658 ± 0.098 | 0.342 | 28/48 | 30: 2, 60: 22 |
| svm | v2_12 | 0.530 | 0.925 | 0.674 | 0.624 | 0.622 | 0.638 ± 0.334 | 0.678 ± 0.063 | 0.547 | 29/48 | 60: 24 |
| svm | v2_12_no_pose | 0.562 | 0.851 | 0.677 | 0.713 | 0.712 | 0.741 ± 0.260 | 0.675 ± 0.143 | 0.267 | 31/48 | 30: 6, 60: 18 |
| xgboost | v1_9_fixed | 0.610 | 0.884 | 0.722 | 0.718 | 0.698 | 0.783 ± 0.247 | 0.731 ± 0.124 | 0.458 | 33/48 | 60: 24 |
| xgboost | v2_12 | 0.601 | 0.872 | 0.712 | 0.745 | 0.720 | 0.819 ± 0.212 | 0.723 ± 0.139 | 0.401 | 33/48 | 30: 1, 60: 23 |
| xgboost | v2_12_no_pose | 0.599 | 0.879 | 0.713 | 0.723 | 0.711 | 0.827 ± 0.181 | 0.720 ± 0.120 | 0.460 | 32/48 | 30: 3, 60: 21 |

## How much did the v1 random split inflate the score?

Same v2 features, 15 s windows, v1 hyperparameters, threshold 0.5. Only the split differs.

| Model | Random split F1 | Random split ROC-AUC | LOSO F1 | LOSO ROC-AUC | F1 inflation |
|---|---|---|---|---|---|
| xgboost | 0.985 | 0.999 | 0.598 | 0.609 | +0.388 |
| svm | 0.859 | 0.933 | 0.533 | 0.536 | +0.326 |

v1 reported XGBoost F1 0.934 / ROC-AUC 0.994 (random split, 2 fps, sideways frames, duplicate video).

## Gate (pre-registered rules)

- ML candidate: **xgboost__v2_12** (best per-participant ROC-AUC on v2_12); baseline: rule__perclos_only.
  - beats_baseline: FAIL (candidate_pp_auc=0.8185, baseline_pp_auc=0.8592)
  - min_pooled_auc: pass (candidate_pooled_auc=0.7451, required=0.7)
  - beats_champion: FAIL (champion_version=4, champion_family=rule__perclos_only, champion_pp_auc_on_current_data=0.8592, required_gain=0.0)
- ML gate: **FAILED**
- Deployment policy `deploy_baseline_if_ml_fails` (added after the first run; gate rules unchanged):
  - baseline min_pooled_auc: pass (baseline_pooled_auc=0.7747, required=0.7)
  - baseline beats_champion: pass (note=same model family as champion v4, re-trained on current data)
- Deployed: **rule** — ML candidate failed; deploying the PERCLOS rule baseline (policy)

## Final model (fit on all 24 participants)

- rule on perclos_only, 60 s windows, config {}, trees None, threshold 0.0201 (from grouped out-of-fold scores).
- MLflow: `drowsiness_classifier_v2` version 5, alias `@champion`, run `937099accb8e4704be57501908ae7958`.
- Expected performance on a NEW person = the nested LOSO numbers above, not a training score.

## Per participant (rule__perclos_only)

| Participant | Window (s) | Windows | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | 60 | 543 | 0.542 | 0.875 | 0.669 | 0.650 |
| 2 | 60 | 548 | 0.867 | 0.740 | 0.798 | 0.893 |
| 3 | 60 | 596 | 0.474 | 1.000 | 0.643 | 1.000 |
| 4 | 60 | 556 | 0.799 | 0.759 | 0.779 | 0.868 |
| 5 | 60 | 548 | 0.719 | 0.801 | 0.757 | 0.827 |
| 6 | 60 | 461 | 0.412 | 0.843 | 0.553 | 0.433 |
| 7 | 60 | 543 | 0.520 | 1.000 | 0.684 | 0.948 |
| 8 | 60 | 543 | 0.728 | 1.000 | 0.842 | 0.997 |
| 9 | 60 | 578 | 0.526 | 0.957 | 0.679 | 0.480 |
| 10 | 60 | 527 | 0.704 | 0.642 | 0.672 | 0.650 |
| 11 | 60 | 566 | 0.899 | 0.950 | 0.924 | 0.988 |
| 12 | 60 | 472 | 0.493 | 1.000 | 0.661 | 1.000 |
| 13 | 60 | 558 | 0.993 | 0.986 | 0.989 | 1.000 |
| 14 | 60 | 559 | 0.742 | 0.968 | 0.840 | 0.964 |
| 15 | 60 | 571 | 0.489 | 1.000 | 0.656 | 0.984 |
| 16 | 60 | 479 | 1.000 | 0.768 | 0.869 | 0.999 |
| 17 | 60 | 548 | 1.000 | 0.913 | 0.954 | 1.000 |
| 18 | 60 | 562 | 0.516 | 1.000 | 0.681 | 0.999 |
| 19 | 60 | 543 | 0.501 | 1.000 | 0.667 | 0.053 |
| 20 | 60 | 627 | 0.553 | 1.000 | 0.712 | 0.948 |
| 21 | 60 | 545 | 0.554 | 1.000 | 0.713 | 1.000 |
| 22 | 60 | 550 | 0.643 | 1.000 | 0.783 | 0.996 |
| 23 | 60 | 538 | 0.969 | 0.788 | 0.869 | 0.943 |
| 24 | 60 | 697 | 0.511 | 1.000 | 0.676 | 1.000 |

## Caveats

- Labels are per video (self-reported KSS), not per moment: a 'drowsy' video contains alert moments.
- 24 participants is still small; the spread across people matters as much as the mean.
- The deployed model family was picked by comparing outer LOSO results (mild optimism). Everything inside a fold (window, hyperparameters, trees, threshold) was chosen without the test person.
- RLDD was recorded at a desk, not while driving.
