# EXPLORATORY: per-driver centering + monotonic XGBoost (Fold 1 only)

Fold 1 has been examined many times, so treat these numbers as hints for what to test on new folds,
not as final results. Protocol: nested leave-one-participant-out, as in the main pipeline.
Centering = feature minus the driver's own median over all their windows (no labels).

| Variant | ROC-AUC / person | Pooled ROC-AUC | Pooled F1 | Precision | Recall | Specificity | Better than rule for | Mean diff | Wilcoxon p | Worst person |
|---|---|---|---|---|---|---|---|---|---|---|
| rule (raw) | 0.811 ± 0.198 | 0.759 | 0.725 | 0.603 | 0.908 | 0.420 | 0/12 | +0.000 | None | 0.43 |
| rule (centered) | 0.811 ± 0.198 | 0.837 | 0.769 | 0.751 | 0.789 | 0.746 | 0/12 | +0.000 | None | 0.43 |
| logreg (centered) | 0.688 ± 0.237 | 0.678 | 0.609 | 0.476 | 0.844 | 0.100 | 3/12 | -0.123 | 0.1475 | 0.25 |
| logreg (centered, no pose) | 0.740 ± 0.247 | 0.770 | 0.702 | 0.672 | 0.734 | 0.652 | 2/12 | -0.071 | 0.084 | 0.23 |
| svm (centered) | 0.723 ± 0.203 | 0.749 | 0.715 | 0.614 | 0.857 | 0.477 | 4/12 | -0.088 | 0.4922 | 0.36 |
| svm (centered, no pose) | 0.763 ± 0.236 | 0.794 | 0.715 | 0.660 | 0.781 | 0.610 | 3/12 | -0.048 | 0.2754 | 0.25 |
| xgb monotonic (raw) | 0.695 ± 0.257 | 0.678 | 0.630 | 0.540 | 0.757 | 0.375 | 4/12 | -0.116 | 0.3652 | 0.17 |
| xgb monotonic (raw, no pose) | 0.782 ± 0.148 | 0.711 | 0.649 | 0.585 | 0.730 | 0.498 | 4/12 | -0.029 | 0.3203 | 0.48 |
| xgb monotonic (centered) | 0.763 ± 0.194 | 0.752 | 0.725 | 0.688 | 0.766 | 0.663 | 5/12 | -0.048 | 0.5566 | 0.45 |
| xgb monotonic (centered, no pose) | 0.795 ± 0.166 | 0.811 | 0.748 | 0.718 | 0.781 | 0.704 | 4/12 | -0.016 | 0.7344 | 0.49 |

## ROC-AUC per held-out person

| Person | rule (raw) | rule (centered) | logreg (centered) | logreg (centered, no pose) | svm (centered) | svm (centered, no pose) | xgb monotonic (raw) | xgb monotonic (raw, no pose) | xgb monotonic (centered) | xgb monotonic (centered, no pose) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.65 | 0.65 | 0.75 | 0.59 | 0.81 | 0.62 | 0.73 | 0.64 | 0.76 | 0.74 |
| 2 | 0.89 | 0.89 | 0.81 | 0.77 | 0.45 | 0.78 | 0.82 | 0.83 | 0.64 | 0.88 |
| 3 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 0.96 | 0.87 | 1.00 | 1.00 |
| 4 | 0.87 | 0.87 | 0.42 | 0.48 | 0.56 | 0.51 | 0.23 | 0.92 | 0.81 | 0.77 |
| 5 | 0.83 | 0.83 | 0.77 | 0.77 | 0.71 | 0.80 | 0.70 | 0.78 | 0.89 | 0.83 |
| 6 | 0.43 | 0.43 | 0.59 | 0.53 | 0.62 | 0.49 | 0.71 | 0.82 | 0.45 | 0.66 |
| 7 | 0.95 | 0.95 | 0.80 | 0.94 | 0.81 | 0.94 | 0.16 | 0.61 | 0.92 | 0.93 |
| 8 | 1.00 | 1.00 | 0.25 | 0.99 | 0.36 | 0.96 | 0.82 | 0.79 | 0.59 | 0.68 |
| 9 | 0.48 | 0.48 | 0.42 | 0.23 | 0.71 | 0.25 | 0.54 | 0.48 | 0.50 | 0.49 |
| 10 | 0.65 | 0.65 | 0.51 | 0.57 | 0.64 | 0.80 | 0.70 | 0.67 | 0.61 | 0.57 |
| 11 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | 1.00 | 1.00 |
| 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
