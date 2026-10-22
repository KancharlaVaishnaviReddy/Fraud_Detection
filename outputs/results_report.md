# Measured experiment results

Run settings: 5-fold stratified CV × 2 repeats; seed 42.
Dataset: 284,807 rows, 30 features, 492 frauds (0.1727%).
All entries below come from this execution; no results are pre-filled in the project.

## Ranking and review-budget metrics

| config                            |   auprc_mean |   auprc_std |   roc_auc_mean |   roc_auc_std |   precision_at_0.1pct_mean |   precision_at_0.5pct_mean |   precision_at_1pct_mean |   recall_at_1pct_mean |   f1_at_1pct_mean |
|:----------------------------------|-------------:|------------:|---------------:|--------------:|---------------------------:|---------------------------:|-------------------------:|----------------------:|------------------:|
| XGBoost | none                    |      0.86163 |     0.02222 |        0.98232 |       0.00631 |                    0.98772 |                    0.30421 |                  0.15439 |               0.89433 |           0.26332 |
| RandomForest | smote              |      0.85405 |     0.02692 |        0.98100 |       0.00734 |                    0.98070 |                    0.30386 |                  0.15509 |               0.89836 |           0.26451 |
| RandomForest | class_weight       |      0.85043 |     0.02781 |        0.95643 |       0.01250 |                    0.97544 |                    0.30421 |                  0.15368 |               0.89024 |           0.26212 |
| XGBoost | class_weight            |      0.84975 |     0.02597 |        0.98139 |       0.00703 |                    0.98421 |                    0.30281 |                  0.15386 |               0.89124 |           0.26242 |
| RandomForest | none               |      0.84935 |     0.02566 |        0.95682 |       0.00920 |                    0.97895 |                    0.30175 |                  0.15351 |               0.88922 |           0.26182 |
| XGBoost | smote                   |      0.84006 |     0.02928 |        0.97714 |       0.00555 |                    0.97368 |                    0.30070 |                  0.15281 |               0.88515 |           0.26062 |
| LogisticRegression | none         |      0.75917 |     0.02300 |        0.97437 |       0.00829 |                    0.87018 |                    0.29754 |                  0.15088 |               0.87403 |           0.25733 |
| LogisticRegression | smote        |      0.73501 |     0.03208 |        0.97708 |       0.00685 |                    0.87193 |                    0.29404 |                  0.15211 |               0.88111 |           0.25943 |
| LogisticRegression | class_weight |      0.73293 |     0.03455 |        0.97870 |       0.00680 |                    0.86667 |                    0.29509 |                  0.15298 |               0.88620 |           0.26092 |
| LOF | unsupervised                |      0.38402 |     0.03715 |        0.95609 |       0.00994 |                    0.53333 |                    0.25439 |                  0.14000 |               0.81095 |           0.23878 |
| IsolationForest | unsupervised    |      0.15856 |     0.04414 |        0.94862 |       0.01163 |                    0.31754 |                    0.14421 |                  0.09877 |               0.57223 |           0.16846 |

## Interpretation notes

AUPRC is reported as average precision. Review-budget precision/recall/F1 use exactly the top-k ranked alerts, where k is the ceiling of the stated percentage of each outer test fold. F1 is therefore shown at a stated review capacity rather than at an arbitrary 0.5 probability threshold.

Cost curves assume 1000 relative units per missed fraud and 10 per false alert. These are illustrative assumptions, not estimated real-world costs.

The paired Wilcoxon results are exploratory: repeated-CV folds share observations and are not independent replicates. Interpret adjusted p-values cautiously; confirmatory inference needs independent temporal or external test data.
