# Evaluation Under Extreme Class Imbalance: Leakage, Review Budgets, and Credit-Card Fraud Detection

**Author(s):** [Add author name(s)]  
**Affiliation:** [Add institution]  
**Status:** Draft template; empirical result fields are intentionally blank until experiments are run.

## Abstract

Credit-card fraud detection is commonly evaluated on data in which fraudulent transactions form a very small minority. This study evaluates how resampling placement and metric choice affect conclusions about model usefulness under constrained manual review. Using the ULB credit-card transaction benchmark, we compare Logistic Regression, Random Forest, and XGBoost with no resampling, class weighting, or SMOTE applied only within each training fold. We additionally compare Isolation Forest and Local Outlier Factor as unsupervised ranking baselines. Models are evaluated with repeated stratified cross-validation using average precision, ROC-AUC, and precision, recall, and F1 at fixed review capacities. A deliberately invalid SMOTE-before-split experiment illustrates why resampling must be confined to training data. We also report an illustrative cost curve and paired Wilcoxon comparisons, with inference qualified for dependence among cross-validation folds. **Results:** [Insert measured finding from `outputs/results_report.md`; do not leave this sentence as a claim until the run is complete.] **Conclusion:** [Insert a conclusion supported by results and limitations.]

**Keywords:** credit-card fraud; class imbalance; SMOTE; data leakage; precision-recall; alert budget

## 1. Introduction

Fraud detection is an imbalanced classification problem: a model may correctly rank most genuine transactions while still producing too many false alerts or missing a substantial share of fraud. The benchmark dataset used here contains 284,807 transactions from two days, including 492 fraudulent transactions (0.172%); its predictors are numerical, with V1–V28 derived through PCA and with Time and Amount retained separately [1]. The dataset provider recommends precision-recall-based evaluation under this class imbalance [1].

ROC-AUC is a useful ranking measure, but it does not directly answer how many of the transactions sent to a limited review team will be fraudulent. Davis and Goadrich show that precision-recall curves can expose performance differences obscured in ROC space under strong class skew [2]. We therefore report average precision (the implementation’s AUPRC summary) and metrics at explicit review capacities alongside ROC-AUC.

Resampling also changes what the evaluation measures if it is performed before the train/test split. A synthetic training point can be generated using information from an observation later treated as test data, and a resampled test set no longer reflects the deployment class distribution. The imbalanced-learn guidance describes this as a source of over-optimistic evaluation and recommends placing sampling in a pipeline so it is fitted only on training folds [3]. A 2024 study empirically documents bias from applying oversampling before cross-validation, although its radiomics findings should not be assumed to quantify bias on this fraud dataset [4].

This paper addresses three questions:

1. How do supervised classifiers and unsupervised anomaly detectors rank fraud under repeated stratified evaluation?
2. How do no-resampling, class-weighting, and train-fold-only SMOTE compare when ranking quality and review-budget utility are considered together?
3. How different are results under a deliberately invalid global-SMOTE-before-split procedure, and why must those results not be presented as valid test performance?

## 2. Related work

SMOTE generates synthetic minority-class examples by interpolating between minority observations [5]. It is a training-time strategy, not a method for balancing the validation distribution. This study treats “SMOTE” as the pipeline strategy `StandardScaler → SMOTE → estimator`, where both scaling and sampling are fitted separately within each training fold.

For imbalanced classification, ROC-AUC and precision-recall summaries emphasize different aspects of a ranking. Precision explicitly accounts for false positives relative to the predicted-positive set, which matters when genuine transactions greatly outnumber fraud. Because operational screening has a finite capacity, we complement curve summaries with top-k precision and recall at fixed percentages of the fold’s transactions. These capacities are evaluation scenarios, not claimed real-world staffing levels.

## 3. Data and methods

### 3.1 Dataset and features

The source is the public ULB *Credit Card Fraud Detection* dataset distributed through Kaggle [1]. The source description reports 284,807 European cardholder transactions over two days, with 492 frauds. V1–V28 are anonymized PCA-derived variables; Time is elapsed seconds from the first transaction and Amount is the transaction amount. The target is `Class`, where 1 denotes fraud. The run configuration records the actual row count, fraud count, prevalence, and selected feature names read by the scripts.

The benchmark is a historical, short-duration dataset with anonymized features. Accordingly, results measure performance on this benchmark and do not establish present-day or production effectiveness.

### 3.2 Validation protocol and leakage control

The primary evaluation uses `RepeatedStratifiedKFold` with 5 folds and 2 repeats by default (10 outer test folds), shuffled with seed 42. The experiment runner records the seed and the software environment. All preprocessing is fitted within each training fold. For SMOTE configurations, the imbalanced-learn pipeline fits StandardScaler and SMOTE on training rows only; the outer test fold remains unresampled. No globally fitted scaler, PCA transform, or sampler is used by the primary experiment.

The compared supervised estimators are Logistic Regression, Random Forest, and XGBoost. Each is evaluated under three strategies: no resampling, model-level class weighting, and SMOTE. Isolation Forest and novelty-mode Local Outlier Factor are fitted without labels and produce anomaly scores. To control LOF runtime, its training data are capped at 5,000 rows by a reproducible uniform sample selected without using labels. This cap is part of the method and must be reported if changed.

### 3.3 Leakage demonstration

A separate script contrasts a valid stratified split followed by train-only SMOTE with an intentionally invalid global-SMOTE-before-split workflow. In the invalid branch, the holdout is drawn from resampled data and is both synthetically balanced and potentially related to points used for training. Therefore, the difference between rows is an illustration of protocol failure, not an isolated causal estimate of the size of information leakage. Only the untouched natural-prevalence test result is eligible for reporting as a valid holdout estimate.

### 3.4 Evaluation measures

Average precision is used as the main ranking summary for the precision-recall curve. ROC-AUC is reported as a complementary metric. For each outer test fold, transactions are ranked by score and the top `ceil(b × n)` transactions are selected for review, where `n` is the fold size and `b` is 0.1%, 0.5%, or 1%. Precision, recall, and F1 are calculated for these exact top-k alert sets. This defines the operating point by review capacity rather than by a universal probability threshold of 0.5. Anomaly scores from unsupervised detectors are rankings and are not interpreted as calibrated probabilities.

An illustrative cost curve uses review capacities of 0.1%, 0.25%, 0.5%, 1%, 2%, and 5%. The default assumed cost is 1,000 relative units per missed fraud and 10 per false alert. At a given capacity, total cost is `missed_fraud_count × missed_fraud_cost + false_alert_count × false_alert_cost`. These values are sensitivity-analysis assumptions, not measured economic costs. Readers should examine alternate cost ratios before operational interpretation.

Results are summarized as mean ± standard deviation over outer folds. The scripts also perform paired two-sided Wilcoxon signed-rank comparisons of fold-level AUPRC for each configuration pair and apply Holm correction across comparisons. Because repeated-CV folds share observations, these p-values are exploratory rather than definitive independent-sample inference; a separate temporal or external test set is preferable for confirmatory claims.

### 3.5 Optional label-efficiency analysis

The optional script retains all negative training examples and varies the number of labeled fraud examples used by supervised models (1%, 5%, 10%, 25%, and 100% of the training-fold frauds). It compares a single stratified holdout against unsupervised models fitted without labels. This one-split analysis is exploratory and is not combined with the repeated-CV estimates.

## 4. Results

**Do not complete this section until the commands in the README have been run on the dataset.** Copy values from generated CSV/Markdown outputs and state the fold and repeat counts. Do not report the invalid global-SMOTE branch as test performance.

### 4.1 Main repeated-CV comparison

[Insert the measured `outputs/results_report.md` table. Identify the best AUPRC configuration and report its mean ± standard deviation. Then compare its review-budget precision and recall to the other leading configurations. Do not infer significance solely from rank order.]

### 4.2 Resampling strategy and model comparison

[Summarize no resampling, class weighting, and train-fold-only SMOTE using `outputs/summary.csv`. State whether observed differences are consistent across folds.]

### 4.3 Review-budget and cost analysis

[Insert selected values from `outputs/summary.csv` and `outputs/cost_curve_summary.csv`. Explain the chosen cost assumptions and include `outputs/cost_curve.png` if the paper format permits. State that costs are illustrative.]

### 4.4 Leakage demonstration

[Report both observed metrics and test prevalences from `outputs/leakage_demo.csv`, clearly labeling the invalid result as invalid. Explain that the test populations differ, so the numerical gap is not an isolated estimate of leakage inflation.]

### 4.5 Statistical comparison

[Use `outputs/wilcoxon_auprc.csv`. Report test statistic, Holm-adjusted p-value, number of paired splits, and the dependence caveat. Avoid causal language.]

## 5. Discussion

[Interpret the measured results in terms of ranking and review capacity. Discuss cases where ROC-AUC and AUPRC tell different stories. Distinguish improvement in AUPRC from improvement in the precision/recall attainable at a given review capacity. Explain any model/strategy trade-offs without extrapolating to live fraud operations.]

## 6. Limitations

The dataset covers only two days, has anonymized PCA features, and may not represent contemporary transactions or temporal drift [1]. Stratified random folds can place observations from nearby periods in both training and test partitions; results therefore do not substitute for a chronological validation. The default cost ratios are illustrative. Unsupervised detectors are evaluated by their score rankings, while model scores may differ in calibration. LOF uses a capped training sample. Repeated-CV fold dependence limits the interpretation of signed-rank p-values. Finally, conclusions apply only to the selected dataset, features, software versions, model settings, and random seeds recorded in the run metadata.

## 7. Conclusion

[Write a brief conclusion only after the results are available. State the key observed methodological lesson and the best-supported review-budget finding, then qualify generalization to production fraud data.]

## References

[1] ULB Machine Learning Group. *Credit Card Fraud Detection*. Kaggle dataset description. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

[2] Jesse Davis and Mark Goadrich. “The Relationship Between Precision-Recall and ROC Curves.” Proceedings of the 23rd International Conference on Machine Learning, 2006. https://mark.goadrich.com/articles/davisgoadrichcamera2.pdf

[3] imbalanced-learn developers. “Common pitfalls and recommended practices.” https://imbalanced-learn.org/stable/common_pitfalls.html

[4] Aydin Demircioğlu. “Applying oversampling before cross-validation will lead to high bias in radiomics.” *Scientific Reports*, 14, 11563 (2024). https://www.nature.com/articles/s41598-024-62585-z

[5] Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. “SMOTE: Synthetic Minority Over-sampling Technique.” *Journal of Artificial Intelligence Research*, 16, 321–357 (2002). https://doi.org/10.1613/jair.953
