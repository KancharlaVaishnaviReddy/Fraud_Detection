# Credit-card fraud detection: evaluation under extreme imbalance

This project rebuilds the supplied notebook as a reproducible experiment suite. Its central question is whether evaluation choices—especially resampling before the split, headline ROC-AUC, and a fixed 0.5 threshold—make fraud models look more useful than they are under a limited manual-review capacity.

The project contains scripts and a research-paper draft. **It does not contain made-up benchmark scores.** Run the experiments on the actual CSV first; the scripts write measured results to `outputs/`.

## 1. Requirements and installation

Use Python 3.10–3.12. You need internet access for the first installation and about 150 MB of disk space for the Kaggle CSV, plus room for Python packages and experiment outputs. A computer with several CPU cores is recommended. The complete repeated-CV suite is compute-intensive because it fits multiple models and SMOTE pipelines in every fold.

### Windows (PowerShell)

```powershell
cd path\to\fraud-detection-research
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
cd path/to/fraud-detection-research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not skip the virtual environment if you already have other Python projects installed; it keeps this project's dependencies separate.

## 2. Get the dataset

Download `creditcard.csv` from the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Kaggle may require a free account and acceptance of its dataset terms. Put the file here:

```text
data/creditcard.csv
```

The CSV is not bundled because of its size and dataset distribution terms. The script expects the label column `Class` (0 = genuine, 1 = fraud). It uses `V1`–`V28`, `Time`, and `Amount` when available. The original dataset contains 284,807 records and 492 frauds; the loader prints and records the actual counts from your copy.

Alternative location:

```bash
python run_experiments.py --data /full/path/to/creditcard.csv
```

## 3. Run the experiments

Start with a fast installation/data check by running the scripts with `--help`, then run the main benchmark from the project root:

```bash
python run_experiments.py
```

By default this uses 5-fold stratified cross-validation repeated twice with a fixed seed. It evaluates Logistic Regression, Random Forest, and XGBoost under no resampling, class weighting, and SMOTE inside an imbalanced-learn pipeline. It also evaluates Isolation Forest and Local Outlier Factor (LOF). LOF is fit on a deterministic, label-independent random sample of up to 5,000 training-fold rows to keep computation manageable; all test-fold rows remain untouched.

The suite can take a long time on a laptop. For a quick functional run (not a paper-quality result), use:

```bash
python run_experiments.py --folds 3 --repeats 1 --jobs 2
```

For the planned 5-fold repeated run, increase the repetitions if compute permits:

```bash
python run_experiments.py --folds 5 --repeats 3 --seed 42
```

The values shown in README examples are run settings, not performance claims. Do not replace the results in a paper with the short smoke run unless you report its settings.

### Leakage comparison

Run this separately:

```bash
python leakage_demo.py
```

It writes `outputs/leakage_demo.csv`. The “invalid” row intentionally runs SMOTE before the split. It has a balanced, synthetic holdout that does not reflect deployment prevalence and can contain information derived from future test rows. The two rows are therefore **not a clean causal estimate of the size of leakage** and are not directly comparable as valid test results. Use the correct train-only row for reporting. The invalid row is included only to show how a common mistake changes the measured evaluation.

### Optional label-efficiency study

```bash
python label_efficiency.py
```

This uses one stratified train/test split, keeps the training negatives, and varies the fraction of training fraud labels available to supervised Logistic Regression, Random Forest, and XGBoost. It compares them with unsupervised detectors fitted without labels. It is an exploratory single-split analysis, not repeated-CV evidence.

## 4. Outputs

The main command creates or overwrites these files in `outputs/`:

- `fold_results.csv`: one row per model, strategy, and outer fold.
- `summary.csv`: mean and standard deviation across outer folds.
- `wilcoxon_auprc.csv`: paired fold-level Wilcoxon signed-rank comparisons with Holm adjustment.
- `cost_curve_summary.csv` and `cost_curve.png`: cost summaries at review capacities from 0.1% through 5%.
- `results_report.md`: measured, ready-to-review tables and methodology notes.
- `run_config.json`: dataset counts, feature names, seeds, settings, platform, and library versions.

Metrics include average precision (AUPRC), ROC-AUC, precision/recall/F1 at fixed review budgets (0.1%, 0.5%, and 1%), and illustrative cost at review budgets from 0.1% to 5%. Top-k selection uses exactly `ceil(budget × fold test rows)` alerts. The default cost assumptions are 1,000 relative units per missed fraud and 10 per false alert. Change them with `--missed-fraud-cost` and `--false-alert-cost`. They are not actual estimates of bank losses or investigator costs.

The paired Wilcoxon test is exploratory. Repeated-CV folds share observations, so they are not independent replications. Report the p-values cautiously; independent temporal or external validation is preferable for confirmatory inference.

## 5. Research-paper draft

Open [`paper/research_paper_draft.md`](paper/research_paper_draft.md). It includes a literature-backed motivation, research questions, methods, limitations, and explicit placeholders for measured results. Fill in those placeholders only after reviewing `outputs/results_report.md` and the CSVs. Add your author name, institution, and final citations/style as required by your school or target venue.

## 6. Reproducibility and caveats

- The seed, split specification, versions, and dataset features are logged. Different hardware/library versions can still cause small numerical variation.
- Scalers and SMOTE are fitted inside each training fold. Validation folds preserve the observed, highly imbalanced distribution.
- F1 is reported at a stated review capacity. This avoids pretending that 0.5 is a universal operational threshold. The leakage demonstration separately reports F1 at 0.5 only to illustrate threshold-dependent results.
- The ULB file covers two days and has anonymized PCA-derived features. It is useful as a benchmark, but is not a current deployment sample and cannot support claims of generalization to contemporary fraud streams.
- Do not tune choices repeatedly against the same CV folds and then treat those folds as an untouched final test. For a stronger paper, consider a chronological holdout where valid timestamps are available and describe the dataset’s limited time span.

## 7. Troubleshooting

- **`Dataset not found`**: ensure the file is named `creditcard.csv` and is inside the project's `data` folder, or pass `--data` with the full path.
- **`ModuleNotFoundError`**: activate the virtual environment and run `pip install -r requirements.txt` again.
- **XGBoost installation fails**: upgrade pip, confirm Python is 3.10–3.12, and reinstall with `python -m pip install --upgrade pip xgboost`.
- **Run is too slow**: use `--jobs 2` (or a reasonable number for your CPU), reduce repeats only for a smoke run, or decrease `--lof-train-cap`. Avoid running multiple copies at once.
- **SMOTE says there are too few positives**: the default benchmark has enough positive samples; if you changed the data or folds, reduce `--smote-k` while keeping it smaller than the minority count in every training fold.
- **Notebook execution**: the original notebook is retained at `notebooks/fraud_detection_fixed_original.ipynb`; the reproducible experiment scripts are the recommended route for the research claims.
