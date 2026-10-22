#!/usr/bin/env python3
"""Leakage-safe repeated-CV benchmark for highly imbalanced fraud classification."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import imblearn
from sklearn.model_selection import RepeatedStratifiedKFold

from src.fraud_utils import (
    BUDGETS, COST_BUDGETS, anomaly_scores, build_supervised,
    build_unsupervised, evaluate_scores, load_dataset, paired_wilcoxon,
)

SUPERVISED = ("LogisticRegression", "RandomForest", "XGBoost")
STRATEGIES = ("none", "class_weight", "smote")
ALL_BUDGETS = tuple(sorted(set(BUDGETS + COST_BUDGETS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--output", default="outputs", help="Directory for result files")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1, help="CPU workers for tree/neighbor models")
    parser.add_argument("--lof-train-cap", type=int, default=5000,
                        help="Uniform training-sample cap for LOF; labels are not used for sampling")
    parser.add_argument("--missed-fraud-cost", type=float, default=1000.0,
                        help="Assumed relative cost units per missed fraud")
    parser.add_argument("--false-alert-cost", type=float, default=10.0,
                        help="Assumed relative cost units per false alert")
    parser.add_argument("--smote-k", type=int, default=5)
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.repeats < 1:
        raise SystemExit("--folds must be at least 2 and --repeats at least 1")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}\nDownload creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at that path, "
            "or pass --data /path/to/creditcard.csv."
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, feature_names = load_dataset(str(data_path))
    print(f"Loaded {len(y):,} rows, {len(feature_names)} features; fraud={int(y.sum()):,} ({y.mean():.4%})")
    print(f"CV: {args.folds} folds × {args.repeats} repeats; outputs -> {output.resolve()}")

    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    rows: list[dict[str, object]] = []
    n_splits = args.folds * args.repeats
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_id = split_id // args.folds
        fold_id = split_id % args.folds
        seed = args.seed + split_id
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\n[repeat {repeat_id + 1}/{args.repeats}, fold {fold_id + 1}/{args.folds}] "
              f"train={len(train_idx):,}, test={len(test_idx):,}, test fraud={int(y_test.sum())}")

        for model_name in SUPERVISED:
            for strategy in STRATEGIES:
                config = f"{model_name} | {strategy}"
                estimator = build_supervised(model_name, strategy, y_train, seed, args.jobs, args.smote_k)
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                scores = anomaly_scores(estimator, X_test)
                elapsed = time.perf_counter() - started
                metrics = evaluate_scores(y_test, scores, ALL_BUDGETS, args.missed_fraud_cost, args.false_alert_cost)
                rows.append({
                    "config": config, "model": model_name, "strategy": strategy,
                    "split_id": split_id, "repeat": repeat_id + 1, "fold": fold_id + 1,
                    "seed": seed, "train_rows": len(train_idx), "test_rows": len(test_idx),
                    "test_fraud_count": int(y_test.sum()), "fit_and_score_seconds": elapsed,
                    **metrics,
                })
                print(f"  {config}: AUPRC={metrics['auprc']:.5f}, ROC-AUC={metrics['roc_auc']:.5f} ({elapsed:.1f}s)")

        # Unsupervised estimators are fitted without labels and only see outer-fold training features.
        for model_name in ("IsolationForest", "LOF"):
            config = f"{model_name} | unsupervised"
            estimator = build_unsupervised(model_name, seed, args.jobs)
            if model_name == "LOF" and len(train_idx) > args.lof_train_cap:
                rng = np.random.default_rng(seed)
                chosen = np.sort(rng.choice(len(train_idx), size=args.lof_train_cap, replace=False))
                X_fit = X_train.iloc[chosen]
            else:
                X_fit = X_train
            started = time.perf_counter()
            estimator.fit(X_fit)
            scores = anomaly_scores(estimator, X_test)
            elapsed = time.perf_counter() - started
            metrics = evaluate_scores(y_test, scores, ALL_BUDGETS, args.missed_fraud_cost, args.false_alert_cost)
            rows.append({
                "config": config, "model": model_name, "strategy": "unsupervised",
                "split_id": split_id, "repeat": repeat_id + 1, "fold": fold_id + 1,
                "seed": seed, "train_rows": len(X_fit), "test_rows": len(test_idx),
                "test_fraud_count": int(y_test.sum()), "fit_and_score_seconds": elapsed,
                **metrics,
            })
            print(f"  {config}: AUPRC={metrics['auprc']:.5f}, ROC-AUC={metrics['roc_auc']:.5f} ({elapsed:.1f}s)")

        pd.DataFrame(rows).to_csv(output / "fold_results.csv", index=False)

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(output / "fold_results.csv", index=False)
    metric_columns = [c for c in fold_results.columns if c in {
        "auprc", "roc_auc", "fit_and_score_seconds"
    } or c.startswith(("precision_at_", "recall_at_", "f1_at_", "cost_at_"))]
    summary = fold_results.groupby(["model", "strategy", "config"])[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("auprc_mean", ascending=False)
    summary.to_csv(output / "summary.csv", index=False)

    tests = paired_wilcoxon(fold_results, "auprc")
    tests.to_csv(output / "wilcoxon_auprc.csv", index=False)

    # Fold-level review-cost summaries retain the declared costs and alert budgets.
    cost_cols = [f"cost_at_{100 * b:g}pct" for b in COST_BUDGETS]
    available_costs = [c for c in cost_cols if c in fold_results.columns]
    cost_summary = fold_results.groupby("config")[available_costs].agg(["mean", "std"])
    cost_summary.columns = [f"{metric}_{stat}" for metric, stat in cost_summary.columns]
    cost_summary.reset_index().to_csv(output / "cost_curve_summary.csv", index=False)

    config = {
        "dataset": str(data_path.resolve()), "rows": int(len(y)), "features": feature_names,
        "fraud_count": int(y.sum()), "fraud_prevalence": float(y.mean()),
        "folds": args.folds, "repeats": args.repeats, "seed": args.seed,
        "review_budgets": list(BUDGETS), "cost_curve_budgets": list(COST_BUDGETS),
        "missed_fraud_cost_assumption": args.missed_fraud_cost,
        "false_alert_cost_assumption": args.false_alert_cost,
        "lof_train_cap": args.lof_train_cap, "smote_k": args.smote_k,
        "python": platform.python_version(), "platform": platform.platform(),
        "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
    }
    try:
        import xgboost
        config["xgboost"] = xgboost.__version__
    except ImportError:
        config["xgboost"] = "not detected"
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Cost curves: cost on each test fold at the corresponding fixed alert capacity.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pct = np.array(COST_BUDGETS) * 100
    for config_name, group in fold_results.groupby("config"):
        means = [group[f"cost_at_{100 * b:g}pct"].mean() for b in COST_BUDGETS]
        ax.plot(x_pct, means, marker="o", linewidth=1.4, label=config_name)
    ax.set_xlabel("Transactions sent for manual review (% of fold test set)")
    ax.set_ylabel("Assumed total cost (relative cost units)")
    ax.set_title(f"Review-budget cost curve (missed fraud={args.missed_fraud_cost:g}, false alert={args.false_alert_cost:g})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output / "cost_curve.png", dpi=170)
    plt.close(fig)

    # Compact Markdown tables based only on this run's measured results.
    display_cols = ["config", "auprc_mean", "auprc_std", "roc_auc_mean", "roc_auc_std",
                    "precision_at_0.1pct_mean", "precision_at_0.5pct_mean", "precision_at_1pct_mean",
                    "recall_at_1pct_mean", "f1_at_1pct_mean"]
    selected = [c for c in display_cols if c in summary.columns]
    report_lines = [
        "# Measured experiment results", "",
        f"Run settings: {args.folds}-fold stratified CV × {args.repeats} repeats; seed {args.seed}.",
        f"Dataset: {len(y):,} rows, {len(feature_names)} features, {int(y.sum()):,} frauds ({y.mean():.4%}).",
        "All entries below come from this execution; no results are pre-filled in the project.", "",
        "## Ranking and review-budget metrics", "",
        summary[selected].to_markdown(index=False, floatfmt=".5f"), "",
        "## Interpretation notes", "",
        "AUPRC is reported as average precision. Review-budget precision/recall/F1 use exactly the top-k ranked alerts, "
        "where k is the ceiling of the stated percentage of each outer test fold. F1 is therefore shown at a stated "
        "review capacity rather than at an arbitrary 0.5 probability threshold.", "",
        f"Cost curves assume {args.missed_fraud_cost:g} relative units per missed fraud and "
        f"{args.false_alert_cost:g} per false alert. These are illustrative assumptions, not estimated real-world costs.", "",
        "The paired Wilcoxon results are exploratory: repeated-CV folds share observations and are not independent "
        "replicates. Interpret adjusted p-values cautiously; confirmatory inference needs independent temporal or external test data.",
        "",
    ]
    (output / "results_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("\nCompleted.")
    print(f"Summary: {output / 'summary.csv'}")
    print(f"Report:  {output / 'results_report.md'}")
    print(f"Cost plot: {output / 'cost_curve.png'}")


if __name__ == "__main__":
    main()
