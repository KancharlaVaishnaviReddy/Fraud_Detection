#!/usr/bin/env python3
"""Optional one-holdout label-efficiency experiment; unsupervised models never use labels for fitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.fraud_utils import (
    anomaly_scores, build_supervised, build_unsupervised, evaluate_scores, load_dataset,
)

FRAUD_LABEL_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--output", default="outputs/label_efficiency.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--lof-train-cap", type=int, default=5000)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}. See README.md for download instructions.")
    X, y, _ = load_dataset(args.data)
    train_i, test_i = train_test_split(
        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = y[train_i], y[test_i]
    pos_local = np.flatnonzero(y_train == 1)
    neg_local = np.flatnonzero(y_train == 0)
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, object]] = []

    for fraction in FRAUD_LABEL_FRACTIONS:
        n_pos = max(2, int(np.ceil(len(pos_local) * fraction)))
        n_pos = min(n_pos, len(pos_local))
        chosen_pos = np.sort(rng.choice(pos_local, size=n_pos, replace=False))
        chosen = np.concatenate([neg_local, chosen_pos])
        X_labeled, y_labeled = X_train.iloc[chosen], y_train[chosen]
        for model_name in ("LogisticRegression", "RandomForest", "XGBoost"):
            model = build_supervised(model_name, "class_weight", y_labeled, args.seed, args.jobs)
            model.fit(X_labeled, y_labeled)
            scores = anomaly_scores(model, X_test)
            results.append({
                "model": model_name, "label_fraction_of_train_frauds": fraction,
                "fraud_labels_used": n_pos, "negative_rows_used": len(neg_local),
                "test_fraud_count": int(y_test.sum()),
                **evaluate_scores(y_test, scores),
            })
            print(f"{model_name}, {fraction:.0%} fraud labels ({n_pos} positives): "
                  f"AUPRC={results[-1]['auprc']:.5f}")

    # Fit unsupervised models on training features only; labels are not passed to fit.
    for model_name in ("IsolationForest", "LOF"):
        model = build_unsupervised(model_name, args.seed, args.jobs)
        X_fit = X_train
        if model_name == "LOF" and len(X_fit) > args.lof_train_cap:
            selected = np.sort(rng.choice(len(X_fit), size=args.lof_train_cap, replace=False))
            X_fit = X_fit.iloc[selected]
        model.fit(X_fit)
        scores = anomaly_scores(model, X_test)
        results.append({
            "model": model_name, "label_fraction_of_train_frauds": 0.0,
            "fraud_labels_used": 0, "negative_rows_used": len(X_fit),
            "test_fraud_count": int(y_test.sum()),
            **evaluate_scores(y_test, scores),
        })
        print(f"{model_name}, no labels for fitting: AUPRC={results[-1]['auprc']:.5f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"\nSaved one stratified holdout comparison to {output}.")
    print("This optional analysis is a single split, not repeated-CV evidence; report it as exploratory.")


if __name__ == "__main__":
    main()
