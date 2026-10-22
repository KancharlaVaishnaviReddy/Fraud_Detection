#!/usr/bin/env python3
"""Compare train-only SMOTE with globally generated SMOTE samples on one shared real holdout."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.fraud_utils import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--output", default="outputs/leakage_demo.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--smote-k", type=int, default=5)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}. See README.md for download instructions.")

    X, y, _ = load_dataset(args.data)
    train_i, test_i = train_test_split(
        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = y[train_i], y[test_i]

    # Shared, train-fitted feature scaling keeps scaler leakage out of this comparison.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Correct protocol: the untouched real test rows are never used by SMOTE.
    X_train_sm, y_train_sm = SMOTE(
        random_state=args.seed, k_neighbors=args.smote_k
    ).fit_resample(X_train_scaled, y_train)
    correct = LogisticRegression(max_iter=1500, solver="lbfgs", random_state=args.seed)
    correct.fit(X_train_sm, y_train_sm)
    correct_scores = correct.predict_proba(X_test_scaled)[:, 1]

    # Deliberately flawed protocol: globally fit SMOTE before splitting, but preserve
    # the exact same original test fold for evaluation. We discard original held-out
    # rows from the training matrix and retain only globally generated synthetic rows.
    # Those synthetic rows can still interpolate from held-out fraud observations,
    # which is the leakage mechanism. The number of synthetic rows is chosen from
    # training-fold counts only so both training sets finish class-balanced and equal
    # in size; evaluation prevalence and observations are identical between methods.
    n_train_neg = int(np.sum(y_train == 0))
    n_train_pos = int(np.sum(y_train == 1))
    n_synthetic_needed = n_train_neg - n_train_pos
    n_global_positive = int(np.sum(y == 1))
    target_global_positive = n_global_positive + n_synthetic_needed
    if n_synthetic_needed <= 0:
        raise ValueError("Expected fewer fraud than genuine transactions in the training split.")

    # Keep the scaler train-fitted. Only SMOTE sees the held-out samples here.
    X_all_scaled = scaler.transform(X)
    X_global_resampled, y_global_resampled = SMOTE(
        sampling_strategy={1: target_global_positive},
        random_state=args.seed,
        k_neighbors=args.smote_k,
    ).fit_resample(X_all_scaled, y)
    X_global_synthetic = X_global_resampled[len(y):]
    y_global_synthetic = y_global_resampled[len(y):]
    if not np.all(y_global_synthetic == 1):
        raise RuntimeError("Expected SMOTE to append only positive-class synthetic observations.")

    X_bad_train = np.vstack([X_train_scaled, X_global_synthetic])
    y_bad_train = np.concatenate([y_train, y_global_synthetic])
    wrong = LogisticRegression(max_iter=1500, solver="lbfgs", random_state=args.seed)
    wrong.fit(X_bad_train, y_bad_train)
    wrong_scores = wrong.predict_proba(X_test_scaled)[:, 1]

    def row(protocol: str, valid: bool, train_rows: int, train_prevalence: float,
            scores: np.ndarray, note: str) -> dict[str, object]:
        predictions = scores >= 0.5
        return {
            "protocol": protocol,
            "valid_for_reporting": valid,
            "seed": args.seed,
            "train_rows": train_rows,
            "train_fraud_prevalence": train_prevalence,
            "test_rows": len(test_i),
            "test_fraud_count": int(y_test.sum()),
            "test_fraud_prevalence": float(y_test.mean()),
            "auprc": float(average_precision_score(y_test, scores)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "f1_at_0.5": float(f1_score(y_test, predictions, zero_division=0)),
            "note": note,
        }

    results = pd.DataFrame([
        row(
            "correct_train_only_smote", True, len(y_train_sm), float(y_train_sm.mean()), correct_scores,
            "Valid comparison arm: scaler and SMOTE fit on training rows only; test is untouched.",
        ),
        row(
            "invalid_global_smote_contamination", False, len(y_bad_train), float(y_bad_train.mean()), wrong_scores,
            "Invalid comparison arm: global SMOTE used held-out rows to create training synthetics; original test is unchanged.",
        ),
    ])
    correct_row, wrong_row = results.iloc[0], results.iloc[1]
    results["auprc_gap_invalid_minus_correct"] = [
        0.0, float(wrong_row.auprc - correct_row.auprc)
    ]
    results["f1_at_0.5_gap_invalid_minus_correct"] = [
        0.0, float(wrong_row["f1_at_0.5"] - correct_row["f1_at_0.5"])
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(results.to_string(index=False))
    print(f"\nSame untouched test set for both arms: {len(y_test):,} rows; fraud prevalence={y_test.mean():.4%}.")
    print(f"AUPRC gap (invalid - correct): {wrong_row.auprc - correct_row.auprc:+.6f}")
    print(f"F1@0.5 gap (invalid - correct): {wrong_row['f1_at_0.5'] - correct_row['f1_at_0.5']:+.6f}")
    print(f"Saved: {out}")
    print("The invalid arm is a controlled leakage illustration, not a reportable model result or a production-valid estimate.")


if __name__ == "__main__":
    main()
