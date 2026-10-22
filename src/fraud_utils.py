from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

TARGET = "Class"
BUDGETS = (0.001, 0.005, 0.01)
COST_BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def load_dataset(path: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Read the ULB CSV and return numeric predictors without fitting transforms."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Target column {TARGET!r} was not found in {path!r}.")
    y = pd.to_numeric(frame[TARGET], errors="raise").astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The target column must contain both binary labels 0 and 1.")

    # Prefer the documented benchmark fields. A numeric-column fallback supports
    # compatible derivatives while still excluding the target and common row IDs.
    benchmark = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    columns = [c for c in benchmark if c in frame.columns]
    if not columns:
        excluded = {TARGET, "id", "ID", "transaction_id"}
        columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not columns:
        raise ValueError("No numeric feature columns were found.")
    X = frame[columns].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Missing feature values found. Add a documented imputation step before running.")
    return X, y, columns


def build_supervised(model_name: str, strategy: str, y_train: np.ndarray, seed: int,
                     n_jobs: int = 1, smote_k: int = 5) -> Any:
    """Create a fresh scaler/model pipeline; sampling is inside the training pipeline."""
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if strategy == "smote":
        minority = int(np.bincount(y_train).min())
        if minority <= smote_k:
            raise ValueError(f"SMOTE needs more than {smote_k} minority observations in each training fold.")
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=smote_k)))

    weight = strategy == "class_weight"
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            C=1.0, max_iter=1500, solver="lbfgs",
            class_weight="balanced" if weight else None,
            random_state=seed,
        )
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(
            n_estimators=150, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample" if weight else None,
            n_jobs=n_jobs, random_state=seed,
        )
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is required. Install dependencies with `pip install -r requirements.txt`.") from exc
        positives = max(int(np.sum(y_train == 1)), 1)
        negatives = int(np.sum(y_train == 0))
        spw = negatives / positives if weight else 1.0
        estimator = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", scale_pos_weight=spw,
            n_jobs=n_jobs, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown supervised model: {model_name}")
    steps.append(("model", estimator))
    cls = ImbPipeline if strategy == "smote" else Pipeline
    return cls(steps)


def build_unsupervised(model_name: str, seed: int, n_jobs: int = 1) -> Pipeline:
    if model_name == "IsolationForest":
        model = IsolationForest(
            n_estimators=200, max_samples="auto", contamination="auto",
            random_state=seed, n_jobs=n_jobs,
        )
    elif model_name == "LOF":
        model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown unsupervised model: {model_name}")
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def anomaly_scores(fitted_model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a score where larger values mean more likely fraud/anomalous."""
    if hasattr(fitted_model, "predict_proba"):
        scores = fitted_model.predict_proba(X)[:, 1]
    elif hasattr(fitted_model, "score_samples"):
        # sklearn anomaly estimators assign lower scores to more anomalous rows.
        scores = -fitted_model.score_samples(X)
    elif hasattr(fitted_model, "decision_function"):
        scores = fitted_model.decision_function(X)
    else:
        raise TypeError("Model does not expose a continuous ranking score.")
    return np.asarray(scores, dtype=float)


def top_k_indices(scores: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(math.ceil(len(scores) * budget)))
    k = min(k, len(scores))
    # Stable sorting makes ties reproducible while selecting exactly k alerts.
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return order[:k]


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray,
                    budgets: tuple[float, ...] = BUDGETS,
                    missed_fraud_cost: float = 1000.0,
                    false_alert_cost: float = 10.0) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Each evaluation fold must contain both classes.")
    result: dict[str, float] = {
        "auprc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    for budget in budgets:
        chosen = top_k_indices(scores, budget)
        pred = np.zeros(len(y_true), dtype=int)
        pred[chosen] = 1
        tag = f"{100 * budget:g}pct"
        result[f"precision_at_{tag}"] = float(y_true[chosen].mean())
        result[f"recall_at_{tag}"] = float(y_true[chosen].sum() / max(int(y_true.sum()), 1))
        result[f"f1_at_{tag}"] = float(f1_score(y_true, pred, zero_division=0))
        fn = int(y_true.sum() - y_true[chosen].sum())
        fp = int(len(chosen) - y_true[chosen].sum())
        result[f"cost_at_{tag}"] = float(fn * missed_fraud_cost + fp * false_alert_cost)
    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error correction, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def paired_wilcoxon(fold_results: pd.DataFrame, metric: str = "auprc") -> pd.DataFrame:
    """Pair configurations on identical outer split IDs; adjust p-values by Holm."""
    configs = sorted(fold_results["config"].unique())
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(configs, 2):
        a = fold_results.loc[fold_results.config == left, ["split_id", metric]].rename(columns={metric: "a"})
        b = fold_results.loc[fold_results.config == right, ["split_id", metric]].rename(columns={metric: "b"})
        paired = a.merge(b, on="split_id", how="inner").dropna()
        if len(paired) < 2:
            continue
        difference = paired["a"].to_numpy() - paired["b"].to_numpy()
        if np.allclose(difference, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(paired["a"], paired["b"], zero_method="zsplit", alternative="two-sided", method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append({
            "metric": metric, "config_a": left, "config_b": right,
            "n_paired_splits": len(paired),
            "mean_a": float(paired.a.mean()), "mean_b": float(paired.b.mean()),
            "mean_difference_a_minus_b": float(difference.mean()),
            "wilcoxon_statistic": statistic, "p_value": p_value,
        })
    if rows:
        adjusted = holm_adjust([row["p_value"] for row in rows])
        for row, p_adj in zip(rows, adjusted):
            row["p_value_holm"] = p_adj
    return pd.DataFrame(rows)
