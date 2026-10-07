"""
train.py
========
Trains, benchmarks, and serializes machine learning models for fraud detection:
- Baseline: Logistic Regression (Balanced)
- Tree Ensemble 1: LightGBM Classifier
- Tree Ensemble 2: XGBoost Classifier (Champion)

Tracks financial evaluation metrics (PR-AUC, ROC-AUC, F1, Recall) and saves
the top-performing champion model for production inference.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.data_loader import load_raw_data, generate_fraud_labels, get_project_root
from src.feature_engineering import prepare_train_test_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_benchmark_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Trains Logistic Regression, LightGBM, and XGBoost models.
    Computes comparative performance on the test holdout set.
    """
    # Calculate positive weight ratio for cost-sensitive boosting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Probability predictions for positive (fraud) class
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        }
        logger.info(
            f"--> {name} Results: PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | "
            f"Recall={recall:.4f} | Precision={precision:.4f} | F1={f1:.4f}"
        )

    # Determine champion model based on PR-AUC (key AmEx fraud ranking metric)
    champion_name = max(results.keys(), key=lambda k: results[k]["pr_auc"])
    logger.info(f"Selected Champion Model: '{champion_name}' with PR-AUC={results[champion_name]['pr_auc']}")

    return results, fitted_models, champion_name


def save_artifacts(
    champion_model: Any,
    champion_name: str,
    feature_engineer: Any,
    feature_cols: list,
    benchmark_metrics: Dict[str, Any],
    models_dir: str = None
) -> str:
    """Saves serialized champion model and metrics to disk."""
    if models_dir is None:
        models_dir = os.path.join(get_project_root(), "models")
    os.makedirs(models_dir, exist_ok=True)

    artifact_bundle = {
        "model_name": champion_name,
        "model": champion_model,
        "feature_engineer": feature_engineer,
        "feature_names": feature_cols
    }

    model_path = os.path.join(models_dir, "champion_fraud_model.joblib")
    joblib.dump(artifact_bundle, model_path)
    logger.info(f"Champion artifact saved to: {model_path}")

    metrics_path = os.path.join(models_dir, "model_benchmarks.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_metrics, f, indent=4)
    logger.info(f"Benchmark metrics saved to: {metrics_path}")

    return model_path


def run_training_pipeline() -> Tuple[Dict[str, Any], str]:
    """Runs complete end-to-end training pipeline."""
    df_raw = load_raw_data()
    df_labeled = generate_fraud_labels(df_raw)
    X_train, X_test, y_train, y_test, engineer = prepare_train_test_pipeline(df_labeled)

    metrics, fitted_models, champion_name = train_and_benchmark_models(
        X_train, X_test, y_train, y_test
    )

    champion_model = fitted_models[champion_name]
    model_path = save_artifacts(
        champion_model=champion_model,
        champion_name=champion_name,
        feature_engineer=engineer,
        feature_cols=X_train.columns.tolist(),
        benchmark_metrics=metrics
    )

    return metrics, model_path


if __name__ == "__main__":
    run_training_pipeline()
