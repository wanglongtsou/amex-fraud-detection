"""
explain.py
==========
Provides model explainability using SHAP (SHapley Additive exPlanations)
for financial transaction risk assessment:
- Global feature impact (SHAP Beeswarm & Bar importance)
- Local case-study explanation (SHAP Waterfall for high-risk flagged transactions)
Crucial for regulatory fairness, auditability, and fraud investigator workflows.
"""

import os
import json
import logging
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data_loader import load_raw_data, generate_fraud_labels, get_project_root
from src.feature_engineering import prepare_train_test_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_shap_explanations() -> Dict[str, Any]:
    """Generates global and local SHAP explanation plots and audits."""
    root = get_project_root()
    model_path = os.path.join(root, "models", "champion_fraud_model.joblib")
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at {model_path}. Run train.py first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    df_raw = load_raw_data()
    df_labeled = generate_fraud_labels(df_raw)
    _, X_test, _, y_test, _ = prepare_train_test_pipeline(df_labeled)

    logger.info("Initializing TreeExplainer for champion model...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 1. Global Beeswarm Summary Plot
    summary_path = os.path.join(reports_dir, "shap_summary.png")
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Global Feature Impact (Fraud Risk Direction)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP summary beeswarm plot to: {summary_path}")

    # 2. Global Feature Importance Bar Plot
    bar_path = os.path.join(reports_dir, "shap_feature_importance.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("SHAP Mean |Value| Feature Importance", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP bar importance plot to: {bar_path}")

    # 3. Local Decision Waterfall for a high-risk flagged transaction
    y_prob = model.predict_proba(X_test)[:, 1]
    high_risk_indices = np.where(y_prob > 0.8)[0]
    
    if len(high_risk_indices) > 0:
        sample_idx = int(high_risk_indices[0])
    else:
        sample_idx = int(np.argmax(y_prob))

    sample_shap = shap_values[sample_idx]
    sample_risk = float(y_prob[sample_idx])
    actual_label = int(y_test.iloc[sample_idx])

    waterfall_path = os.path.join(reports_dir, "shap_waterfall.png")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sample_shap, max_display=10, show=False)
    plt.title(
        f"Case Study: Flagged Transaction Explanation (Risk Score: {sample_risk:.2%}, Actual: {actual_label})",
        fontsize=12,
        fontweight="bold",
        pad=15
    )
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP waterfall explanation plot to: {waterfall_path}")

    # Case study audit export
    case_study = {
        "sample_index": sample_idx,
        "predicted_fraud_probability": round(sample_risk, 4),
        "actual_ground_truth": actual_label,
        "top_features": {
            feat: round(float(val), 4)
            for feat, val in zip(X_test.iloc[sample_idx].index, X_test.iloc[sample_idx].values)
            if abs(val) > 0
        }
    }
    case_path = os.path.join(reports_dir, "case_study_transaction.json")
    with open(case_path, "w") as f:
        json.dump(case_study, f, indent=4)
    logger.info(f"Saved investigator case study audit to: {case_path}")

    return case_study


if __name__ == "__main__":
    generate_shap_explanations()
