"""
evaluate.py
===========
Evaluates the champion fraud model with banking-grade financial metrics:
- Precision-Recall and ROC curves
- Confusion Matrix visualization
- Cost-Benefit Optimization: Simulates financial dollar loss from False Negatives
  (fraud exposure) vs. False Positives (customer friction / manual review cost)
  to determine the profit-maximizing threshold cutoff.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    auc,
    average_precision_score,
    roc_auc_score,
    classification_report
)

from src.data_loader import load_raw_data, generate_fraud_labels, get_project_root
from src.feature_engineering import prepare_train_test_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_pr_and_roc_curves(
    y_test: pd.Series,
    y_prob: np.ndarray,
    output_path: str
) -> None:
    """Plots and saves dual PR and ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    axes[0].plot(recall, precision, color="#006FCF", lw=2.5, label=f"Champion PR Curve (AUC = {pr_auc:.4f})")
    axes[0].axhline(y=y_test.mean(), color="gray", linestyle="--", label=f"No-skill baseline ({y_test.mean():.3f})")
    axes[0].set_title("Precision-Recall Curve (Fraud Detection)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Recall (Fraud Coverage)", fontsize=11)
    axes[0].set_ylabel("Precision (True Fraud Ratio)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    axes[1].plot(fpr, tpr, color="#00175A", lw=2.5, label=f"Champion ROC Curve (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (Friction)", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR & ROC curves plot to: {output_path}")


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str,
    threshold: float
) -> None:
    """Plots labeled confusion matrix heatmap with banking transaction terminology."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix (Decision Threshold = {threshold:.2f})", fontsize=13, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def evaluate_financial_cost_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    cost_false_negative: float = 250.0,
    cost_false_positive: float = 15.0,
    output_chart_path: str = None
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Simulates financial dollar loss across decision thresholds:
    Cost = (False Negatives * Cost_FN) + (False Positives * Cost_FP)
    
    Cost_FN represents average unrecovered dollar loss of a fraudulent charge.
    Cost_FP represents customer dispute/review friction and notification cost.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    fns = []
    fps = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
        costs.append(total_cost)
        fns.append(fn)
        fps.append(fp)

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    optimal_threshold = float(thresholds[opt_idx])
    min_cost = float(costs[opt_idx])

    # Default 0.5 threshold comparison
    default_preds = (y_prob >= 0.5).astype(int)
    tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_test, default_preds).ravel()
    default_cost = float((fn_def * cost_false_negative) + (fp_def * cost_false_positive))
    estimated_savings = default_cost - min_cost

    summary = {
        "cost_params": {
            "cost_per_false_negative_usd": cost_false_negative,
            "cost_per_false_positive_usd": cost_false_positive
        },
        "default_threshold_0.5": {
            "total_cost_usd": default_cost,
            "false_negatives": int(fn_def),
            "false_positives": int(fp_def)
        },
        "optimal_threshold": {
            "threshold": round(optimal_threshold, 4),
            "total_cost_usd": min_cost,
            "false_negatives": int(fns[opt_idx]),
            "false_positives": int(fps[opt_idx])
        },
        "business_impact": {
            "estimated_savings_usd": round(estimated_savings, 2),
            "cost_reduction_percent": round((estimated_savings / max(1.0, default_cost)) * 100, 2)
        }
    }

    if output_chart_path:
        plt.figure(figsize=(9, 5))
        plt.plot(thresholds, costs, color="#E0292B", lw=2.5, label="Total Financial Cost ($)")
        plt.axvline(
            optimal_threshold,
            color="#006FCF",
            linestyle="--",
            lw=2,
            label=f"Optimal Cutoff (t* = {optimal_threshold:.2f}, Loss = ${min_cost:,.0f})"
        )
        plt.axvline(
            0.5,
            color="gray",
            linestyle=":",
            lw=1.5,
            label=f"Default Threshold (t = 0.50, Loss = ${default_cost:,.0f})"
        )
        plt.title("Financial Cost Curve vs. Decision Threshold (AmEx Portfolio Optimization)", fontsize=13, fontweight="bold")
        plt.xlabel("Probability Cutoff Threshold", fontsize=11)
        plt.ylabel("Expected Financial Cost ($ USD)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig(output_chart_path, dpi=300)
        plt.close()
        logger.info(f"Saved financial cost curve to: {output_chart_path}")

    return optimal_threshold, min_cost, default_cost, summary


def run_full_evaluation() -> Dict[str, Any]:
    """Loads champion artifact, test set, and generates full banking evaluation suite."""
    root = get_project_root()
    model_path = os.path.join(root, "models", "champion_fraud_model.joblib")
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at {model_path}. Run train.py first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_engineer = bundle["feature_engineer"]

    df_raw = load_raw_data()
    df_labeled = generate_fraud_labels(df_raw)
    X_train, X_test, y_train, y_test, _ = prepare_train_test_pipeline(df_labeled)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Dual PR / ROC curves
    pr_roc_path = os.path.join(reports_dir, "pr_roc_curves.png")
    plot_pr_and_roc_curves(y_test, y_prob, pr_roc_path)

    # 2. Financial cost optimization
    cost_chart_path = os.path.join(reports_dir, "business_cost_curve.png")
    opt_t, opt_cost, def_cost, cost_summary = evaluate_financial_cost_curve(
        y_test, y_prob, output_chart_path=cost_chart_path
    )

    # 3. Confusion Matrix at optimal threshold
    opt_preds = (y_prob >= opt_t).astype(int)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, opt_preds, cm_path, threshold=opt_t)

    # 4. Save metrics JSON
    summary_path = os.path.join(reports_dir, "financial_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cost_summary, f, indent=4)
    logger.info(f"Financial evaluation report saved to: {summary_path}")

    logger.info(
        f"Evaluation Summary:\n"
        f"  Default (0.5) Loss: ${def_cost:,.2f}\n"
        f"  Optimal ({opt_t:.2f}) Loss: ${opt_cost:,.2f}\n"
        f"  Savings: ${cost_summary['business_impact']['estimated_savings_usd']:,.2f} "
        f"({cost_summary['business_impact']['cost_reduction_percent']}%)"
    )

    return cost_summary


if __name__ == "__main__":
    run_full_evaluation()
