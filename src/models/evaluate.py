"""
ToxiPred — Model Evaluation Module

Comprehensive evaluation suite for toxicity prediction models.
Computes metrics, generates publication-quality plots, and
optimizes classification thresholds.

Metrics:
- ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- Confusion Matrix
- Calibration analysis

Plots:
- ROC curve
- Precision-Recall curve
- Confusion matrix heatmap
- Calibration curve
- Model comparison bar chart
- Feature importance
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from src.utils.config import (
    PLOTS_DIR, METRICS_DIR, DEFAULT_THRESHOLD,
    PLOT_DPI, PLOT_FIGSIZE, PLOT_STYLE,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT,
    THRESHOLD_FILENAME, MODELS_DIR,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_json

logger = get_logger(__name__)

# Set plotting style
try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dictionary of metric_name → value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate Brier Score
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the optimal classification threshold by maximizing a metric.

    For toxicity screening, we may want to prioritize recall
    (catching toxic compounds) over precision.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ("f1", "balanced_accuracy", "youden").

    Returns:
        Optimal threshold value.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = -1
    best_threshold = DEFAULT_THRESHOLD

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J statistic: sensitivity + specificity - 1
            tn = np.sum((y_pred == 0) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(
        f"Optimal threshold ({metric}): {best_threshold:.2f} "
        f"(score: {best_score:.4f})"
    )

    return float(best_threshold)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2.5, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_ACCENT, lw=1.5, linestyle="--", label="Random", alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLOR_PRIMARY)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "roc_curve.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Baseline: prevalence of positive class
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(recall, precision, color=COLOR_SECONDARY, lw=2.5, label=f"XGBoost (AP = {ap:.3f})")
    ax.axhline(y=baseline, color=COLOR_ACCENT, lw=1.5, linestyle="--", label=f"Baseline ({baseline:.2f})", alpha=0.7)
    ax.fill_between(recall, precision, alpha=0.15, color=COLOR_SECONDARY)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "precision_recall_curve.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save confusion matrix heatmap."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Toxic", "Toxic"],
        yticklabels=["Non-Toxic", "Toxic"],
        ax=ax, annot_kws={"size": 16},
        linewidths=0.5, linecolor="white",
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix (threshold = {threshold:.2f})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "confusion_matrix.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save calibration curve plot."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # Uncalibrated
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot(prob_pred, prob_true, "s-", color=COLOR_PRIMARY, lw=2, label="Uncalibrated", markersize=8)

    # Calibrated (if available)
    if y_prob_calibrated is not None:
        prob_true_c, prob_pred_c = calibration_curve(y_true, y_prob_calibrated, n_bins=n_bins)
        ax.plot(prob_pred_c, prob_true_c, "o-", color=COLOR_SECONDARY, lw=2, label="Calibrated", markersize=8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color=COLOR_ACCENT, lw=1.5, label="Perfectly Calibrated", alpha=0.7)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "calibration_curve.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Calibration curve saved to {save_path}")


def plot_model_comparison(
    cv_comparison: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save model comparison bar chart."""
    metrics_to_plot = ["cv_roc_auc_mean", "cv_f1_mean", "cv_precision_mean", "cv_recall_mean"]
    available = [m for m in metrics_to_plot if m in cv_comparison.columns]

    if not available:
        logger.warning("No comparison metrics available for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for grouped bar chart
    models = cv_comparison["model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, "#FFB347", COLOR_ACCENT]

    for i, metric in enumerate(available):
        label = metric.replace("cv_", "").replace("_mean", "").replace("_", " ").title()
        values = cv_comparison[metric].tolist()
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i % len(colors)], alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison (Cross-Validation)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "model_comparison.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison chart saved to {save_path}")


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save a reliability curve (calibration plot)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot([0, 1], [0, 1], color=COLOR_ACCENT, linestyle="--", label="Perfectly calibrated", alpha=0.7)
    ax.plot(prob_pred, prob_true, "s-", color=COLOR_PRIMARY, label="Model")
    
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_title("Reliability Curve (Calibration Plot)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "reliability_curve.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reliability curve saved to {save_path}")


def run_evaluation(
    model: Any,
    calibrated_model: Optional[Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_comparison: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline on the test set.

    Args:
        model: Trained (uncalibrated) model.
        calibrated_model: Calibrated model (optional).
        X_test: Test features.
        y_test: Test labels.
        cv_comparison: Cross-validation comparison DataFrame.
        test_df: Test DataFrame with SMILES (for logging).

    Returns:
        Dictionary with all metrics and artifacts.
    """
    logger.info("=" * 60)
    logger.info("RUNNING EVALUATION")
    logger.info("=" * 60)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    y_prob_cal = None
    if calibrated_model is not None:
        try:
            y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Calibrated model prediction failed: {e}")

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_prob, metric="f1")

    # Save optimal threshold
    save_json(
        {"optimal_threshold": optimal_threshold, "method": "f1_maximization"},
        MODELS_DIR / THRESHOLD_FILENAME,
    )

    # Compute metrics at default and optimal thresholds
    metrics_default = compute_all_metrics(y_test, y_prob, threshold=DEFAULT_THRESHOLD)
    metrics_optimal = compute_all_metrics(y_test, y_prob, threshold=optimal_threshold)

    # Log test metrics
    logger.info("\nTest Set Metrics (default threshold = 0.5):")
    for k, v in metrics_default.items():
        if isinstance(v, float):
            logger.info(f"  {k:25s}: {v:.4f}")

    logger.info(f"\nTest Set Metrics (optimal threshold = {optimal_threshold:.2f}):")
    for k, v in metrics_optimal.items():
        if isinstance(v, float):
            logger.info(f"  {k:25s}: {v:.4f}")

    # Save metrics
    save_json(metrics_default, METRICS_DIR / "test_metrics_default.json")
    save_json(metrics_optimal, METRICS_DIR / "test_metrics_optimal.json")

    # Classification report
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    report = classification_report(y_test, y_pred_opt, target_names=["Non-Toxic", "Toxic"])
    logger.info(f"\nClassification Report:\n{report}")

    report_path = METRICS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification Report (threshold = {optimal_threshold:.2f})\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    # Generate plots
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_prob, threshold=optimal_threshold)
    plot_calibration_curve(y_test, y_prob, y_prob_calibrated=y_prob_cal)
    plot_reliability_curve(y_test, y_prob)
    
    if cv_comparison is not None:
        plot_model_comparison(cv_comparison)
    elif (METRICS_DIR / "model_comparison.csv").exists():
        comp_df = pd.read_csv(METRICS_DIR / "model_comparison.csv")
        plot_model_comparison(comp_df)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return {
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "optimal_threshold": optimal_threshold,
        "y_prob": y_prob,
        "y_prob_calibrated": y_prob_cal,
        "classification_report": report,
    }
