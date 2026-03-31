"""
ToxiPred — Model Training Pipeline

Trains and compares multiple classifiers for hepatotoxicity prediction:
- XGBoost (primary model)
- Logistic Regression (baseline)
- Random Forest (baseline)

Includes:
- 5-fold stratified cross-validation
- Class imbalance handling
- Model calibration
- Performance comparison
- Artifact saving
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.models.domain import ApplicabilityDomain
from src.utils.config import (
    RANDOM_SEED, CV_FOLDS, PRIMARY_METRIC,
    XGBOOST_PARAMS, LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS,
    MODELS_DIR, METRICS_DIR, FEATURE_MODE,
    MODEL_FILENAME, CALIBRATED_MODEL_FILENAME,
    SCALER_FILENAME, FEATURE_NAMES_FILENAME,
    AD_MODEL_FILENAME, THRESHOLD_FILENAME,
    COL_LABEL,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_model, save_json
from src.features.featurize import featurize_dataset, get_all_feature_names
from src.data.split_data import stratified_split, save_splits

logger = get_logger(__name__)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight for XGBoost based on class distribution.

    scale_pos_weight = count(negative) / count(positive)

    Args:
        y: Binary label array.

    Returns:
        Float ratio for class balancing.
    """
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        logger.warning("No positive samples found!")
        return 1.0
    ratio = n_neg / n_pos
    logger.info(f"Class balance: neg={n_neg}, pos={n_pos}, scale_pos_weight={ratio:.2f}")
    return ratio


def build_models(scale_pos_weight: float) -> Dict[str, Any]:
    """
    Instantiate all models with appropriate class balancing.

    Args:
        scale_pos_weight: Ratio for XGBoost class balancing.

    Returns:
        Dictionary of model_name → model_instance.
    """
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params["scale_pos_weight"] = scale_pos_weight

    models = {
        "XGBoost": xgb.XGBClassifier(**xgb_params),
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "RandomForest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
    }

    return models


def cross_validate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Run stratified cross-validation for all models and compare.

    Args:
        models: Dictionary of model_name → model_instance.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of CV folds.

    Returns:
        DataFrame with cross-validation results for each model.
    """
    logger.info(f"Running {cv_folds}-fold stratified cross-validation...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = []

    for name, model in models.items():
        logger.info(f"  Cross-validating {name}...")
        try:
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            row = {
                "model": name,
                "cv_roc_auc_mean": np.mean(cv_results["test_roc_auc"]),
                "cv_roc_auc_std": np.std(cv_results["test_roc_auc"]),
                "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
                "cv_accuracy_std": np.std(cv_results["test_accuracy"]),
                "cv_f1_mean": np.mean(cv_results["test_f1"]),
                "cv_f1_std": np.std(cv_results["test_f1"]),
                "cv_precision_mean": np.mean(cv_results["test_precision"]),
                "cv_precision_std": np.std(cv_results["test_precision"]),
                "cv_recall_mean": np.mean(cv_results["test_recall"]),
                "cv_recall_std": np.std(cv_results["test_recall"]),
                "train_roc_auc_mean": np.mean(cv_results["train_roc_auc"]),
            }
            results.append(row)

            logger.info(
                f"    ROC-AUC: {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}"
            )
            logger.info(
                f"    F1:      {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}"
            )

        except Exception as e:
            logger.error(f"  Cross-validation failed for {name}: {e}")
            results.append({"model": name, "cv_roc_auc_mean": 0.0, "error": str(e)})

    comparison_df = pd.DataFrame(results)
    return comparison_df


def train_final_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "XGBoost",
) -> Any:
    """
    Train the final model on the full training set.

    For XGBoost, uses early stopping on the validation set.

    Args:
        model: Model instance to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        model_name: Name for logging.

    Returns:
        Trained model instance.
    """
    logger.info(f"Training final {model_name} model...")

    if isinstance(model, xgb.XGBClassifier):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        try:
            logger.info(f"  Best iteration: {model.best_iteration}")
        except AttributeError:
            logger.info(f"  Training completed (no early stopping)")
    else:
        model.fit(X_train, y_train)

    logger.info(f"Final {model_name} model trained successfully")
    return model


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Calibrate the model's probability outputs using isotonic regression.

    Calibrated probabilities better reflect true risk levels.

    Args:
        model: Trained model to calibrate.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Calibrated model wrapper.
    """
    logger.info("Calibrating model probabilities...")

    calibrated = CalibratedClassifierCV(
        model,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    logger.info("Model calibration complete")
    return calibrated


def run_training_pipeline(
    df: pd.DataFrame,
    feature_mode: str = FEATURE_MODE,
) -> Dict[str, Any]:
    """
    Execute the full training pipeline end-to-end.

    Steps:
    1. Split data (train/val/test)
    2. Featurize all splits
    3. Cross-validate all models
    4. Train final XGBoost model
    5. Calibrate the model
    6. Save all artifacts

    Args:
        df: Cleaned DataFrame with smiles and label columns.
        feature_mode: Feature generation mode.

    Returns:
        Dictionary with training results, model, and paths.
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Split Data ---
    train_df, val_df, test_df = stratified_split(df)
    split_summary = save_splits(train_df, val_df, test_df)

    # --- Step 2: Featurize ---
    logger.info("Featurizing training set...")
    X_train, feature_names, train_df = featurize_dataset(train_df, mode=feature_mode)
    y_train = train_df[COL_LABEL].values

    logger.info("Featurizing validation set...")
    X_val, _, val_df = featurize_dataset(val_df, mode=feature_mode)
    y_val = val_df[COL_LABEL].values

    logger.info("Featurizing test set...")
    X_test, _, test_df = featurize_dataset(test_df, mode=feature_mode)
    y_test = test_df[COL_LABEL].values

    # Scale descriptors if using combined mode
    scaler = None
    if feature_mode in ("descriptors", "combined"):
        scaler = StandardScaler()
        n_fp = 2048 if feature_mode == "combined" else 0

        if feature_mode == "combined":
            # Only scale the descriptor part, not fingerprints
            X_train_fp = X_train[:, :n_fp]
            X_train_desc = X_train[:, n_fp:]
            X_train_desc = scaler.fit_transform(X_train_desc)
            X_train = np.hstack([X_train_fp, X_train_desc])

            X_val_fp = X_val[:, :n_fp]
            X_val_desc = X_val[:, n_fp:]
            X_val_desc = scaler.transform(X_val_desc)
            X_val = np.hstack([X_val_fp, X_val_desc])

            X_test_fp = X_test[:, :n_fp]
            X_test_desc = X_test[:, n_fp:]
            X_test_desc = scaler.transform(X_test_desc)
            X_test = np.hstack([X_test_fp, X_test_desc])
        else:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        save_model(scaler, MODELS_DIR / SCALER_FILENAME)

    # --- Step 3: Cross-Validate ---
    spw = compute_scale_pos_weight(y_train)
    models = build_models(spw)
    cv_comparison = cross_validate_models(models, X_train, y_train)

    # Save comparison
    cv_comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    logger.info("\nModel Comparison:")
    logger.info(f"\n{cv_comparison.to_string(index=False)}")

    # --- Step 4: Train Final Model ---
    # Rebuild XGBoost with best cross-validation insights
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params["scale_pos_weight"] = spw
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model = train_final_model(final_model, X_train, y_train, X_val, y_val)

    # --- Step 5: Calibrate ---
    calibrated_model = calibrate_model(final_model, X_val, y_val)

    # --- Step 6: Applicability Domain ---
    ad_model = ApplicabilityDomain(random_seed=RANDOM_SEED)
    ad_model.fit(X_train, feature_names=feature_names)
    save_model(ad_model, MODELS_DIR / AD_MODEL_FILENAME)

    # --- Step 7: Save Artifacts ---
    save_model(final_model, MODELS_DIR / MODEL_FILENAME)
    save_model(calibrated_model, MODELS_DIR / CALIBRATED_MODEL_FILENAME)
    
    # Save feature names
    save_json(
        {"feature_names": feature_names, "feature_mode": feature_mode},
        MODELS_DIR / FEATURE_NAMES_FILENAME,
    )

    # Save split summary
    save_json(split_summary, METRICS_DIR / "split_summary.json")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "model": final_model,
        "calibrated_model": calibrated_model,
        "ad_model": ad_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_mode": feature_mode,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
        "cv_comparison": cv_comparison,
        "split_summary": split_summary,
    }
