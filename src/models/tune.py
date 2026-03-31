"""
ToxiPred — Hyperparameter Tuning Module

Implements RandomizedSearchCV for XGBoost hyperparameter optimization.
Optimizes for ROC-AUC by default with stratified cross-validation.
"""

from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

from src.utils.config import (
    RANDOM_SEED, CV_FOLDS, XGBOOST_PARAMS,
    XGBOOST_SEARCH_SPACE, TUNING_N_ITER,
    MODELS_DIR, METRICS_DIR,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_json, save_model

logger = get_logger(__name__)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float = 1.0,
    n_iter: int = TUNING_N_ITER,
    cv_folds: int = CV_FOLDS,
    scoring: str = "roc_auc",
) -> Dict[str, Any]:
    """
    Perform randomized hyperparameter search for XGBoost.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        scale_pos_weight: Class balance weight.
        n_iter: Number of random parameter combinations to try.
        cv_folds: Number of CV folds.
        scoring: Metric to optimize.

    Returns:
        Dictionary with best parameters, best score, and tuned model.
    """
    logger.info(f"Starting XGBoost hyperparameter tuning ({n_iter} iterations)...")

    # Base model with fixed parameters
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    base_model = xgb.XGBClassifier(**base_params)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=XGBOOST_SEARCH_SPACE,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_
    best_model = search.best_estimator_

    logger.info(f"Best {scoring}: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Save results
    tuning_results = {
        "best_score": float(best_score),
        "best_params": best_params,
        "scoring_metric": scoring,
        "n_iter": n_iter,
        "cv_folds": cv_folds,
    }
    save_json(tuning_results, METRICS_DIR / "tuning_results.json")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "search": search,
    }


def tune_and_retrain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float = 1.0,
    n_iter: int = TUNING_N_ITER,
) -> xgb.XGBClassifier:
    """
    Tune hyperparameters, then retrain the best model with early stopping.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        scale_pos_weight: Class balance weight.
        n_iter: Number of search iterations.

    Returns:
        Tuned and retrained XGBClassifier.
    """
    # Tune
    results = tune_xgboost(X_train, y_train, scale_pos_weight, n_iter)
    best_params = results["best_params"]

    # Retrain with best params + early stopping
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        **best_params,
    }

    logger.info("Retraining with best parameters and early stopping...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    save_model(model, MODELS_DIR / "xgboost_tuned.joblib")
    logger.info("Tuned model saved")

    return model
