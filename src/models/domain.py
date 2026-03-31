"""
ToxiPred — Applicability Domain Module

Detects if a test molecule is within the structural region of the 
training set (In-Domain) or is a structural outlier (Out-of-Domain).

Uses IsolationForest on descriptor + fingerprint features for 
robust anomaly detection in high-dimensional chemical space.
"""

from typing import Any, List, Optional, Tuple, Dict
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ApplicabilityDomain:
    """
    Handles training-set domain boundary detection.
    """
    def __init__(self, contamination: float = 0.05, random_seed: int = 42):
        """
        Args:
            contamination: Expected fraction of outliers in the training set.
            random_seed: Random seed for reproducibility.
        """
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_seed,
            n_jobs=-1
        )
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the outlier detector on training features.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            feature_names: List of feature names.
        """
        logger.info(f"Fitting Applicability Domain model on {X.shape[0]} samples...")
        self.model.fit(X)
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        self._is_fitted = True
        logger.info("Applicability Domain fitting complete.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict domain status and anomaly scores.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Tuple of (status_array, score_array).
            status: 1 for in-domain, -1 for outlier/OOD.
            score: Raw anomaly score (lower is more anomalous).
        """
        if not self._is_fitted:
            raise ValueError("ApplicabilityDomain model is not fitted.")
        
        status = self.model.predict(X)
        scores = self.model.decision_function(X)
        return status, scores

    def check_single(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Check a single instance.
        
        Args:
            x: Single feature vector (1, n_features).
            
        Returns:
            Dictionary with status and score.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        status, score = self.predict(x)
        is_ood = bool(status[0] == -1)
        
        return {
            "is_ood": is_ood,
            "status": "Out-of-Domain" if is_ood else "In-Domain",
            "score": float(score[0]),
            "warning": "Structure is distant from the training distribution. Prediction confidence may be lower." if is_ood else None
        }

    def save(self, path: Path):
        """Save the fitted model."""
        joblib.dump(self, path)
        logger.info(f"Applicability Domain model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ApplicabilityDomain':
        """Load a fitted model."""
        return joblib.load(path)
