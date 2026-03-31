"""
ToxiPred — File I/O Utilities

Helper functions for saving and loading models, metrics, DataFrames,
and other artifacts in a consistent manner.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: Path) -> None:
    """Save a trained model to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> Any:
    """Load a trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {k: _convert(v) for k, v in data.items()}

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"JSON saved to {path}")


def load_json(path: Path) -> dict:
    """Load a JSON file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path) as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info(f"DataFrame saved to {path} ({len(df)} rows)")


def load_dataframe(path: Path, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"DataFrame loaded from {path} ({len(df)} rows)")
    return df


def save_numpy(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info(f"Array saved to {path} (shape: {arr.shape})")


def load_numpy(path: Path) -> np.ndarray:
    """Load a numpy array from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Array file not found: {path}")
    arr = np.load(path)
    logger.info(f"Array loaded from {path} (shape: {arr.shape})")
    return arr
