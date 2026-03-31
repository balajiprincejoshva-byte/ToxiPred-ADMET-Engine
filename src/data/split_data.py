"""
ToxiPred — Data Splitting Module

Implements stratified train/validation/test splitting with
reproducible random seeds and detailed reporting.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, COL_LABEL, PROCESSED_DATA_DIR,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into stratified train/validation/test sets.

    Uses two-stage splitting:
    1. Split off test set
    2. Split remaining into train and validation

    Stratification ensures each split preserves the class distribution.

    Args:
        df: Cleaned DataFrame with 'label' column.
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.15).
        test_ratio: Fraction for testing (default: 0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to ~1.0 or dataset is too small.
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")

    if len(df) < 30:
        raise ValueError(
            f"Dataset too small for splitting ({len(df)} samples). "
            "Need at least 30 compounds."
        )

    logger.info(
        f"Splitting {len(df)} compounds: "
        f"{train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test"
    )

    # Stage 1: Split off test set
    remaining_ratio = train_ratio + val_ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[COL_LABEL],
        random_state=random_seed,
    )

    # Stage 2: Split remaining into train and validation
    val_relative = val_ratio / remaining_ratio
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df[COL_LABEL],
        random_state=random_seed,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Log split statistics
    _log_split_stats("Train", train_df)
    _log_split_stats("Val", val_df)
    _log_split_stats("Test", test_df)

    return train_df, val_df, test_df


def _log_split_stats(name: str, df: pd.DataFrame) -> None:
    """Log statistics for a data split."""
    n = len(df)
    n_toxic = (df[COL_LABEL] == 1).sum()
    n_nontoxic = (df[COL_LABEL] == 0).sum()
    logger.info(
        f"  {name:5s}: {n:5d} compounds "
        f"(toxic: {n_toxic} [{n_toxic / n:.1%}], "
        f"non-toxic: {n_nontoxic} [{n_nontoxic / n:.1%}])"
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Save train/val/test splits to CSV files.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.

    Returns:
        Dictionary with split summary statistics.
    """
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Splits saved to {PROCESSED_DATA_DIR}")

    summary = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_toxic_ratio": float((train_df[COL_LABEL] == 1).mean()),
        "val_toxic_ratio": float((val_df[COL_LABEL] == 1).mean()),
        "test_toxic_ratio": float((test_df[COL_LABEL] == 1).mean()),
    }

    return summary


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved splits from disk.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    logger.info(
        f"Loaded splits: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df
