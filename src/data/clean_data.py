"""
ToxiPred — Data Cleaning Module

Handles SMILES validation, canonicalization, deduplication,
missing label handling, and data quality reporting.
"""

from typing import Optional

import pandas as pd
from rdkit import Chem

from src.utils.config import COL_SMILES, COL_LABEL, COL_SOURCE, MIN_DATASET_SIZE
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_and_canonicalize_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate all SMILES strings and replace with canonical forms.

    Molecules that cannot be parsed by RDKit are flagged for removal.

    Args:
        df: DataFrame with a 'smiles' column.

    Returns:
        DataFrame with validated, canonical SMILES and a 'valid' column.
    """
    logger.info("Validating and canonicalizing SMILES strings...")

    canonical_smiles = []
    valid_flags = []

    for idx, smi in enumerate(df[COL_SMILES]):
        if pd.isna(smi) or not isinstance(smi, str) or len(smi.strip()) == 0:
            canonical_smiles.append(None)
            valid_flags.append(False)
            continue

        try:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical)
                valid_flags.append(True)
            else:
                canonical_smiles.append(None)
                valid_flags.append(False)
        except Exception:
            canonical_smiles.append(None)
            valid_flags.append(False)

    df = df.copy()
    df[COL_SMILES] = canonical_smiles
    df["_valid"] = valid_flags

    n_invalid = sum(not v for v in valid_flags)
    logger.info(
        f"SMILES validation: {sum(valid_flags)} valid, "
        f"{n_invalid} invalid ({n_invalid / len(df):.1%})"
    )

    return df


def remove_invalid_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with invalid SMILES or missing labels.

    Args:
        df: DataFrame with '_valid' flag from validation step.

    Returns:
        Cleaned DataFrame with invalid rows removed.
    """
    initial_count = len(df)

    # Remove invalid SMILES
    if "_valid" in df.columns:
        df = df[df["_valid"] == True].drop(columns=["_valid"])  # noqa: E712
    else:
        df = df.dropna(subset=[COL_SMILES])

    after_smiles = len(df)
    logger.info(f"Removed {initial_count - after_smiles} invalid SMILES")

    # Remove missing labels
    df = df.dropna(subset=[COL_LABEL])
    after_labels = len(df)
    logger.info(f"Removed {after_smiles - after_labels} missing labels")

    # Ensure label is integer
    df[COL_LABEL] = df[COL_LABEL].astype(int)

    # Validate labels are binary
    valid_labels = df[COL_LABEL].isin([0, 1])
    if not valid_labels.all():
        n_invalid_labels = (~valid_labels).sum()
        logger.warning(f"Removing {n_invalid_labels} non-binary labels")
        df = df[valid_labels]

    logger.info(f"After cleaning: {len(df)} compounds (from {initial_count})")
    return df


def deduplicate_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate molecules based on canonical SMILES.

    For duplicates with conflicting labels, uses majority vote.
    For ties, keeps as toxic (conservative approach for safety screening).

    Args:
        df: DataFrame with canonical SMILES.

    Returns:
        Deduplicated DataFrame.
    """
    initial_count = len(df)

    # Check for duplicates
    duplicated_smiles = df[df.duplicated(subset=[COL_SMILES], keep=False)]
    n_duplicated = len(duplicated_smiles)

    if n_duplicated == 0:
        logger.info("No duplicate molecules found")
        return df

    logger.info(f"Found {n_duplicated} duplicate entries "
                f"({duplicated_smiles[COL_SMILES].nunique()} unique molecules)")

    # Check for label conflicts
    conflicts = duplicated_smiles.groupby(COL_SMILES)[COL_LABEL].nunique()
    n_conflicts = (conflicts > 1).sum()
    if n_conflicts > 0:
        logger.warning(
            f"{n_conflicts} molecules have conflicting labels across sources. "
            "Using majority vote (toxic on tie for conservative screening)."
        )

    # Resolve: group by SMILES, aggregate labels
    def _resolve_group(group):
        """Resolve a group of duplicate molecules."""
        if len(group) == 1:
            return group.iloc[0]

        # Majority vote for label
        label_counts = group[COL_LABEL].value_counts()
        if len(label_counts) > 1 and label_counts.iloc[0] == label_counts.iloc[1]:
            # Tie: default to toxic (conservative)
            resolved_label = 1
        else:
            resolved_label = label_counts.index[0]

        # Join source names
        sources = group[COL_SOURCE].unique() if COL_SOURCE in group.columns else ["unknown"]
        resolved_source = "+".join(sorted(sources))

        result = group.iloc[0].copy()
        result[COL_LABEL] = resolved_label
        if COL_SOURCE in result.index:
            result[COL_SOURCE] = resolved_source
        return result

    df = df.groupby(COL_SMILES, group_keys=False).apply(_resolve_group).reset_index(drop=True)

    logger.info(f"After deduplication: {len(df)} compounds (from {initial_count})")
    return df


def report_class_balance(df: pd.DataFrame) -> dict:
    """
    Report and log the class distribution.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dictionary with class balance statistics.
    """
    counts = df[COL_LABEL].value_counts()
    total = len(df)

    balance = {
        "total": total,
        "toxic": int(counts.get(1, 0)),
        "nontoxic": int(counts.get(0, 0)),
        "toxic_ratio": float(counts.get(1, 0) / total) if total > 0 else 0,
        "imbalance_ratio": (
            float(counts.max() / counts.min()) if len(counts) == 2 and counts.min() > 0
            else float("inf")
        ),
    }

    logger.info("=" * 50)
    logger.info("Class Balance Report:")
    logger.info(f"  Total:     {balance['total']}")
    logger.info(f"  Toxic:     {balance['toxic']} ({balance['toxic_ratio']:.1%})")
    logger.info(f"  Non-toxic: {balance['nontoxic']} ({1 - balance['toxic_ratio']:.1%})")
    logger.info(f"  Imbalance: {balance['imbalance_ratio']:.2f}:1")

    if balance["imbalance_ratio"] > 3:
        logger.warning(
            "Dataset is significantly imbalanced. "
            "Class weighting / resampling will be applied during training."
        )
    logger.info("=" * 50)

    return balance


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: validate → remove invalid → deduplicate → report.

    Args:
        df: Raw merged DataFrame.

    Returns:
        Cleaned, deduplicated DataFrame ready for featurization.

    Raises:
        ValueError: If cleaned dataset is too small.
    """
    logger.info("Starting data cleaning pipeline...")

    df = validate_and_canonicalize_smiles(df)
    df = remove_invalid_molecules(df)
    df = deduplicate_molecules(df)

    balance = report_class_balance(df)

    if len(df) < MIN_DATASET_SIZE:
        raise ValueError(
            f"Cleaned dataset has only {len(df)} compounds, "
            f"below minimum threshold of {MIN_DATASET_SIZE}. "
            "Check data sources and cleaning parameters."
        )

    logger.info(f"Cleaning complete: {len(df)} compounds ready for featurization")
    return df
