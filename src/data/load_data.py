"""
ToxiPred — Data Loading Module

Downloads and loads toxicity datasets from public sources:
- DILI (Drug-Induced Liver Injury) from Therapeutics Data Commons
- ClinTox from MoleculeNet / DeepChem

Standardizes column names and merges datasets with documented logic.
"""

import io
import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    COL_SMILES, COL_LABEL, COL_SOURCE, RANDOM_SEED,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def download_dili_dataset(save_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the DILI (Drug-Induced Liver Injury) dataset.

    Source: Therapeutics Data Commons (TDC)
    - 475 drugs from FDA National Center for Toxicological Research
    - Binary classification: 1 = can cause liver injury, 0 = no

    If PyTDC is available, uses it directly. Otherwise falls back to
    a pre-downloaded CSV if available.

    Args:
        save_dir: Directory to save the raw CSV. Defaults to RAW_DATA_DIR.

    Returns:
        DataFrame with standardized columns: smiles, label, source.
    """
    save_dir = save_dir or RAW_DATA_DIR
    save_path = save_dir / "dili_raw.csv"

    # Try loading from cached file first
    if save_path.exists():
        logger.info(f"Loading cached DILI dataset from {save_path}")
        df = pd.read_csv(save_path)
        if COL_SMILES in df.columns and COL_LABEL in df.columns:
            logger.info(f"DILI dataset: {len(df)} compounds")
            return df

    # Try downloading via PyTDC
    try:
        from tdc.single_pred import Tox
        logger.info("Downloading DILI dataset via PyTDC...")
        data = Tox(name="DILI")
        raw_df = data.get_data()

        # Standardize columns
        df = pd.DataFrame({
            COL_SMILES: raw_df["Drug"].values,
            COL_LABEL: raw_df["Y"].astype(int).values,
            COL_SOURCE: "DILI_TDC",
        })

        df.to_csv(save_path, index=False)
        logger.info(f"DILI dataset saved: {len(df)} compounds")
        return df

    except ImportError:
        logger.warning("PyTDC not installed. Attempting alternative loading...")
    except Exception as e:
        logger.warning(f"PyTDC download failed: {e}. Attempting alternative...")

    # Fallback: generate a minimal synthetic dataset for development
    # This should only trigger if neither PyTDC nor cached data is available
    logger.warning(
        "Could not load DILI dataset from TDC. "
        "Install PyTDC (pip install PyTDC) for the full dataset. "
        "Proceeding with ClinTox only if available."
    )
    return pd.DataFrame(columns=[COL_SMILES, COL_LABEL, COL_SOURCE])


def download_clintox_dataset(save_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the ClinTox dataset from MoleculeNet.

    Source: DeepChem / MoleculeNet
    - ~1,491 compounds
    - Binary classification: CT_TOX = clinical trial toxicity
    - Also includes FDA_APPROVED status (not used as primary label)

    Uses direct HTTP download of the CSV from DeepChem's S3 bucket.

    Args:
        save_dir: Directory to save the raw CSV. Defaults to RAW_DATA_DIR.

    Returns:
        DataFrame with standardized columns: smiles, label, source.
    """
    save_dir = save_dir or RAW_DATA_DIR
    save_path = save_dir / "clintox_raw.csv"

    # Try loading from cached file first
    if save_path.exists():
        logger.info(f"Loading cached ClinTox dataset from {save_path}")
        df = pd.read_csv(save_path)
        if COL_SMILES in df.columns and COL_LABEL in df.columns:
            logger.info(f"ClinTox dataset: {len(df)} compounds")
            return df

    # Download from DeepChem S3
    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz"

    try:
        import urllib.request
        logger.info(f"Downloading ClinTox dataset from {url}...")

        response = urllib.request.urlopen(url, timeout=60)
        compressed = response.read()
        decompressed = gzip.decompress(compressed)
        raw_df = pd.read_csv(io.BytesIO(decompressed))

        logger.info(f"ClinTox raw columns: {list(raw_df.columns)}")
        logger.info(f"ClinTox raw shape: {raw_df.shape}")

        # Use CT_TOX as the toxicity label
        # CT_TOX = 1 means the drug failed clinical trials due to toxicity
        smiles_col = "smiles" if "smiles" in raw_df.columns else raw_df.columns[0]
        tox_col = "CT_TOX" if "CT_TOX" in raw_df.columns else raw_df.columns[1]

        df = pd.DataFrame({
            COL_SMILES: raw_df[smiles_col].values,
            COL_LABEL: raw_df[tox_col].astype(int).values,
            COL_SOURCE: "ClinTox",
        })

        df.to_csv(save_path, index=False)
        logger.info(f"ClinTox dataset saved: {len(df)} compounds")
        return df

    except Exception as e:
        logger.error(f"Failed to download ClinTox dataset: {e}")
        return pd.DataFrame(columns=[COL_SMILES, COL_LABEL, COL_SOURCE])


def merge_datasets(
    dili_df: pd.DataFrame,
    clintox_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge DILI and ClinTox datasets into a unified toxicity dataset.

    Merging Logic (documented):
    - DILI label=1: Drug causes liver injury → toxic
    - DILI label=0: Drug does not cause liver injury → non-toxic
    - ClinTox CT_TOX=1: Drug failed clinical trial due to toxicity → toxic
    - ClinTox CT_TOX=0: Drug did not fail due to toxicity → non-toxic

    Both datasets contribute to a binary toxicity classification.
    The DILI dataset is specifically about hepatotoxicity.
    The ClinTox dataset covers broader clinical trial toxicity.

    This merge is a pragmatic choice to increase dataset size.
    The limitation is documented: ClinTox toxicity is not exclusively hepatotoxicity.

    Deduplication is handled in clean_data.py after canonicalization.

    Args:
        dili_df: DILI dataset with standardized columns.
        clintox_df: ClinTox dataset with standardized columns.

    Returns:
        Merged DataFrame with smiles, label, source columns.
    """
    frames = []

    if len(dili_df) > 0:
        logger.info(f"Including DILI dataset: {len(dili_df)} compounds")
        frames.append(dili_df)

    if len(clintox_df) > 0:
        logger.info(f"Including ClinTox dataset: {len(clintox_df)} compounds")
        frames.append(clintox_df)

    if not frames:
        raise ValueError(
            "No datasets available for merging. "
            "Ensure at least one dataset can be downloaded."
        )

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Merged dataset: {len(merged)} total compounds "
        f"({merged[COL_LABEL].sum()} toxic, "
        f"{(merged[COL_LABEL] == 0).sum()} non-toxic)"
    )

    return merged


def load_all_data(force_download: bool = False) -> pd.DataFrame:
    """
    Main entry point: download, load, and merge all datasets.

    Args:
        force_download: If True, re-download even if cached files exist.

    Returns:
        Merged and standardized DataFrame.
    """
    if force_download:
        # Remove cached files to force re-download
        for f in RAW_DATA_DIR.glob("*_raw.csv"):
            f.unlink()
            logger.info(f"Removed cached file: {f}")

    logger.info("=" * 60)
    logger.info("Loading toxicity datasets...")
    logger.info("=" * 60)

    dili_df = download_dili_dataset()
    clintox_df = download_clintox_dataset()

    merged = merge_datasets(dili_df, clintox_df)

    # Save merged raw data
    merged_path = PROCESSED_DATA_DIR / "merged_raw.csv"
    merged.to_csv(merged_path, index=False)
    logger.info(f"Merged raw data saved to {merged_path}")

    return merged


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for reporting.

    Args:
        df: Dataset DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_compounds": len(df),
        "toxic_count": int((df[COL_LABEL] == 1).sum()),
        "nontoxic_count": int((df[COL_LABEL] == 0).sum()),
        "toxic_ratio": float((df[COL_LABEL] == 1).mean()),
        "sources": df[COL_SOURCE].value_counts().to_dict() if COL_SOURCE in df.columns else {},
        "missing_smiles": int(df[COL_SMILES].isna().sum()),
        "missing_labels": int(df[COL_LABEL].isna().sum()),
    }

    logger.info(f"Dataset Summary:")
    logger.info(f"  Total compounds: {summary['total_compounds']}")
    logger.info(f"  Toxic: {summary['toxic_count']} ({summary['toxic_ratio']:.1%})")
    logger.info(f"  Non-toxic: {summary['nontoxic_count']}")
    logger.info(f"  Sources: {summary['sources']}")

    return summary
