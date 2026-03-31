"""
ToxiPred — Molecular Featurization Module

Converts SMILES strings into machine-readable feature vectors using:
- Morgan (circular) fingerprints via RDKit
- Classical molecular descriptors
- Combined fingerprint + descriptor mode

Supports deterministic, reproducible feature generation with
graceful handling of invalid inputs.
"""

from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.config import (
    MORGAN_RADIUS, MORGAN_NBITS, FEATURE_MODE,
    COL_SMILES, DESCRIPTOR_NAMES,
)
from src.utils.logging_utils import get_logger
from src.features.descriptors import compute_descriptors, DESCRIPTOR_FEATURE_NAMES

logger = get_logger(__name__)


def smiles_to_morgan_fingerprint(
    smiles: str,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
    return_bit_info: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[Dict]]]:
    """
    Convert a single SMILES string to a Morgan fingerprint bit vector.

    Args:
        smiles: Input SMILES string.
        radius: Fingerprint radius (default: 2 for ECFP4-equivalent).
        n_bits: Number of bits in the fingerprint vector.
        return_bit_info: If True, also return bit→substructure mapping.

    Returns:
        Numpy array of shape (n_bits,) or None if invalid.
        If return_bit_info=True, returns (array, bit_info_dict).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if return_bit_info:
                return None, None
            return None

        if return_bit_info:
            bit_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits, bitInfo=bit_info
            )
            arr = np.zeros(n_bits, dtype=np.int8)
            fp_on = fp.GetOnBits()
            for bit in fp_on:
                arr[bit] = 1
            return arr, bit_info
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.int8)
            fp_on = fp.GetOnBits()
            for bit in fp_on:
                arr[bit] = 1
            return arr

    except Exception as e:
        logger.warning(f"Fingerprint generation failed for '{smiles}': {e}")
        if return_bit_info:
            return None, None
        return None


def batch_fingerprints(
    smiles_list: List[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate Morgan fingerprints for a batch of SMILES strings.

    Invalid molecules produce zero vectors and are logged.

    Args:
        smiles_list: List of SMILES strings.
        radius: Fingerprint radius.
        n_bits: Number of fingerprint bits.

    Returns:
        Tuple of:
        - Numpy array of shape (n_valid, n_bits)
        - List of valid indices (for alignment with original data)
    """
    logger.info(f"Generating Morgan fingerprints for {len(smiles_list)} molecules...")
    fingerprints = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fingerprint(smi, radius, n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            logger.debug(f"Skipping invalid SMILES at index {idx}: '{smi}'")

    n_failed = len(smiles_list) - len(valid_indices)
    if n_failed > 0:
        logger.warning(
            f"{n_failed}/{len(smiles_list)} molecules failed fingerprint generation"
        )

    X = np.array(fingerprints, dtype=np.float32)
    logger.info(f"Fingerprint matrix shape: {X.shape}")
    return X, valid_indices


def featurize_dataset(
    df: pd.DataFrame,
    mode: str = FEATURE_MODE,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Full featurization pipeline for a DataFrame.

    Supports three modes:
    - "fingerprint": Morgan fingerprints only (2048 bits)
    - "descriptors": Classical molecular descriptors only (~9 features)
    - "combined": Concatenation of fingerprints + descriptors

    Args:
        df: DataFrame with 'smiles' column.
        mode: Feature mode ("fingerprint", "descriptors", or "combined").

    Returns:
        Tuple of:
        - Feature matrix (numpy array)
        - List of feature names
        - Filtered DataFrame (rows with valid features only)
    """
    logger.info(f"Featurizing dataset in '{mode}' mode...")

    smiles_list = df[COL_SMILES].tolist()

    if mode == "fingerprint":
        X, valid_idx = batch_fingerprints(smiles_list)
        feature_names = [f"morgan_bit_{i}" for i in range(MORGAN_NBITS)]
        filtered_df = df.iloc[valid_idx].reset_index(drop=True)

    elif mode == "descriptors":
        X_desc, valid_idx = compute_descriptors(smiles_list)
        X = X_desc
        feature_names = DESCRIPTOR_FEATURE_NAMES
        filtered_df = df.iloc[valid_idx].reset_index(drop=True)

    elif mode == "combined":
        # Generate both features, then intersect valid indices
        X_fp, valid_fp = batch_fingerprints(smiles_list)
        X_desc, valid_desc = compute_descriptors(smiles_list)

        # Intersect valid indices
        valid_set_fp = set(valid_fp)
        valid_set_desc = set(valid_desc)
        valid_both = sorted(valid_set_fp & valid_set_desc)

        # Re-index into each feature matrix
        fp_reindex = [valid_fp.index(i) for i in valid_both]
        desc_reindex = [valid_desc.index(i) for i in valid_both]

        X_fp_filtered = X_fp[fp_reindex]
        X_desc_filtered = X_desc[desc_reindex]

        X = np.hstack([X_fp_filtered, X_desc_filtered])
        feature_names = (
            [f"morgan_bit_{i}" for i in range(MORGAN_NBITS)]
            + DESCRIPTOR_FEATURE_NAMES
        )
        filtered_df = df.iloc[valid_both].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown feature mode: '{mode}'. Use 'fingerprint', 'descriptors', or 'combined'.")

    logger.info(
        f"Featurization complete: {X.shape[0]} molecules × {X.shape[1]} features"
    )

    return X, feature_names, filtered_df


def get_fingerprint_feature_names() -> List[str]:
    """Return list of Morgan fingerprint feature names."""
    return [f"morgan_bit_{i}" for i in range(MORGAN_NBITS)]


def get_all_feature_names(mode: str = FEATURE_MODE) -> List[str]:
    """Return feature names for the given mode."""
    if mode == "fingerprint":
        return get_fingerprint_feature_names()
    elif mode == "descriptors":
        return DESCRIPTOR_FEATURE_NAMES
    elif mode == "combined":
        return get_fingerprint_feature_names() + DESCRIPTOR_FEATURE_NAMES
    else:
        raise ValueError(f"Unknown mode: {mode}")
