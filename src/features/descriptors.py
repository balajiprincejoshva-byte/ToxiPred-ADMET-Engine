"""
ToxiPred — Classical Molecular Descriptors

Computes RDKit-based physicochemical descriptors commonly used
in ADMET prediction and drug-likeness assessment.

Descriptors computed:
- Molecular Weight (MolWt)
- LogP (Wildman-Crippen partition coefficient)
- Hydrogen Bond Donors (HBD)
- Hydrogen Bond Acceptors (HBA)
- Topological Polar Surface Area (TPSA)
- Rotatable Bonds
- Ring Count
- Aromatic Fraction (fraction of aromatic atoms)
- Fraction CSP3 (fraction of sp3-hybridized carbons)
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Ordered list of descriptor names (matches compute order)
DESCRIPTOR_FEATURE_NAMES = [
    "MolWt",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "RotatableBonds",
    "RingCount",
    "AromaticFraction",
    "FractionCSP3",
]


def compute_single_descriptors(smiles: str) -> Optional[np.ndarray]:
    """
    Compute classical molecular descriptors for a single SMILES string.

    Args:
        smiles: Input SMILES string.

    Returns:
        Numpy array of descriptor values, or None if molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Compute each descriptor
        mol_wt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        ring_count = Lipinski.RingCount(mol)

        # Aromatic fraction: ratio of aromatic atoms to total heavy atoms
        n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        n_heavy = mol.GetNumHeavyAtoms()
        aromatic_fraction = n_aromatic / n_heavy if n_heavy > 0 else 0.0

        # Fraction CSP3
        fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

        descriptors = np.array([
            mol_wt,
            logp,
            hbd,
            hba,
            tpsa,
            rotatable_bonds,
            ring_count,
            aromatic_fraction,
            fraction_csp3,
        ], dtype=np.float32)

        return descriptors

    except Exception as e:
        logger.warning(f"Descriptor computation failed for '{smiles}': {e}")
        return None


def compute_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute descriptors for a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Tuple of:
        - Numpy array of shape (n_valid, n_descriptors)
        - List of valid indices (for alignment with original data)
    """
    logger.info(f"Computing molecular descriptors for {len(smiles_list)} molecules...")

    all_descriptors = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        desc = compute_single_descriptors(smi)
        if desc is not None:
            all_descriptors.append(desc)
            valid_indices.append(idx)

    n_failed = len(smiles_list) - len(valid_indices)
    if n_failed > 0:
        logger.warning(
            f"{n_failed}/{len(smiles_list)} molecules failed descriptor computation"
        )

    X = np.array(all_descriptors, dtype=np.float32)
    logger.info(f"Descriptor matrix shape: {X.shape}")
    return X, valid_indices


def get_descriptor_summary(smiles: str) -> Optional[dict]:
    """
    Get a human-readable dictionary of descriptors for a single molecule.

    Useful for display in the UI.

    Args:
        smiles: Input SMILES string.

    Returns:
        Dictionary mapping descriptor names to values, or None.
    """
    desc = compute_single_descriptors(smiles)
    if desc is None:
        return None

    return dict(zip(DESCRIPTOR_FEATURE_NAMES, desc.tolist()))
