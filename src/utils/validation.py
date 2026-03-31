"""
ToxiPred — SMILES Validation Utilities

Functions for validating, canonicalizing, and checking molecular SMILES strings
using RDKit. Provides robust error handling for invalid chemical inputs.
"""

from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.utils.logging_utils import get_logger
from src.utils.config import OOD_MW_MIN, OOD_MW_MAX

logger = get_logger(__name__)


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a SMILES string and return its canonical form.

    Args:
        smiles: Input SMILES string.

    Returns:
        Tuple of (is_valid, canonical_smiles_or_None).
    """
    if not smiles or not isinstance(smiles, str):
        return False, None

    smiles = smiles.strip()
    if len(smiles) == 0:
        return False, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical
    except Exception:
        return False, None


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid (convenience wrapper)."""
    valid, _ = validate_smiles(smiles)
    return valid


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Return the canonical SMILES form, or None if invalid.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string or None.
    """
    _, canonical = validate_smiles(smiles)
    return canonical


def check_out_of_domain(smiles: str) -> dict:
    """
    Check if a molecule might be outside the model's training domain.

    Returns a dictionary with warning flags:
    - ood_mw: True if molecular weight is outside expected range
    - mw_value: Actual molecular weight
    - warnings: List of human-readable warning strings

    Args:
        smiles: Valid SMILES string.

    Returns:
        Dictionary with out-of-domain check results.
    """
    result = {
        "ood_mw": False,
        "mw_value": None,
        "warnings": [],
    }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["warnings"].append("Could not parse SMILES for domain check.")
            return result

        mw = Descriptors.MolWt(mol)
        result["mw_value"] = round(mw, 2)

        if mw < OOD_MW_MIN:
            result["ood_mw"] = True
            result["warnings"].append(
                f"Molecular weight ({mw:.1f} Da) is below typical drug range "
                f"({OOD_MW_MIN}–{OOD_MW_MAX} Da). Prediction may be less reliable."
            )
        elif mw > OOD_MW_MAX:
            result["ood_mw"] = True
            result["warnings"].append(
                f"Molecular weight ({mw:.1f} Da) is above typical drug range "
                f"({OOD_MW_MIN}–{OOD_MW_MAX} Da). Prediction may be less reliable."
            )

        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms < 5:
            result["warnings"].append(
                f"Very small molecule ({num_atoms} heavy atoms). "
                "Prediction confidence may be low."
            )

    except Exception as e:
        result["warnings"].append(f"Domain check error: {str(e)}")

    return result


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit Mol object.

    Args:
        smiles: Input SMILES string.

    Returns:
        RDKit Mol object or None if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None
