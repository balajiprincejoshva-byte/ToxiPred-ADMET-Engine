"""
ToxiPred — SMILES Validation Tests

Tests for the SMILES validation and canonicalization utilities.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.validation import (
    validate_smiles,
    is_valid_smiles,
    canonicalize_smiles,
    check_out_of_domain,
    smiles_to_mol,
)


class TestValidateSmiles:
    """Tests for SMILES validation."""

    def test_valid_smiles_aspirin(self):
        """Aspirin should be a valid SMILES."""
        is_valid, canonical = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert is_valid is True
        assert canonical is not None
        assert len(canonical) > 0

    def test_valid_smiles_caffeine(self):
        """Caffeine should be a valid SMILES."""
        is_valid, canonical = validate_smiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert is_valid is True
        assert canonical is not None

    def test_valid_smiles_acetaminophen(self):
        """Acetaminophen should be a valid SMILES."""
        is_valid, canonical = validate_smiles("CC(=O)Nc1ccc(O)cc1")
        assert is_valid is True
        assert canonical is not None

    def test_invalid_smiles_random_string(self):
        """Random text should be invalid SMILES."""
        is_valid, canonical = validate_smiles("NOT_A_MOLECULE")
        assert is_valid is False
        assert canonical is None

    def test_invalid_smiles_empty_string(self):
        """Empty string should be invalid."""
        is_valid, canonical = validate_smiles("")
        assert is_valid is False
        assert canonical is None

    def test_invalid_smiles_none(self):
        """None input should be invalid."""
        is_valid, canonical = validate_smiles(None)
        assert is_valid is False
        assert canonical is None

    def test_invalid_smiles_whitespace(self):
        """Whitespace-only string should be invalid."""
        is_valid, canonical = validate_smiles("   ")
        assert is_valid is False
        assert canonical is None

    def test_invalid_smiles_special_chars(self):
        """Special characters should be invalid."""
        is_valid, canonical = validate_smiles("!@#$%^&*()")
        assert is_valid is False
        assert canonical is None

    def test_canonicalization_consistency(self):
        """Same molecule in different SMILES should give same canonical."""
        # Both represent benzene
        _, canon1 = validate_smiles("c1ccccc1")
        _, canon2 = validate_smiles("C1=CC=CC=C1")
        assert canon1 == canon2

    def test_smiles_with_whitespace(self):
        """SMILES with leading/trailing whitespace should be handled."""
        is_valid, canonical = validate_smiles("  CC(=O)O  ")
        assert is_valid is True
        assert canonical is not None


class TestIsValidSmiles:
    """Tests for the convenience wrapper."""

    def test_valid(self):
        assert is_valid_smiles("CCO") is True

    def test_invalid(self):
        assert is_valid_smiles("INVALID") is False

    def test_empty(self):
        assert is_valid_smiles("") is False


class TestCanonicalizeSmiles:
    """Tests for canonicalization."""

    def test_returns_canonical(self):
        result = canonicalize_smiles("OCC")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_returns_none(self):
        result = canonicalize_smiles("INVALID")
        assert result is None


class TestOutOfDomain:
    """Tests for out-of-domain detection."""

    def test_normal_drug(self):
        """Aspirin should be in-domain."""
        result = check_out_of_domain("CC(=O)Oc1ccccc1C(=O)O")
        assert result["ood_mw"] is False
        assert result["mw_value"] is not None

    def test_very_small_molecule(self):
        """Methane (CH4) should trigger out-of-domain."""
        result = check_out_of_domain("C")
        assert len(result["warnings"]) > 0

    def test_invalid_smiles(self):
        """Invalid SMILES should return warnings."""
        result = check_out_of_domain("INVALID")
        assert len(result["warnings"]) > 0


class TestSmilesToMol:
    """Tests for SMILES to RDKit Mol conversion."""

    def test_valid_conversion(self):
        mol = smiles_to_mol("CCO")
        assert mol is not None

    def test_invalid_returns_none(self):
        mol = smiles_to_mol("INVALID")
        assert mol is None

    def test_none_input(self):
        mol = smiles_to_mol(None)
        assert mol is None

    def test_empty_input(self):
        mol = smiles_to_mol("")
        assert mol is None
