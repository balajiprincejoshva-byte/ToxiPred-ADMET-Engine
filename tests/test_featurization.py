"""
ToxiPred — Featurization Tests

Tests for Morgan fingerprint and molecular descriptor computation.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.featurize import (
    smiles_to_morgan_fingerprint,
    batch_fingerprints,
)
from src.features.descriptors import (
    compute_single_descriptors,
    compute_descriptors,
    get_descriptor_summary,
    DESCRIPTOR_FEATURE_NAMES,
)
from src.utils.config import MORGAN_NBITS


class TestMorganFingerprints:
    """Tests for Morgan fingerprint generation."""

    def test_valid_molecule_shape(self):
        """Valid molecule should produce correct shape fingerprint."""
        fp = smiles_to_morgan_fingerprint("CCO")
        assert fp is not None
        assert fp.shape == (MORGAN_NBITS,)
        assert fp.dtype == np.int8

    def test_valid_molecule_binary(self):
        """Fingerprint values should be 0 or 1."""
        fp = smiles_to_morgan_fingerprint("c1ccccc1")
        assert fp is not None
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_molecule_returns_none(self):
        """Invalid SMILES should return None."""
        fp = smiles_to_morgan_fingerprint("INVALID")
        assert fp is None

    def test_deterministic(self):
        """Same molecule should produce same fingerprint."""
        fp1 = smiles_to_morgan_fingerprint("CC(=O)Nc1ccc(O)cc1")
        fp2 = smiles_to_morgan_fingerprint("CC(=O)Nc1ccc(O)cc1")
        assert fp1 is not None and fp2 is not None
        np.testing.assert_array_equal(fp1, fp2)

    def test_different_molecules_differ(self):
        """Different molecules should generally have different fingerprints."""
        fp1 = smiles_to_morgan_fingerprint("CCO")
        fp2 = smiles_to_morgan_fingerprint("c1ccccc1")
        assert fp1 is not None and fp2 is not None
        assert not np.array_equal(fp1, fp2)

    def test_with_bit_info(self):
        """Should return bit info when requested."""
        fp, bit_info = smiles_to_morgan_fingerprint("CCO", return_bit_info=True)
        assert fp is not None
        assert bit_info is not None
        assert isinstance(bit_info, dict)
        assert len(bit_info) > 0

    def test_custom_radius(self):
        """Should work with different radius values."""
        fp = smiles_to_morgan_fingerprint("CCO", radius=3)
        assert fp is not None
        assert fp.shape == (MORGAN_NBITS,)


class TestBatchFingerprints:
    """Tests for batch fingerprint generation."""

    def test_batch_valid(self):
        """Batch of valid molecules should produce correct matrix."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        X, valid_idx = batch_fingerprints(smiles_list)
        assert X.shape == (3, MORGAN_NBITS)
        assert valid_idx == [0, 1, 2]

    def test_batch_with_invalid(self):
        """Batch with invalid molecules should skip them."""
        smiles_list = ["CCO", "INVALID", "c1ccccc1"]
        X, valid_idx = batch_fingerprints(smiles_list)
        assert X.shape == (2, MORGAN_NBITS)
        assert valid_idx == [0, 2]

    def test_batch_all_invalid(self):
        """Batch with all invalid should return empty matrix."""
        smiles_list = ["INVALID1", "INVALID2"]
        X, valid_idx = batch_fingerprints(smiles_list)
        assert X.shape[0] == 0
        assert valid_idx == []

    def test_batch_empty(self):
        """Empty batch should return empty matrix."""
        X, valid_idx = batch_fingerprints([])
        assert X.shape[0] == 0
        assert valid_idx == []


class TestMolecularDescriptors:
    """Tests for classical molecular descriptor computation."""

    def test_valid_molecule(self):
        """Valid molecule should produce correct number of descriptors."""
        desc = compute_single_descriptors("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        assert desc is not None
        assert desc.shape == (len(DESCRIPTOR_FEATURE_NAMES),)
        assert desc.dtype == np.float32

    def test_molecular_weight_reasonable(self):
        """Molecular weight should be in reasonable range for aspirin."""
        desc = compute_single_descriptors("CC(=O)Oc1ccccc1C(=O)O")
        assert desc is not None
        mw = desc[0]  # MolWt is first
        assert 170 < mw < 190  # Aspirin MW ~ 180.16

    def test_invalid_molecule_returns_none(self):
        """Invalid SMILES should return None."""
        desc = compute_single_descriptors("INVALID")
        assert desc is None

    def test_batch_descriptors(self):
        """Batch descriptor computation should work."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        X, valid_idx = compute_descriptors(smiles_list)
        assert X.shape == (3, len(DESCRIPTOR_FEATURE_NAMES))
        assert valid_idx == [0, 1, 2]

    def test_descriptor_summary(self):
        """Descriptor summary should return named dictionary."""
        summary = get_descriptor_summary("CCO")
        assert summary is not None
        assert isinstance(summary, dict)
        assert "MolWt" in summary
        assert "LogP" in summary

    def test_no_nan_values(self):
        """Descriptors should not contain NaN for valid molecules."""
        desc = compute_single_descriptors("CC(=O)Nc1ccc(O)cc1")
        assert desc is not None
        assert not np.any(np.isnan(desc))
