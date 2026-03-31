"""
ToxiPred — Data Loading Tests

Tests for dataset loading, cleaning, and splitting functionality.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import COL_SMILES, COL_LABEL, COL_SOURCE
from src.data.clean_data import (
    validate_and_canonicalize_smiles,
    remove_invalid_molecules,
    deduplicate_molecules,
    report_class_balance,
)
from src.data.split_data import stratified_split


class TestDataCleaning:
    """Tests for the data cleaning pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            COL_SMILES: [
                "CCO",           # Ethanol (valid)
                "c1ccccc1",      # Benzene (valid)
                "INVALID",       # Invalid
                "CC(=O)O",       # Acetic acid (valid)
                None,            # Missing
                "",              # Empty
                "CCO",           # Duplicate
            ],
            COL_LABEL: [0, 1, 0, 1, 0, 1, 1],
            COL_SOURCE: ["test"] * 7,
        })

    def test_validate_canonicalize(self, sample_df):
        """Should validate and canonicalize SMILES correctly."""
        result = validate_and_canonicalize_smiles(sample_df)
        assert "_valid" in result.columns
        assert result["_valid"].sum() == 4  # 3 valid + 1 duplicate = 4 valid

    def test_remove_invalid(self, sample_df):
        """Should remove invalid and missing SMILES."""
        validated = validate_and_canonicalize_smiles(sample_df)
        cleaned = remove_invalid_molecules(validated)
        assert len(cleaned) == 4  # Only valid SMILES remain
        assert "_valid" not in cleaned.columns

    def test_deduplicate(self, sample_df):
        """Should remove duplicate SMILES."""
        validated = validate_and_canonicalize_smiles(sample_df)
        cleaned = remove_invalid_molecules(validated)
        deduped = deduplicate_molecules(cleaned)
        # Should have at most 3 unique valid molecules
        assert len(deduped) <= len(cleaned)
        assert deduped[COL_SMILES].nunique() == len(deduped)

    def test_class_balance_report(self, sample_df):
        """Should return class balance statistics."""
        validated = validate_and_canonicalize_smiles(sample_df)
        cleaned = remove_invalid_molecules(validated)
        balance = report_class_balance(cleaned)

        assert "total" in balance
        assert "toxic" in balance
        assert "nontoxic" in balance
        assert "toxic_ratio" in balance
        assert balance["total"] == len(cleaned)
        assert balance["toxic"] + balance["nontoxic"] == balance["total"]

    def test_labels_are_binary(self, sample_df):
        """Cleaned data should only have binary labels."""
        validated = validate_and_canonicalize_smiles(sample_df)
        cleaned = remove_invalid_molecules(validated)
        assert set(cleaned[COL_LABEL].unique()).issubset({0, 1})


class TestDataSplitting:
    """Tests for stratified data splitting."""

    @pytest.fixture
    def clean_df(self):
        """Create a larger clean DataFrame for splitting tests."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            COL_SMILES: [f"C{'C' * i}O" for i in range(n)],
            COL_LABEL: np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            COL_SOURCE: ["test"] * n,
        })

    def test_split_ratios(self, clean_df):
        """Split sizes should approximately match specified ratios."""
        train, val, test = stratified_split(clean_df)
        total = len(clean_df)

        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05

    def test_no_overlap(self, clean_df):
        """Splits should not share any SMILES."""
        train, val, test = stratified_split(clean_df)

        train_smi = set(train[COL_SMILES].tolist())
        val_smi = set(val[COL_SMILES].tolist())
        test_smi = set(test[COL_SMILES].tolist())

        assert len(train_smi & val_smi) == 0
        assert len(train_smi & test_smi) == 0
        assert len(val_smi & test_smi) == 0

    def test_stratification(self, clean_df):
        """Each split should roughly preserve class ratios."""
        train, val, test = stratified_split(clean_df)

        overall_ratio = clean_df[COL_LABEL].mean()
        train_ratio = train[COL_LABEL].mean()
        val_ratio = val[COL_LABEL].mean()
        test_ratio = test[COL_LABEL].mean()

        # Allow 10% tolerance
        assert abs(train_ratio - overall_ratio) < 0.10
        assert abs(val_ratio - overall_ratio) < 0.15
        assert abs(test_ratio - overall_ratio) < 0.15

    def test_total_preserved(self, clean_df):
        """Total samples should be preserved after splitting."""
        train, val, test = stratified_split(clean_df)
        assert len(train) + len(val) + len(test) == len(clean_df)

    def test_reproducible(self, clean_df):
        """Same seed should give same split."""
        train1, val1, test1 = stratified_split(clean_df, random_seed=42)
        train2, val2, test2 = stratified_split(clean_df, random_seed=42)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
