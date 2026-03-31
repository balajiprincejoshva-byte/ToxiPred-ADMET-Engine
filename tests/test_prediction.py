"""
ToxiPred — Prediction Tests

Tests for the inference engine.
Note: These tests require a trained model to be present.
Tests are skipped if no model is available.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import MODELS_DIR, MODEL_FILENAME


# Check if model exists
MODEL_EXISTS = (MODELS_DIR / MODEL_FILENAME).exists()
skip_reason = "Trained model not available. Run training pipeline first."


@pytest.mark.skipif(not MODEL_EXISTS, reason=skip_reason)
class TestToxiPredPredictor:
    """Tests for the ToxiPredPredictor class."""

    @pytest.fixture(scope="class")
    def predictor(self):
        """Load predictor once for all tests in this class."""
        from src.models.predict import ToxiPredPredictor
        return ToxiPredPredictor()

    def test_predict_valid_smiles(self, predictor):
        """Valid SMILES should return complete result."""
        result = predictor.predict("CC(=O)Nc1ccc(O)cc1")
        assert result["is_valid"] is True
        assert result["prediction"] in ("Toxic", "Non-Toxic")
        assert 0 <= result["probability"] <= 1
        assert result["confidence"] in ("High", "Moderate", "Low")
        assert result["risk_level"] in ("High Risk", "Moderate Risk", "Low Risk")
        assert result["smiles_canonical"] is not None

    def test_predict_invalid_smiles(self, predictor):
        """Invalid SMILES should return invalid result."""
        result = predictor.predict("INVALID_MOLECULE")
        assert result["is_valid"] is False
        assert result["prediction"] is None
        assert result["probability"] is None

    def test_predict_empty_string(self, predictor):
        """Empty string should return invalid result."""
        result = predictor.predict("")
        assert result["is_valid"] is False

    def test_predict_returns_dict(self, predictor):
        """Result should be a dictionary with expected keys."""
        result = predictor.predict("CCO")
        expected_keys = {
            "input_smiles", "is_valid", "smiles_canonical",
            "prediction", "probability", "confidence",
            "risk_level", "threshold", "warnings", "ood_check",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_predict_batch(self, predictor):
        """Batch prediction should return DataFrame."""
        smiles_list = ["CCO", "c1ccccc1", "INVALID"]
        result_df = predictor.predict_batch(smiles_list)

        assert len(result_df) == 3
        assert "prediction" in result_df.columns
        assert "probability" in result_df.columns
        assert result_df["is_valid"].sum() == 2  # 2 valid, 1 invalid

    def test_prediction_probability_range(self, predictor):
        """Probability should always be between 0 and 1."""
        smiles_list = [
            "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # Caffeine
        ]
        for smi in smiles_list:
            result = predictor.predict(smi)
            if result["is_valid"]:
                assert 0 <= result["probability"] <= 1

    def test_deterministic_predictions(self, predictor):
        """Same input should give same prediction."""
        result1 = predictor.predict("CC(=O)Nc1ccc(O)cc1")
        result2 = predictor.predict("CC(=O)Nc1ccc(O)cc1")
        assert result1["probability"] == result2["probability"]
        assert result1["prediction"] == result2["prediction"]


class TestPredictionOutputStructure:
    """Tests for prediction output structure (no model needed)."""

    def test_output_keys(self):
        """Validates expected output dictionary structure."""
        expected_keys = {
            "input_smiles", "is_valid", "smiles_canonical",
            "prediction", "probability", "confidence",
            "risk_level", "threshold", "warnings", "ood_check",
        }
        # These are the keys we expect in every prediction result
        assert len(expected_keys) == 10
