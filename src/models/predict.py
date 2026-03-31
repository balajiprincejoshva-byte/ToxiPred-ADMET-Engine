"""
ToxiPred — Inference Engine

Production-ready prediction module for hepatotoxicity risk assessment.
Supports single-molecule and batch prediction with structured output.

Features:
- SMILES validation before prediction
- Out-of-domain detection
- Confidence scoring
- Model calibration (when available)
- CSV batch processing
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.config import (
    MODELS_DIR, FEATURE_MODE, DEFAULT_THRESHOLD,
    MODEL_FILENAME, CALIBRATED_MODEL_FILENAME,
    SCALER_FILENAME, FEATURE_NAMES_FILENAME,
    THRESHOLD_FILENAME, MORGAN_NBITS,
    AD_MODEL_FILENAME,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import load_model, load_json
from src.utils.validation import validate_smiles, check_out_of_domain
from src.features.featurize import smiles_to_morgan_fingerprint
from src.features.descriptors import compute_single_descriptors
from src.models.domain import ApplicabilityDomain
from src.features.chem_insights import get_medicinal_chemistry_insights

logger = get_logger(__name__)


class ToxiPredPredictor:
    """
    Hepatotoxicity prediction engine.

    Loads trained model artifacts and provides a clean API for
    single-molecule and batch toxicity predictions.

    Usage:
        predictor = ToxiPredPredictor()
        result = predictor.predict("CC(=O)Nc1ccc(O)cc1")
        print(result)
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_calibrated: bool = True,
    ):
        """
        Initialize the predictor by loading model artifacts.

        Args:
            model_dir: Directory containing saved model files.
            use_calibrated: Whether to use calibrated model if available.
        """
        self.model_dir = model_dir or MODELS_DIR
        self.model = None
        self.calibrated_model = None
        self.ad_model = None
        self.scaler = None
        self.feature_mode = FEATURE_MODE
        self.threshold = DEFAULT_THRESHOLD
        self.feature_names = None
        self._loaded = False

        self._load_artifacts(use_calibrated)

    def _load_artifacts(self, use_calibrated: bool) -> None:
        """Load all model artifacts from disk."""
        try:
            # Load primary model
            model_path = self.model_dir / MODEL_FILENAME
            self.model = load_model(model_path)

            # Load calibrated model (optional)
            if use_calibrated:
                cal_path = self.model_dir / CALIBRATED_MODEL_FILENAME
                if cal_path.exists():
                    self.calibrated_model = load_model(cal_path)
                    logger.info("Calibrated model loaded")
                else:
                    logger.info("No calibrated model found, using uncalibrated")

            # Load applicability domain model
            ad_path = self.model_dir / AD_MODEL_FILENAME
            if ad_path.exists():
                self.ad_model = ApplicabilityDomain.load(ad_path)
                logger.info("Applicability Domain model loaded")

            # Load scaler (optional, for descriptor/combined modes)
            scaler_path = self.model_dir / SCALER_FILENAME
            if scaler_path.exists():
                self.scaler = load_model(scaler_path)
                logger.info("Descriptor scaler loaded")

            # Load feature configuration
            feat_path = self.model_dir / FEATURE_NAMES_FILENAME
            if feat_path.exists():
                feat_config = load_json(feat_path)
                self.feature_mode = feat_config.get("feature_mode", FEATURE_MODE)
                self.feature_names = feat_config.get("feature_names", None)

            # Load optimal threshold
            thresh_path = self.model_dir / THRESHOLD_FILENAME
            if thresh_path.exists():
                thresh_config = load_json(thresh_path)
                self.threshold = thresh_config.get("optimal_threshold", DEFAULT_THRESHOLD)
                logger.info(f"Using optimal threshold: {self.threshold:.3f}")

            self._loaded = True
            logger.info("Predictor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise

    def _featurize_single(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate features for a single molecule.

        Args:
            smiles: Validated SMILES string.

        Returns:
            Feature vector or None.
        """
        if self.feature_mode == "fingerprint":
            fp = smiles_to_morgan_fingerprint(smiles)
            return fp.reshape(1, -1).astype(np.float32) if fp is not None else None

        elif self.feature_mode == "descriptors":
            desc = compute_single_descriptors(smiles)
            if desc is None:
                return None
            desc = desc.reshape(1, -1)
            if self.scaler is not None:
                desc = self.scaler.transform(desc)
            return desc

        elif self.feature_mode == "combined":
            fp = smiles_to_morgan_fingerprint(smiles)
            desc = compute_single_descriptors(smiles)
            if fp is None or desc is None:
                return None
            fp = fp.reshape(1, -1).astype(np.float32)
            desc = desc.reshape(1, -1)
            if self.scaler is not None:
                desc = self.scaler.transform(desc)
            return np.hstack([fp, desc])

        else:
            raise ValueError(f"Unknown feature mode: {self.feature_mode}")

    def predict(self, smiles: str) -> Dict[str, Any]:
        """
        Predict hepatotoxicity risk for a single SMILES string.

        Returns a structured result dictionary with:
        - is_valid: Whether the SMILES was parseable
        - smiles_canonical: Canonical SMILES form
        - prediction: "Toxic" or "Non-Toxic"
        - probability: Float probability of toxicity
        - confidence: Confidence level description
        - risk_level: "High Risk", "Moderate Risk", or "Low Risk"
        - threshold: Classification threshold used
        - warnings: List of any warnings
        - ood_check: Out-of-domain analysis

        Args:
            smiles: Input SMILES string.

        Returns:
            Prediction result dictionary.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Initialize the predictor first.")

        result = {
            "input_smiles": smiles,
            "is_valid": False,
            "smiles_canonical": None,
            "prediction": None,
            "probability": None,
            "confidence": None,
            "risk_level": None,
            "threshold": self.threshold,
            "warnings": [],
            "ood_check": {},
            "ad_status": "Unknown",
            "chem_insights": {},
        }

        # Validate SMILES
        is_valid, canonical = validate_smiles(smiles)
        if not is_valid:
            result["warnings"].append("Invalid SMILES string. Cannot make prediction.")
            return result

        result["is_valid"] = True
        result["smiles_canonical"] = canonical

        # Out-of-domain check (Basic MW-based)
        ood = check_out_of_domain(canonical)
        result["ood_check"] = ood
        result["warnings"].extend(ood["warnings"])

        # Medicinal Chemistry Insights
        insights = get_medicinal_chemistry_insights(canonical)
        result["chem_insights"] = insights
        if insights.get("alerts"):
            result["warnings"].extend(insights["alerts"])

        # Featurize
        X = self._featurize_single(canonical)
        if X is None:
            result["warnings"].append("Feature generation failed for this molecule.")
            return result

        # Applicability Domain (Advanced Structural AD)
        if self.ad_model is not None:
            ad_res = self.ad_model.check_single(X)
            result["ad_status"] = ad_res["status"]
            if ad_res["is_ood"]:
                result["warnings"].append(ad_res["warning"])

        # Predict
        try:
            # Use calibrated model if available
            model = self.calibrated_model if self.calibrated_model else self.model
            prob = float(model.predict_proba(X)[0, 1])

            result["probability"] = round(prob, 4)
            result["prediction"] = "Toxic" if prob >= self.threshold else "Non-Toxic"

            # Risk level
            if prob >= 0.7:
                result["risk_level"] = "High Risk"
            elif prob >= 0.4:
                result["risk_level"] = "Moderate Risk"
            else:
                result["risk_level"] = "Low Risk"

            # Confidence assessment
            distance_from_threshold = abs(prob - self.threshold)
            if distance_from_threshold > 0.3:
                result["confidence"] = "High"
            elif distance_from_threshold > 0.15:
                result["confidence"] = "Moderate"
            else:
                result["confidence"] = "Low"

            if result["confidence"] == "Low":
                result["warnings"].append(
                    "Prediction is near the decision boundary. "
                    "Experimental validation is recommended."
                )

        except Exception as e:
            result["warnings"].append(f"Prediction error: {str(e)}")
            logger.error(f"Prediction failed: {e}")

        return result

    def predict_batch(
        self,
        smiles_list: List[str],
        sort_by_risk: bool = True,
    ) -> pd.DataFrame:
        """
        Predict toxicity for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            sort_by_risk: If True, sort results by toxicity probability (descending).

        Returns:
            DataFrame with prediction results for each molecule.
        """
        logger.info(f"Running batch prediction on {len(smiles_list)} molecules...")

        results = []
        for smi in smiles_list:
            result = self.predict(smi)
            results.append({
                "input_smiles": result["input_smiles"],
                "canonical_smiles": result.get("smiles_canonical"),
                "is_valid": result["is_valid"],
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "risk_level": result.get("risk_level"),
                "confidence": result.get("confidence"),
                "ad_status": result.get("ad_status"),
                "alerts": len(result.get("chem_insights", {}).get("alerts", [])),
                "warnings": "; ".join(result.get("warnings", [])),
            })

        df = pd.DataFrame(results)
        
        if sort_by_risk and not df.empty and "probability" in df.columns:
            df = df.sort_values(by="probability", ascending=False).reset_index(drop=True)
            
        logger.info(f"Batch prediction complete: {len(df)} results")
        return df

    def predict_from_csv(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        smiles_column: str = "smiles",
    ) -> pd.DataFrame:
        """
        Run predictions on a CSV file and save results.

        Args:
            input_path: Path to input CSV with SMILES column.
            output_path: Path to save results CSV.
            smiles_column: Name of the SMILES column.

        Returns:
            DataFrame with prediction results.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading molecules from {input_path}...")
        input_df = pd.read_csv(input_path)

        if smiles_column not in input_df.columns:
            # Try case-insensitive match
            matches = [c for c in input_df.columns if c.lower() == smiles_column.lower()]
            if matches:
                smiles_column = matches[0]
            else:
                raise ValueError(
                    f"Column '{smiles_column}' not found. "
                    f"Available columns: {list(input_df.columns)}"
                )

        smiles_list = input_df[smiles_column].tolist()
        results_df = self.predict_batch(smiles_list)

        # Merge with original data
        if len(input_df.columns) > 1:
            other_cols = [c for c in input_df.columns if c != smiles_column]
            for col in other_cols:
                results_df[col] = input_df[col].values

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        return results_df
