#!/usr/bin/env python3
"""
ToxiPred — Full Pipeline Runner

Executes the complete ML pipeline end-to-end:
1. Download and load datasets
2. Clean and validate data
3. Featurize molecules
4. Train and compare models
5. Evaluate on test set
6. Generate explainability artifacts
7. Save all outputs

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-download
    python scripts/run_pipeline.py --tune
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logger
from src.utils.config import PROCESSED_DATA_DIR, METRICS_DIR

logger = setup_logger("pipeline", log_file=PROJECT_ROOT / "pipeline.log")


def parse_args():
    parser = argparse.ArgumentParser(description="ToxiPred Training Pipeline")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download (use cached data)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning (takes longer)",
    )
    parser.add_argument(
        "--feature-mode", type=str, default="combined",
        choices=["fingerprint", "descriptors", "combined"],
        help="Feature generation mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  ToxiPred — Full Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Feature mode: {args.feature_mode}")
    logger.info(f"Skip download: {args.skip_download}")
    logger.info(f"Hyperparameter tuning: {args.tune}")
    logger.info("")

    # ---- Step 1: Load Data ----
    logger.info("STEP 1: Loading datasets...")
    from src.data.load_data import load_all_data, get_data_summary

    if args.skip_download and (PROCESSED_DATA_DIR / "cleaned_data.csv").exists():
        import pandas as pd
        df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
        logger.info(f"Loaded cached cleaned data: {len(df)} compounds")
    else:
        df = load_all_data(force_download=not args.skip_download)
        summary = get_data_summary(df)

        # ---- Step 2: Clean Data ----
        logger.info("\nSTEP 2: Cleaning data...")
        from src.data.clean_data import clean_dataset

        df = clean_dataset(df)

        # Save cleaned data
        df.to_csv(PROCESSED_DATA_DIR / "cleaned_data.csv", index=False)
        logger.info(f"Cleaned data saved: {len(df)} compounds")

    # ---- Step 3: Train Models ----
    logger.info("\nSTEP 3: Training models...")
    from src.models.train import run_training_pipeline

    results = run_training_pipeline(df, feature_mode=args.feature_mode)

    # ---- Step 4: Hyperparameter Tuning (Optional) ----
    if args.tune:
        logger.info("\nSTEP 4: Hyperparameter tuning...")
        from src.models.tune import tune_and_retrain
        from src.models.train import compute_scale_pos_weight

        spw = compute_scale_pos_weight(results["y_train"])
        tuned_model = tune_and_retrain(
            results["X_train"], results["y_train"],
            results["X_val"], results["y_val"],
            scale_pos_weight=spw,
            n_iter=30,  # Reduced for speed
        )
        # Optionally replace the main model
        results["model"] = tuned_model

    # ---- Step 5: Evaluate ----
    logger.info("\nSTEP 5: Evaluation...")
    from src.models.evaluate import run_evaluation

    eval_results = run_evaluation(
        model=results["model"],
        calibrated_model=results.get("calibrated_model"),
        X_test=results["X_test"],
        y_test=results["y_test"],
        cv_comparison=results.get("cv_comparison"),
        test_df=results.get("test_df"),
    )

    # ---- Step 6: Explainability ----
    logger.info("\nSTEP 6: Generating explainability artifacts...")
    try:
        from src.explainability.shap_explain import run_global_explanation

        explanation_results = run_global_explanation(
            model=results["model"],
            X_train=results["X_train"],
            feature_names=results["feature_names"],
        )
        logger.info("SHAP explainability artifacts generated successfully")
    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")

    # ---- Summary ----
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Duration: {elapsed:.1f} seconds")
    logger.info(f"  Test ROC-AUC: {eval_results['metrics_optimal'].get('roc_auc', 'N/A')}")
    logger.info(f"  Test F1:      {eval_results['metrics_optimal'].get('f1_score', 'N/A')}")
    logger.info(f"  Threshold:    {eval_results['optimal_threshold']}")
    logger.info("")
    logger.info("  Artifacts saved to:")
    logger.info(f"    Models:  {PROJECT_ROOT / 'models'}")
    logger.info(f"    Plots:   {PROJECT_ROOT / 'artifacts' / 'plots'}")
    logger.info(f"    Metrics: {PROJECT_ROOT / 'artifacts' / 'metrics'}")
    logger.info(f"    Explanations: {PROJECT_ROOT / 'artifacts' / 'explanations'}")
    logger.info("")
    logger.info("  To run the app:")
    logger.info("    streamlit run src/app/streamlit_app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
