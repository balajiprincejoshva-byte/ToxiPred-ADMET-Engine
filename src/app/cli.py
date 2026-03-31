"""
ToxiPred — Command Line Interface (CLI)

Provides a terminal-based interface for single-molecule and batch 
toxicity predictions. Optimized for integration into bioinformatics pipelines.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

from src.models.predict import ToxiPredPredictor
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="ToxiPred — In-Silico Hepatotoxicity Prediction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict toxicity for a single SMILES")
    predict_parser.add_argument("smiles", help="SMILES string of the molecule")
    predict_parser.add_argument("--json", action="store_true", help="Output result as JSON")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Predict toxicity for a CSV file")
    batch_parser.add_argument("input", help="Path to input CSV file")
    batch_parser.add_argument("output", help="Path to save output CSV file")
    batch_parser.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column")
    
    args = parser.parse_args()
    
    if args.command == "predict":
        try:
            predictor = ToxiPredPredictor()
            result = predictor.predict(args.smiles)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("\n" + "="*40)
                print(" TOXIPRED PREDICTION RESULT")
                print("="*40)
                print(f"Input SMILES:  {result['input_smiles']}")
                print(f"Status:        {'Valid' if result['is_valid'] else 'Invalid'}")
                
                if result['is_valid']:
                    risk_color = "TOXIC" if result['prediction'] == "Toxic" else "NON-TOXIC"
                    print(f"Prediction:    {risk_color}")
                    print(f"Probability:   {result['probability']:.4f}")
                    print(f"Confidence:    {result['confidence']}")
                    print(f"AD Status:     {result['ad_status']}")
                    
                    if result['warnings']:
                        print("\nWarnings:")
                        for w in result['warnings']:
                            print(f"  - {w}")
                print("="*40 + "\n")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)
            
    elif args.command == "batch":
        try:
            predictor = ToxiPredPredictor()
            print(f"Running batch prediction on {args.input}...")
            results_df = predictor.predict_from_csv(
                args.input, 
                args.output, 
                smiles_column=args.smiles_col
            )
            print(f"Success! Results saved to {args.output}")
            print(f"Toxic detected: {(results_df['prediction'] == 'Toxic').sum()} / {len(results_df)}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
