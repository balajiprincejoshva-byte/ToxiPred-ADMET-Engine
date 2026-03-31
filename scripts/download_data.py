#!/usr/bin/env python3
"""
ToxiPred — Data Download Script

Downloads public toxicity datasets for model training:
- DILI (Drug-Induced Liver Injury) from TDC
- ClinTox from MoleculeNet

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_all_data, get_data_summary
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Downloading toxicity datasets...")
    df = load_all_data(force_download=True)
    summary = get_data_summary(df)

    logger.info("\nDownload complete!")
    logger.info(f"Total compounds: {summary['total_compounds']}")
    logger.info(f"Toxic: {summary['toxic_count']}")
    logger.info(f"Non-toxic: {summary['nontoxic_count']}")


if __name__ == "__main__":
    main()
