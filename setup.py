"""
ToxiPred — Package Setup

Minimal setup.py for development installation.
Install in editable mode: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="toxipred",
    version="1.0.0",
    description="In-Silico ADMET & Hepatotoxicity Prediction Engine",
    author="ToxiPred Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.43.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
        "rdkit>=2023.3.1",
        "tqdm>=4.65.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "toxipred=src.app.cli:main",
        ],
    },
)
