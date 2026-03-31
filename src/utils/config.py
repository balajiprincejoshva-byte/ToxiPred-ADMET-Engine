"""
ToxiPred — Central Configuration

All project-wide constants, paths, hyperparameters, and feature settings
are defined here to ensure reproducibility and easy tuning.
"""

import os
from pathlib import Path


# ==============================================================================
# Project Paths
# ==============================================================================

# Root of the ToxiPred project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_INPUTS_DIR = DATA_DIR / "sample_inputs"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"

# Output artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_INPUTS_DIR,
          MODELS_DIR, PLOTS_DIR, EXPLANATIONS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Reproducibility
# ==============================================================================

RANDOM_SEED = 42


# ==============================================================================
# Dataset Configuration
# ==============================================================================

# Column names after standardization
COL_SMILES = "smiles"
COL_LABEL = "label"
COL_SOURCE = "source"

# Data split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum required molecules after cleaning
MIN_DATASET_SIZE = 100


# ==============================================================================
# Molecular Featurization
# ==============================================================================

# Morgan fingerprint settings
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# Feature mode: "fingerprint", "descriptors", "combined"
FEATURE_MODE = "combined"

# Classical molecular descriptors to compute
DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "HBD",        # Hydrogen bond donors
    "HBA",        # Hydrogen bond acceptors
    "TPSA",       # Topological polar surface area
    "RotatableBonds",
    "RingCount",
    "AromaticFraction",
    "FractionCSP3",
]


# ==============================================================================
# Model Training
# ==============================================================================

# Cross-validation folds
CV_FOLDS = 5

# Primary scoring metric for model selection
PRIMARY_METRIC = "roc_auc"

# Classification threshold (can be optimized during evaluation)
DEFAULT_THRESHOLD = 0.5

# XGBoost default hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Hyperparameter search space for tuning
XGBOOST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

# Number of random search iterations
TUNING_N_ITER = 50

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


# ==============================================================================
# Evaluation
# ==============================================================================

# Plot settings
PLOT_DPI = 150
PLOT_FIGSIZE = (8, 6)
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# Color palette for plots
COLOR_PRIMARY = "#00D4AA"      # Biotech green
COLOR_SECONDARY = "#0077B6"    # Deep blue
COLOR_ACCENT = "#FF6B6B"       # Alert red
COLOR_NEUTRAL = "#6C757D"      # Gray
COLOR_BG_DARK = "#1A1A2E"      # Dark background


# ==============================================================================
# Explainability
# ==============================================================================

# Number of top SHAP features to display
SHAP_TOP_FEATURES = 20

# Maximum background samples for SHAP explainer
SHAP_BACKGROUND_SAMPLES = 100


# ==============================================================================
# Inference
# ==============================================================================

# Molecular weight bounds for out-of-domain warning
OOD_MW_MIN = 100.0
OOD_MW_MAX = 900.0

# Model artifact filenames
MODEL_FILENAME = "xgboost_model.joblib"
CALIBRATED_MODEL_FILENAME = "xgboost_calibrated.joblib"
SCALER_FILENAME = "descriptor_scaler.joblib"
FEATURE_NAMES_FILENAME = "feature_names.json"
THRESHOLD_FILENAME = "optimal_threshold.json"
AD_MODEL_FILENAME = "applicability_domain.joblib"


# ==============================================================================
# Streamlit App
# ==============================================================================

APP_TITLE = "ToxiPred"
APP_SUBTITLE = "In-Silico ADMET & Hepatotoxicity Prediction Engine"
APP_DESCRIPTION = (
    "A computational tool for predicting Drug-Induced Liver Injury (DILI) risk "
    "from molecular structures. Designed to support early-stage compound "
    "prioritization in drug discovery pipelines."
)
APP_DISCLAIMER = (
    "⚠️ **Research Tool Only** — This system is a computational screening aid, "
    "not a clinical diagnostic. Predictions should be validated through "
    "experimental assays and expert review before any decision-making. "
    "Do not use for clinical, regulatory, or patient-facing decisions."
)
