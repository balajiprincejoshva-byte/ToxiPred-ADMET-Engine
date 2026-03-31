"""
ToxiPred — SHAP Explainability Module

Provides model interpretability through SHAP (SHapley Additive exPlanations).
Supports both global and local explanations for XGBoost predictions.

Features:
- Global feature importance (SHAP summary plot)
- Local explanation for individual predictions (waterfall / force plot)
- Fingerprint bit → substructure mapping via RDKit bitInfo
- Descriptor-level contribution analysis
- Molecular visualization with optional substructure highlights

Important caveat:
  Morgan fingerprint bits are hashed representations. While we can map
  top SHAP features back to atomic environments using RDKit's bitInfo,
  this mapping is approximate — multiple substructures can hash to the
  same bit. Interpretations should be treated as indicative, not definitive.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    SHAP_TOP_FEATURES, SHAP_BACKGROUND_SAMPLES,
    EXPLANATIONS_DIR, PLOTS_DIR, PLOT_DPI,
    MORGAN_NBITS, MORGAN_RADIUS, FEATURE_MODE,
    COLOR_PRIMARY, COLOR_ACCENT,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    max_samples: int = SHAP_BACKGROUND_SAMPLES,
) -> Any:
    """
    Create a SHAP TreeExplainer for the model.

    For XGBoost, uses TreeExplainer for exact Shapley values.
    Falls back to KernelExplainer for other model types.

    Args:
        model: Trained model (XGBoost, RF, etc.).
        X_background: Background dataset for SHAP calculations.
        max_samples: Maximum background samples (for efficiency).

    Returns:
        SHAP explainer object.
    """
    import shap

    # Subsample background if too large
    if len(X_background) > max_samples:
        indices = np.random.RandomState(42).choice(
            len(X_background), max_samples, replace=False
        )
        X_background = X_background[indices]

    try:
        # TreeExplainer for tree-based models (fast, exact)
        explainer = shap.TreeExplainer(model)
        logger.info("Created SHAP TreeExplainer")
    except Exception:
        # Fallback to KernelExplainer
        logger.info("Falling back to SHAP KernelExplainer (slower)")
        explainer = shap.KernelExplainer(model.predict_proba, X_background)

    return explainer


def compute_shap_values(
    explainer: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values for a set of samples.

    Args:
        explainer: SHAP explainer object.
        X: Feature matrix.

    Returns:
        SHAP values array.
    """
    import shap

    shap_values = explainer.shap_values(X)

    # For binary classification, take positive class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return shap_values[1]
    return shap_values


def plot_global_importance(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = SHAP_TOP_FEATURES,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate and save a global SHAP feature importance summary plot.

    Args:
        shap_values: SHAP values array (n_samples × n_features).
        X: Feature matrix.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        save_path: Path to save the plot.
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use SHAP's built-in summary plot
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_size=None,
    )

    plt.title("Global Feature Importance (SHAP Values)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = save_path or (PLOTS_DIR / "shap_global_importance.png")
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Global SHAP importance plot saved to {save_path}")


def plot_global_bar_importance(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = SHAP_TOP_FEATURES,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate a bar chart of mean absolute SHAP values.

    More readable than the dot plot for presentations.

    Args:
        shap_values: SHAP values array.
        feature_names: Feature names.
        top_n: Number of top features.
        save_path: Save path.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

    # Get top N
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_names))

    bars = ax.barh(y_pos, top_values, color=COLOR_PRIMARY, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (PLOTS_DIR / "shap_bar_importance.png")
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP bar importance saved to {save_path}")


def explain_single_prediction(
    explainer: Any,
    X_single: np.ndarray,
    feature_names: Optional[List[str]] = None,
    smiles: Optional[str] = None,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a local explanation for a single molecule prediction.

    Returns the top contributing features and generates a waterfall plot.

    Args:
        explainer: SHAP explainer.
        X_single: Feature vector (1 × n_features or n_features,).
        feature_names: Feature names.
        smiles: Input SMILES (for labeling).
        top_n: Number of top features in explanation.
        save_path: Path to save the waterfall plot.

    Returns:
        Dictionary with explanation details.
    """
    import shap

    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    shap_values = compute_shap_values(explainer, X_single)
    if shap_values.ndim == 2:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(shap_vals))]

    # Get top contributing features
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature": feature_names[idx],
            "feature_index": int(idx),
            "shap_value": float(shap_vals[idx]),
            "feature_value": float(X_single[0, idx]),
            "direction": "toxic" if shap_vals[idx] > 0 else "non-toxic",
        })

    # Generate waterfall plot
    try:
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=X_single[0],
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=top_n, show=False)

        title = f"SHAP Explanation"
        if smiles:
            title += f"\n{smiles[:60]}{'...' if len(str(smiles)) > 60 else ''}"
        plt.title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()

        save_path = save_path or (EXPLANATIONS_DIR / "shap_waterfall.png")
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP waterfall plot saved to {save_path}")

    except Exception as e:
        logger.warning(f"Waterfall plot generation failed: {e}")

    return {
        "top_features": top_features,
        "shap_values": shap_vals.tolist(),
        "smiles": smiles,
    }


def map_fingerprint_bits_to_substructures(
    smiles: str,
    top_bit_indices: List[int],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> List[Dict]:
    """
    Map Morgan fingerprint bit indices to molecular substructures.

    Uses RDKit's bitInfo to identify which atomic environments
    activate specific fingerprint bits.

    CAVEAT: This mapping is approximate due to bit collisions in
    hashed fingerprints. Multiple distinct substructures can map
    to the same bit index. Treat results as indicative, not definitive.

    Args:
        smiles: SMILES string of the molecule.
        top_bit_indices: List of fingerprint bit indices to map.
        radius: Morgan fingerprint radius.
        n_bits: Number of fingerprint bits.

    Returns:
        List of dictionaries with bit → substructure mapping info.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)

    mappings = []
    for bit_idx in top_bit_indices:
        if bit_idx in bit_info:
            environments = bit_info[bit_idx]
            atoms_involved = set()
            for center_atom, env_radius in environments:
                # Get atoms within the radius from center
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, center_atom)
                atoms = set()
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
                atoms.add(center_atom)
                atoms_involved.update(atoms)

            mappings.append({
                "bit_index": bit_idx,
                "atoms": sorted(atoms_involved),
                "n_environments": len(environments),
                "note": "Approximate mapping (hashed fingerprint)",
            })
        else:
            mappings.append({
                "bit_index": bit_idx,
                "atoms": [],
                "n_environments": 0,
                "note": "Bit not active in this molecule",
            })

    return mappings


def render_molecule_with_highlights(
    smiles: str,
    highlight_atoms: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
    img_size: Tuple[int, int] = (400, 300),
) -> Optional[bytes]:
    """
    Render a molecule image with optional atom highlighting.

    Args:
        smiles: SMILES string.
        highlight_atoms: List of atom indices to highlight.
        save_path: Path to save the image.
        img_size: Image dimensions (width, height).

    Returns:
        PNG image bytes or None.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        if highlight_atoms:
            img = Draw.MolToImage(
                mol,
                size=img_size,
                highlightAtoms=highlight_atoms,
                highlightColor=(0.0, 0.83, 0.67),  # Biotech green
            )
        else:
            img = Draw.MolToImage(mol, size=img_size)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(save_path))
            logger.info(f"Molecule image saved to {save_path}")

        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.warning(f"Molecule rendering failed: {e}")
        return None


def run_global_explanation(
    model: Any,
    X_train: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full global explainability pipeline.

    Args:
        model: Trained model.
        X_train: Training features (for background).
        feature_names: Feature names.

    Returns:
        Dictionary with global SHAP analysis results.
    """
    logger.info("Running global SHAP analysis...")

    explainer = create_shap_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_train)

    # Generate plots
    plot_global_importance(shap_values, X_train, feature_names)
    plot_global_bar_importance(shap_values, feature_names)

    # Compute mean absolute importance
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    if feature_names:
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
    else:
        importance_df = pd.DataFrame({
            "feature_index": range(len(mean_abs)),
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(EXPLANATIONS_DIR / "feature_importance.csv", index=False)

    logger.info("Global SHAP analysis complete")

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "importance_df": importance_df,
    }
