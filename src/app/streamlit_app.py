"""
ToxiPred — Streamlit Application

A polished web interface for hepatotoxicity prediction from SMILES strings.
Provides single-molecule prediction, batch processing, model performance
visualization, and explainability analysis.

Run: streamlit run src/app/streamlit_app.py
"""

import sys
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    APP_TITLE, APP_SUBTITLE, APP_DESCRIPTION, APP_DISCLAIMER,
    MODELS_DIR, PLOTS_DIR, METRICS_DIR, EXPLANATIONS_DIR,
    FEATURE_MODE, MORGAN_NBITS,
)


# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title=f"{APP_TITLE} — Hepatotoxicity Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Custom CSS
# ==============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a2742 50%, #0d2137 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 212, 170, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #00D4AA;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        line-height: 1.5;
    }

    .main-header .subtitle {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #0f1923 0%, #162230 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 170, 0.15);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* Risk levels & Status */
    .risk-high { color: #FF6B6B; }
    .risk-moderate { color: #FFB347; }
    .risk-low { color: #00D4AA; }

    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Warning box */
    .warning-box {
        background: rgba(255, 179, 71, 0.1);
        border: 1px solid rgba(255, 179, 71, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #FFB347;
        font-size: 0.9rem;
    }

    /* Insight cards */
    .insight-box {
        background: rgba(0, 212, 170, 0.05);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .insight-alert {
        color: #FFB347;
        font-weight: 600;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
    }

    /* Comparison cards */
    .compare-card {
        background: linear-gradient(135deg, #0f1923 0%, #162230 100%);
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid rgba(226, 232, 240, 0.1);
        transition: all 0.3s ease;
    }

    .compare-winner {
        border: 2px solid #00D4AA;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.15);
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255, 107, 107, 0.08);
        border: 1px solid rgba(255, 107, 107, 0.2);
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* Sidebar info */
    .sidebar-info {
        background: rgba(0, 212, 170, 0.08);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 212, 170, 0.15);
    }

    /* Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00D4AA, #00B894);
        color: #0a1628;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
    }

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #00D4AA;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State & Caching
# ==============================================================================

@st.cache_resource
def load_predictor():
    """Load the prediction model (cached across sessions)."""
    try:
        from src.models.predict import ToxiPredPredictor
        predictor = ToxiPredPredictor()
        return predictor
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        st.info("Run the training pipeline first: `python scripts/run_pipeline.py`")
        return None


@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer (cached)."""
    try:
        from src.explainability.shap_explain import create_shap_explainer
        from src.utils.io_utils import load_model, load_json

        model = load_model(MODELS_DIR / "xgboost_model.joblib")

        # Load a small background dataset
        train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            from src.features.featurize import featurize_dataset
            X_bg, feat_names, _ = featurize_dataset(train_df, mode=FEATURE_MODE)
            # Use subset for efficiency
            if len(X_bg) > 100:
                X_bg = X_bg[:100]

            explainer = create_shap_explainer(model, X_bg)

            # Load feature names
            feat_path = MODELS_DIR / "feature_names.json"
            if feat_path.exists():
                feat_config = load_json(feat_path)
                feat_names = feat_config.get("feature_names", feat_names)

            return explainer, feat_names
    except Exception as e:
        st.warning(f"SHAP explainer could not be loaded: {e}")
    return None, None


def load_metrics():
    """Load saved evaluation metrics."""
    metrics = {}

    for name in ["test_metrics_default", "test_metrics_optimal"]:
        path = METRICS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)

    return metrics


# ==============================================================================
# Sidebar
# ==============================================================================

def render_sidebar():
    """Render the sidebar with project info and navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🧬</span>
            <h2 style="color: #00D4AA; margin: 0.5rem 0 0.2rem; font-size: 1.5rem;">ToxiPred</h2>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">v1.0.0</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="sidebar-info">
            <p style="font-size: 0.85rem; margin: 0; color: #e2e8f0;">
                <strong>🎯 Purpose</strong><br>
                Computational screening tool for Drug-Induced Liver Injury (DILI) risk assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-info">
            <p style="font-size: 0.85rem; margin: 0; color: #e2e8f0;">
                <strong>🧪 Model</strong><br>
                XGBoost classifier trained on DILI + ClinTox datasets with
                Morgan fingerprints and molecular descriptors.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Research Tool Only</strong><br>
            This is a computational screening aid. Predictions must be validated
            experimentally. Not for clinical use.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<p style='color: #64748b; font-size: 0.75rem; text-align: center;'>"
            "Built with RDKit · XGBoost · SHAP · Streamlit"
            "</p>",
            unsafe_allow_html=True,
        )


# ==============================================================================
# Tab 1: Predict
# ==============================================================================

def render_predict_tab():
    """Single molecule prediction interface."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 ToxiPred</h1>
        <p class="subtitle">In-Silico ADMET & Hepatotoxicity Prediction Engine</p>
        <p>Paste a SMILES string below to predict Drug-Induced Liver Injury (DILI) risk.
        The model analyzes molecular structure using Morgan fingerprints and physicochemical
        descriptors to estimate hepatotoxicity probability.</p>
    </div>
    """, unsafe_allow_html=True)

    # Session state for example molecule selection and auto-predict
    if "default_smiles" not in st.session_state:
        st.session_state.default_smiles = ""
    if "auto_predict" not in st.session_state:
        st.session_state.auto_predict = False

    col1, col2 = st.columns([3, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES String",
            value=st.session_state.default_smiles,
            placeholder="e.g., CC(=O)Nc1ccc(O)cc1 (Acetaminophen)",
            help="SMILES (Simplified Molecular Input Line Entry System) notation for your molecule.",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔬 Predict", use_container_width=True, key="predict_btn")

    # Auto-trigger prediction if an example was clicked
    if st.session_state.auto_predict:
        predict_clicked = True
        st.session_state.auto_predict = False

    # Example molecules
    with st.expander("📋 Example Molecules", expanded=False):
        examples = {
            "Acetaminophen (hepatotoxic)": "CC(=O)Nc1ccc(O)cc1",
            "Isoniazid (hepatotoxic)": "NNC(=O)c1ccncc1",
            "Diclofenac (hepatotoxic)": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
            "Aspirin (generally safe)": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine (generally safe)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "Ibuprofen (generally safe)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Metformin (generally safe)": "CN(C)C(=N)NC(=N)N",
        }

        cols = st.columns(4)
        for i, (name, smi) in enumerate(examples.items()):
            with cols[i % 4]:
                if st.button(name, key=f"example_{i}", use_container_width=True):
                    st.session_state.default_smiles = smi
                    st.session_state.auto_predict = True
                    st.rerun()

    # Prediction results
    if predict_clicked and smiles_input:
        predictor = load_predictor()
        if predictor is None:
            return

        with st.spinner("Analyzing molecular structure..."):
            result = predictor.predict(smiles_input)

        if not result["is_valid"]:
            st.error("❌ Invalid SMILES string. Please check your input.")
            return

        # Results display
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)

        prob = result["probability"]
        prediction = result["prediction"]
        risk = result["risk_level"]
        confidence = result["confidence"]
        ad_status = result["ad_status"]

        with col1:
            risk_color = "#FF6B6B" if risk == "High Risk" else ("#FFB347" if risk == "Moderate Risk" else "#00D4AA")
            st.markdown(f"""
            <div class="result-card">
                <p class="metric-label">Prediction</p>
                <p class="metric-value" style="color: {risk_color};">{prediction}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <p class="metric-label">Toxicity Probability</p>
                <p class="metric-value" style="color: {risk_color};">{prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # AD status colored
            ad_color = "#00D4AA" if ad_status == "In-Domain" else "#FFB347"
            st.markdown(f"""
            <div class="result-card">
                <p class="metric-label">Applicability Domain</p>
                <p class="metric-value" style="color: {ad_color}; font-size: 1.6rem;">{ad_status}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            conf_color = "#00D4AA" if confidence == "High" else ("#FFB347" if confidence == "Moderate" else "#FF6B6B")
            st.markdown(f"""
            <div class="result-card">
                <p class="metric-label">Confidence</p>
                <p class="metric-value" style="color: {conf_color}; font-size: 1.8rem;">{confidence}</p>
            </div>
            """, unsafe_allow_html=True)

        # Insights row
        st.markdown("#### 🧪 Medicinal Chemist Insights")
        i_col1, i_col2 = st.columns(2)
        
        insights = result.get("chem_insights", {})
        
        with i_col1:
            st.markdown(f"""
            <div class="insight-box">
                <div class="metric-label" style="margin-bottom:0.8rem;">Structural Alerts & Risks</div>
            """, unsafe_allow_html=True)
            
            if insights.get("alerts"):
                for alert in insights["alerts"]:
                    st.markdown(f'<div class="insight-alert">⚠️ {alert}</div>', unsafe_allow_html=True)
            else:
                st.success("No chemical structural alerts detected for this molecule.")
            st.markdown("</div>", unsafe_allow_html=True)

        with i_col2:
            st.markdown(f"""
            <div class="insight-box">
                <div class="metric-label" style="margin-bottom:0.8rem;">Physicochemical Profile</div>
            """, unsafe_allow_html=True)
            if insights.get("properties"):
                props = insights["properties"]
                p_cols = st.columns(2)
                for j, (k, v) in enumerate(props.items()):
                    with p_cols[j % 2]:
                        st.markdown(f"**{k}:** `{v}`")
            st.markdown("</div>", unsafe_allow_html=True)

        # Molecule visualization
        st.markdown("#### 🔍 Structural Visualization")
        col_mol, col_spacer = st.columns([1, 1])

        with col_mol:
            try:
                from src.explainability.shap_explain import render_molecule_with_highlights
                mol_img = render_molecule_with_highlights(result["smiles_canonical"], img_size=(450, 350))
                if mol_img:
                    st.image(mol_img, caption=f"Canonical SMILES: {result['smiles_canonical']}")
            except Exception:
                st.info(f"Canonical SMILES: `{result['smiles_canonical']}`")

        # SHAP Explanation
        st.markdown("---")
        st.markdown("#### 🔍 Model Explanation (SHAP)")

        with st.spinner("Computing SHAP explanation..."):
            try:
                explainer, feat_names = load_shap_explainer()
                if explainer is not None:
                    from src.explainability.shap_explain import (
                        explain_single_prediction,
                        map_fingerprint_bits_to_substructures,
                    )
                    from src.features.featurize import smiles_to_morgan_fingerprint
                    from src.features.descriptors import compute_single_descriptors

                    # Build feature vector
                    if FEATURE_MODE == "combined":
                        fp = smiles_to_morgan_fingerprint(result["smiles_canonical"])
                        desc = compute_single_descriptors(result["smiles_canonical"])
                        if fp is not None and desc is not None:
                            if predictor.scaler is not None:
                                desc = predictor.scaler.transform(desc.reshape(1, -1))[0]
                            X_single = np.concatenate([fp, desc]).reshape(1, -1).astype(np.float32)
                        else:
                            X_single = None
                    elif FEATURE_MODE == "fingerprint":
                        fp = smiles_to_morgan_fingerprint(result["smiles_canonical"])
                        X_single = fp.reshape(1, -1).astype(np.float32) if fp is not None else None
                    else:
                        X_single = None

                    if X_single is not None:
                        explanation = explain_single_prediction(
                            explainer, X_single,
                            feature_names=feat_names,
                            smiles=result["smiles_canonical"],
                            top_n=15,
                        )

                        if explanation and explanation.get("top_features"):
                            # Show top contributing features
                            top_feats = explanation["top_features"][:10]
                            feat_df = pd.DataFrame(top_feats)
                            feat_df = feat_df[["feature", "shap_value", "direction"]]
                            feat_df.columns = ["Feature", "SHAP Value", "Direction"]

                            st.dataframe(feat_df, use_container_width=True, hide_index=True)

                            # Show explanation image if saved
                            waterfall_path = EXPLANATIONS_DIR / "shap_waterfall.png"
                            if waterfall_path.exists():
                                st.image(str(waterfall_path), caption="SHAP Waterfall Plot")

                            st.caption(
                                "⚠️ **Interpretability note:** Morgan fingerprint bits are "
                                "hashed representations. Substructure mappings are approximate "
                                "due to potential bit collisions. Treat as indicative."
                            )
                else:
                    st.info(
                        "SHAP explainer not available. "
                        "Run the training pipeline to enable explanations."
                    )
            except Exception as e:
                st.info(f"Explanation unavailable: {str(e)[:200]}")

        # Download result
        st.markdown("---")
        result_json = json.dumps(result, indent=2, default=str)
        st.download_button(
            "📥 Download Prediction Report (JSON)",
            data=result_json,
            file_name=f"toxipred_result_{smiles_input[:20].replace(' ', '_')}.json",
            mime="application/json",
        )


def render_compare_tab():
    """Side-by-side molecule comparison interface."""
    st.markdown("### ⚖️ Molecular Comparison Mode")
    st.markdown("Compare two compounds to evaluate relative hepatotoxicity risk and structural alerts.")

    col1, col2 = st.columns(2)
    
    with col1:
        smi1 = st.text_input("Molecule A (SMILES)", "CC(=O)Nc1ccc(O)cc1", key="smi_a")
    with col2:
        smi2 = st.text_input("Molecule B (SMILES)", "CC(=O)Oc1ccccc1C(=O)O", key="smi_b")

    if st.button("🚀 Compare Candidates", use_container_width=True):
        predictor = load_predictor()
        if predictor:
            with st.spinner("Analyzing both candidates..."):
                res1 = predictor.predict(smi1)
                res2 = predictor.predict(smi2)

            if not res1["is_valid"] or not res2["is_valid"]:
                st.error("One or both SMILES inputs are invalid.")
                return

            c1, c2 = st.columns(2)
            
            for i, (res, col) in enumerate([(res1, c1), (res2, c2)]):
                with col:
                    # Winner highlighting (lower probability wins in toxicity prediction)
                    is_winner = (res1["probability"] < res2["probability"]) if i == 0 else (res2["probability"] < res1["probability"])
                    card_class = "compare-card compare-winner" if is_winner else "compare-card"
                    
                    risk_color = "#FF6B6B" if res["risk_level"] == "High Risk" else ("#FFB347" if res["risk_level"] == "Moderate Risk" else "#00D4AA")
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4 style="margin-top:0;">Candidate {'A' if i==0 else 'B'}</h4>
                        <p class="metric-label">Risk Assessment</p>
                        <h2 style="color:{risk_color}; margin-bottom:0;">{res['prediction']}</h2>
                        <p style="font-size:1.1rem; opacity:0.8;">{res['probability']:.1%} Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Structure visualization
                    try:
                        from src.explainability.shap_explain import render_molecule_with_highlights
                        mol_img = render_molecule_with_highlights(res["smiles_canonical"], img_size=(400, 300))
                        if mol_img:
                            st.image(mol_img)
                    except Exception:
                        st.code(res["smiles_canonical"], language=None)
                    
                    # AD & Insights
                    st.markdown(f"**Applicability Domain:** `{res['ad_status']}`")
                    
                    insights = res.get("chem_insights", {})
                    if insights.get("alerts"):
                        with st.expander(f"⚠️ {len(insights['alerts'])} Alerts Detected", expanded=True):
                            for alert in insights["alerts"]:
                                st.markdown(f"- {alert}")
                    else:
                        st.success("No chemical structural alerts detected.")

                    # Properties table
                    if insights.get("properties"):
                        props = insights["properties"]
                        st.markdown("---")
                        st.table(pd.DataFrame([props]).T.rename(columns={0: "Value"}))

def render_performance_tab():
    """Display model evaluation metrics and plots."""

    st.markdown("### 📊 Model Performance & Evaluation")
    st.markdown(
        "Below are the evaluation results from the test set holdout. "
        "Metrics reflect honest, unbiased performance estimation."
    )

    # Load metrics
    metrics = load_metrics()

    if not metrics:
        st.warning(
            "No evaluation metrics found. Run the training pipeline first:\n\n"
            "```bash\npython scripts/run_pipeline.py\n```"
        )
        return

    # Metrics cards
    if "test_metrics_optimal" in metrics:
        m = metrics["test_metrics_optimal"]
        st.markdown(f"**Using optimized threshold: {m.get('threshold', 0.5):.2f}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
        with col2:
            st.metric("PR-AUC", f"{m.get('pr_auc', 0):.3f}")
        with col3:
            st.metric("F1 Score", f"{m.get('f1_score', 0):.3f}")
        with col4:
            st.metric("MCC", f"{m.get('mcc', 0):.3f}")
        with col5:
            st.metric("Brier Score", f"{m.get('brier_score', 0):.3f}", help="Lower is better (0=perfect calibration)")

        col6, col7, col8, col9 = st.columns(4)
        with col6:
            st.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        with col7:
            st.metric("Precision", f"{m.get('precision', 0):.3f}")
        with col8:
            st.metric("Recall", f"{m.get('recall', 0):.3f}")
        with col9:
            st.metric("Balanced Acc", f"{m.get('balanced_accuracy', 0):.3f}")

    # Plots
    st.markdown("---")
    st.markdown("### Evaluation Plots")

    plot_files = {
        "ROC Curve": "roc_curve.png",
        "Precision-Recall Curve": "precision_recall_curve.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Calibration Curve": "calibration_curve.png",
        "Reliability Curve": "reliability_curve.png",
        "Model Comparison": "model_comparison.png",
    }

    # Show plots in a grid
    available_plots = {k: v for k, v in plot_files.items() if (PLOTS_DIR / v).exists()}

    if available_plots:
        cols = st.columns(2)
        for i, (title, filename) in enumerate(available_plots.items()):
            with cols[i % 2]:
                st.image(str(PLOTS_DIR / filename), caption=title, use_container_width=True)
    else:
        st.info("No plots found. Run the training pipeline to generate evaluation plots.")

    # SHAP plots
    st.markdown("---")
    st.markdown("### Feature Importance (SHAP)")

    shap_plots = {
        "Global SHAP Importance": "shap_global_importance.png",
        "Feature Importance Bar Chart": "shap_bar_importance.png",
    }

    available_shap = {k: v for k, v in shap_plots.items() if (PLOTS_DIR / v).exists()}

    if available_shap:
        cols = st.columns(2)
        for i, (title, filename) in enumerate(available_shap.items()):
            with cols[i % 2]:
                st.image(str(PLOTS_DIR / filename), caption=title, use_container_width=True)

    # Classification report
    report_path = METRICS_DIR / "classification_report.txt"
    if report_path.exists():
        with st.expander("📄 Full Classification Report"):
            with open(report_path) as f:
                st.text(f.read())

    # Model comparison table
    comparison_path = METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        with st.expander("📊 Model Comparison Table"):
            comparison_df = pd.read_csv(comparison_path)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# ==============================================================================
# Tab 3: Batch Predict
# ==============================================================================

def render_batch_tab():
    """Batch prediction from CSV upload."""

    st.markdown("### 📋 Batch Prediction")
    st.markdown(
        "Upload a CSV file containing SMILES strings to predict "
        "hepatotoxicity risk for multiple molecules at once."
    )

    st.markdown("""
    **CSV Format Requirements:**
    - Must contain a column named `smiles` (case-insensitive)
    - One SMILES string per row
    - Additional columns are preserved in the output
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="CSV file with a 'smiles' column",
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded:** {len(input_df)} molecules")

            # Find SMILES column
            smiles_col = None
            for col in input_df.columns:
                if col.lower() in ("smiles", "smi", "molecule", "drug"):
                    smiles_col = col
                    break

            if smiles_col is None:
                st.error("No SMILES column found. Expected column named 'smiles'.")
                return

            st.dataframe(input_df.head(), use_container_width=True, hide_index=True)

            if st.button("🚀 Run Batch Prediction", key="batch_btn"):
                predictor = load_predictor()
                if predictor is None:
                    return

                with st.spinner(f"Predicting {len(input_df)} molecules..."):
                    results_df = predictor.predict_batch(input_df[smiles_col].tolist())

                # Add back original columns
                other_cols = [c for c in input_df.columns if c != smiles_col]
                for col in other_cols:
                    results_df[col] = input_df[col].values

                # Display results
                st.markdown("### 📊 Prediction Results")
                st.info("💡 Results are automatically sorted by **Toxicity Probability** (Highest Risk First).")
                
                # Summary stats
                n_valid = results_df["is_valid"].sum()
                n_toxic = (results_df["prediction"] == "Toxic").sum()
                n_safe = (results_df["prediction"] == "Non-Toxic").sum()
                n_ood = (results_df["ad_status"] == "Out-of-Domain").sum()

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("Valid", int(n_valid))
                with col3:
                    st.metric("Predicted Toxic", int(n_toxic))
                with col4:
                    st.metric("Predicted Non-Toxic", int(n_safe))
                with col5:
                    st.metric("Out-of-Domain", int(n_ood), help="Molecules structural distant from training set")

                # Full results table
                st.dataframe(
                    results_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "probability": st.column_config.ProgressColumn("Risk Probability", format="%.3f", min_value=0, max_value=1),
                        "alerts": st.column_config.NumberColumn("Chem Alerts", format="%d ⚠️"),
                        "ad_status": "Domain Status"
                    }
                )

                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    data=csv,
                    file_name="toxipred_batch_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ==============================================================================
# Tab 4: About
# ==============================================================================

def render_about_tab():
    """Project information, methodology, and disclaimers."""

    st.markdown("### ℹ️ About ToxiPred")

    st.markdown(f"""
    <div class="main-header">
        <h1>🧬 {APP_TITLE}</h1>
        <p class="subtitle">{APP_SUBTITLE}</p>
        <p>{APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 Why This Matters

        Drug-Induced Liver Injury (DILI) is one of the leading causes of
        drug withdrawal from the market and clinical trial failures.
        Early identification of hepatotoxicity risk during compound screening
        can significantly reduce attrition rates and development costs.

        ToxiPred provides a computational triage layer that helps medicinal
        chemists prioritize compounds for further experimental testing,
        enabling more efficient drug discovery workflows.

        #### 🧪 Methodology

        1. **Data**: Merged DILI (TDC) + ClinTox (MoleculeNet) datasets
        2. **Features**: Morgan fingerprints (ECFP4) + molecular descriptors
        3. **Model**: XGBoost with calibrated probabilities
        4. **Validation**: Stratified 5-fold CV + independent test set
        5. **Explainability**: SHAP values with substructure mapping
        """)

    with col2:
        st.markdown("""
        #### 🛠️ Technical Stack

        | Component | Technology |
        |-----------|-----------|
        | Chemistry | RDKit |
        | ML Model | XGBoost |
        | Explainability | SHAP |
        | Featurization | Morgan FP + Descriptors |
        | Web UI | Streamlit |
        | Validation | scikit-learn |

        #### 📊 Model Features

        - **Morgan Fingerprints**: 2048-bit circular fingerprints (radius 2)
        - **Molecular Descriptors**: MolWt, LogP, HBD, HBA, TPSA, etc.
        - **Calibrated Probabilities**: Isotonic regression calibration
        - **Threshold Optimization**: F1-maximized decision boundary
        """)

    st.markdown("---")

    st.markdown("""
    #### ⚠️ Limitations & Responsible Use

    - **Not a clinical tool.** This model is for research and compound prioritization only.
    - **Dataset limitations:** Training data covers ~1,500–2,000 compounds. Chemical space coverage is limited.
    - **No causal claims.** The model identifies statistical associations, not causal mechanisms.
    - **Fingerprint interpretability is approximate.** Morgan bits are hashed, and substructure mappings may be ambiguous.
    - **Model performance is honest.** Metrics are reported without inflation. Limitations are documented.
    - **Always validate experimentally.** Computational predictions should complement, not replace, wet lab testing.

    #### 🔮 Future Improvements

    - Larger and more diverse training datasets
    - Graph neural network (GNN) molecular representations
    - Multi-task learning across ADMET endpoints
    - Attention-based substructure highlighting
    - Integration with chemical databases (ChEMBL, PubChem)
    """)

    st.markdown("---")

    st.markdown(f"""
    <div class="disclaimer">
        {APP_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Main App
# ==============================================================================

def main():
    """Main application entry point."""
    render_sidebar()

    # Tabs
    tab1, tab_comp, tab2, tab3, tab4 = st.tabs([
        "🔬 Predict",
        "⚖️ Compare Molecules",
        "📊 Model Performance",
        "📋 Batch Predict",
        "ℹ️ About",
    ])

    with tab1:
        render_predict_tab()

    with tab_comp:
        render_compare_tab()

    with tab2:
        render_performance_tab()

    with tab3:
        render_batch_tab()

    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
