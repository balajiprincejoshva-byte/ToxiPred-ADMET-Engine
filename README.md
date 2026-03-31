---

## ✨ Elite Features (Scientific Rigor Layer)

ToxiPred goes beyond simple classification by implementing industry-standard safety and reliability checks:

| Feature | Technical Implementation | Impact |
|---------|-------------------------|--------|
| 🛡️ **Applicability Domain** | `IsolationForest` outlier detection on 2048D feature space | Detects structurally novel molecules where model confidence is naturally lower. |
| 📊 **Confidence Calibration** | Isotonic Regression + Brier Score optimization | Transforms raw model scores into reliable probabilities (0.8 score = 80% real-world risk). |
| 🧪 **Med-Chem Insights** | Heuristic rule-based layer (Lipinski, Veber, LogP alerts) | Provides a "second opinion" based on classical medicinal chemistry rules. |
| ⚖️ **Comparison Mode** | Side-by-side SHAP analysis and risk profiling | Enables direct lead-optimization triage between two chemical candidates. |
| 🐚 **Professional CLI** | `toxipred` command-line entry point | Ready for integration into automated high-throughput screening pipelines. |

---

## 🏗️ Architecture

```
ToxiPred/
├── src/
│   ├── models/
│   │   ├── domain.py            # Applicability Domain (IsolationForest)
│   │   ├── predict.py           # Multi-layered inference engine
│   ├── features/
│   │   ├── chem_insights.py     # Medicinal chemistry heuristic alerts
│   ├── app/
│   │   ├── cli.py               # Professional CLI interface
│   │   └── streamlit_app.py     # Elite UI with Comparison & Calibration
...
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/ToxiPred.git
cd ToxiPred
pip install -e .
```

### CLI Usage

ToxiPred provides a professional terminal interface for high-throughput screening:

```bash
# Single molecule prediction
toxipred predict "CC(=O)Nc1ccc(O)cc1"

# Single molecule prediction (JSON output)
toxipred predict "CC(=O)Nc1ccc(O)cc1" --json

# Batch processing from CSV
toxipred batch library.csv results.csv --smiles-col smiles
```

### Run the Elite UI

```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 Methodology (Advanced)

### Reliability & Calibration
A model that is 80% "sure" should be right 80% of the time. ToxiPred optimizes for this using **Brier Score** minimization. The generated **Reliability Curve** (found in `artifacts/plots/`) proves the model's probabilistic accuracy.

### Applicability Domain (AD)
Machine learning models are only as good as the data they've seen. ToxiPred uses an **Isolation Forest** to define a "structural envelope" around the training set. If a molecule falls outside this envelope, it's flagged as **Out-of-Domain (OOD)**, warning the user that the prediction is an extrapolation.

### Medicinal Chemist Layer
The system supplements ML scores with hard-coded alerts:
- **Rule of 5 Violations**: MW, LogP, HBD/HBA checks.
- **Lipophilicity Stress**: Flagging LogP > 3 for metabolic risk.
- **Structural Alerts**: Detection of toxicophores like primary aromatic amines or nitro groups.

---

## 💡 Industry Use-Case

1. **High-Throughput Triage**: Use the CLI to screen 10,000+ compounds from an external vendor.
2. **Structural Optimization**: Use **Compare Mode** in the UI to see how a small structural change (e.g., adding a methyl group) shifts both the ML toxicity score and the medicinal chemistry alerts.
3. **Safety Reporting**: Download the JSON/CSV reports to document the safety profile of a lead candidate before experimental validation.

---

## 🛠️ Technical Highlights

- **Modular Design**: Decoupled featurization, domain analysis, and inference.
- **Scientific Rigor**: Focus on calibration and domain awareness over simple "accuracy".
- **UX Excellence**: Streamlit interface designed with a "Pharma-First" aesthetic.
- **Reproducibility**: `random_seed=42` fixed across all stochastic components (XGBoost, Splitting, IsolationForest).

---

## 📜 License

This project is released under the MIT License.

---

## 📝 Disclaimer

This software is provided for **research and educational purposes only**. It is not intended for clinical, regulatory, or patient-facing decision-making. Predictions should be treated as computational screening signals and must be validated through appropriate experimental assays and expert review.

---

<div align="center">

**Built with** ❤️ **using RDKit · XGBoost · SHAP · Streamlit**

*Designed for high-impact biotech portfolio presentation*

</div>
]]>
