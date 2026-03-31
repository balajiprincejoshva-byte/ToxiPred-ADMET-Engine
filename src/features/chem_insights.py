"""
ToxiPred — Medicinal Chemist Insight Layer

Provides heuristic-driven structural alerts and physicochemical analysis
to supplement machine learning predictions. Based on established 
medicinal chemistry rules (e.g., Lipinski, Veber).
"""

from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments

def get_medicinal_chemistry_insights(smiles: str) -> Dict[str, Any]:
    """
    Analyze a molecule for common ADMET alerts and properties.
    
    Args:
        smiles: Input SMILES string.
        
    Returns:
        Dictionary of insights, alerts, and calculated properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    
    # Core Properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Flags & Alerts
    alerts = []
    
    # Lipinski's Rule of 5 violations
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    if violations > 1:
        alerts.append(f"Lipinski Rule of 5: {violations} violations (Potential poor oral bioavailability)")
        
    # Physicochemical Warnings
    if logp > 3:
        alerts.append("High Lipophilicity: LogP > 3 (Commonly associated with metabolic instability and off-target toxicity)")
    
    if tpsa > 140:
        alerts.append("High TPSA: > 140 Å² (Likely poor membrane permeability)")
    elif tpsa < 20:
        alerts.append("Low TPSA: < 20 Å² (May correlate with high blood-brain barrier penetration or non-specific binding)")

    if rot_bonds > 10:
        alerts.append("High Flexibility: > 10 rotatable bonds (Veber's Rule: may impact oral bioavailability)")

    # Structural Alerts (Simplified Pains-like or common toxicophores)
    # Checking for specific groups like Nitro, Aniline, etc. (indicative example)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N;+1](=[O])O")):
        alerts.append("Structural Alert: Nitro group detected (Potential mutagenicity/toxicity risk)")
    
    if mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1N")):
        alerts.append("Structural Alert: Primary aromatic amine detected (Potential metabolic activation to reactive intermediates)")

    return {
        "properties": {
            "Molecular Weight": round(mw, 2),
            "LogP": round(logp, 2),
            "H-Bond Donors": hbd,
            "H-Bond Acceptors": hba,
            "TPSA": round(tpsa, 2),
            "Rotatable Bonds": rot_bonds
        },
        "violations": violations,
        "alerts": alerts,
        "summary": "Medicinal chemistry profile looks standard." if not alerts else f"Detected {len(alerts)} medicinal chemistry alerts."
    }
