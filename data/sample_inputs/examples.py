"""
ToxiPred — Sample Input Data

Example SMILES strings for testing, demos, and documentation.
Covers a range of drug-like molecules with known toxicity profiles.
"""

# Well-known drugs for demo purposes
SAMPLE_SMILES = {
    # --- Known hepatotoxic drugs ---
    "Acetaminophen (Tylenol)": "CC(=O)Nc1ccc(O)cc1",
    "Isoniazid (TB drug)": "NNC(=O)c1ccncc1",
    "Valproic Acid (anticonvulsant)": "CCCC(CCC)C(=O)O",
    "Amiodarone (antiarrhythmic)": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
    "Diclofenac (NSAID)": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",

    # --- Generally considered safer drugs ---
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Lisinopril": "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
}

# CSV-like batch input for testing
BATCH_SMILES_CSV = """smiles,name
CC(=O)Nc1ccc(O)cc1,Acetaminophen
CC(=O)Oc1ccccc1C(=O)O,Aspirin
Cn1c(=O)c2c(ncn2C)n(C)c1=O,Caffeine
CC(C)Cc1ccc(cc1)C(C)C(=O)O,Ibuprofen
NNC(=O)c1ccncc1,Isoniazid
OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl,Diclofenac
INVALID_SMILES,BadMolecule
,EmptySmiles
"""
