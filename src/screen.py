import os
# Configure linear algebra environments to utilize all 11 CPU cores
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["MKL_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "11"
os.environ["VECLIB_MAXIMUM_THREADS"] = "11"
os.environ["NUMEXPR_NUM_THREADS"] = "11"

import torch
# Configure PyTorch to utilize all 11 CPU cores for accelerated virtual screening
torch.set_num_threads(11)
torch.set_num_interop_threads(11)

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import GPCRBiasedSignalingModel
from src.data_loader import smiles_to_graph_data, LocalProteinTokenizer, GPCR_SEQUENCES, BIAS_MAP

from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def compute_admet_properties(smiles):
    """Compute drug-likeness and ADMET properties using RDKit descriptors.
    
    Returns a dictionary with:
      - MolWt: Molecular weight
      - LogP: Octanol/water partition coefficient  
      - HBD: Hydrogen bond donors
      - HBA: Hydrogen bond acceptors
      - TPSA: Topological polar surface area
      - RotBonds: Rotatable bonds
      - QED_Score: Quantitative Estimate of Drug-likeness (0 to 1)
      - Lipinski_Pass: Whether all Lipinski Rule of Five criteria are met
      - BBB_Permeable: Estimated blood-brain barrier permeability (LogP 1-3, TPSA < 90, MW < 450)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MolWt": None, "LogP": None, "HBD": None, "HBA": None,
            "TPSA": None, "RotBonds": None, "QED_Score": None,
            "Lipinski_Pass": False, "BBB_Permeable": False
        }
    
    mw = round(Descriptors.MolWt(mol), 1)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 1)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    qed_score = round(QED.qed(mol), 3)
    
    # Lipinski Rule of Five
    lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    
    # BBB permeability estimate (simplified Clark/Pardridge model)
    # Criteria: MW < 450, TPSA < 90, 1 < LogP < 3
    bbb_permeable = (mw < 450 and tpsa < 90 and 1.0 < logp < 3.0)
    
    return {
        "MolWt": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "TPSA": tpsa,
        "RotBonds": rot_bonds,
        "QED_Score": qed_score,
        "Lipinski_Pass": lipinski_pass,
        "BBB_Permeable": bbb_permeable
    }

# Define screening compounds: A library of natural products, psychedelics, and neuro-therapeutic analogs
SCREENING_LIBRARY = [
    {"name": "Psilocybin", "smiles": "COP(=O)(O)OC1=CC=CC2=C1C(CCN(C)C)=CN2", "type": "Psychedelic Alkaloid"},
    {"name": "DMT", "smiles": "CN(C)CCC1=CNC2=CC=CC=C12", "type": "Tryptamine Derivative"},
    {"name": "Mescaline", "smiles": "COCc1cc(OC)c(OC)c(OC)c1", "type": "Phenethylamine"},
    {"name": "Lisuride", "smiles": "CCN(CC)C(=O)NC1CN(C)C2Cc3c[nH]c4ccc(C2=C1)c34", "type": "Ergoline Class"},
    {"name": "25I-NBOMe", "smiles": "COC1=C(C=C(C(=C1)I)CCNCC2=CC=CC=C2OC)OC", "type": "Phenethylamine Agonist"},
    {"name": "Loxapine", "smiles": "CN1CCN(CC1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl", "type": "Tricyclic Antipsychotic"},
    {"name": "Ketamine", "smiles": "CNC1(CCCCC1=O)C2=CC=CC=C2Cl", "type": "Aesthetic Dissociative"},
    {"name": "Bromocriptine", "smiles": "CC(C)C[C@H]1C(=O)N2CCC[C@H]2[C@]3(N1C(=O)[C@](O3)(C(C)C)NC(=O)[C@H]4CN([C@@H]5CC6=C(NC7=CC=CC(=C67)C5=C4)Br)C)O", "type": "Ergoline Agonist"},
    {"name": "Novel_Analog_X1", "smiles": "CCN(CC)C(=O)CC1=CNC2=C1C=C(OC)C=C2", "type": "Synthetic Tryptamine"},
    {"name": "Novel_Analog_Y2", "smiles": "CC(C)NCCC1=CNC2=C1C=C(O)C=C2", "type": "Synthetic Phenol-tryptamine"}
]

def run_virtual_screen(target_gpcr_key="5HT2A_HUMAN", max_seq_len=1024, max_atoms=64):
    print("==================================================")
    print(f" Virtual Screening: Target GPCR: {target_gpcr_key} ")
    print("==================================================")
    
    # 1. Check for model weights
    model_path = os.path.join(os.path.dirname(__file__), "..", "best_gpcr_bias_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Trained model checkpoint not found at {model_path}.")
        print("Please run train.py first to train the model weights.")
        return
        
    # 2. Instantiate and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPCRBiasedSignalingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Prepare the target protein sequence
    target_seq = GPCR_SEQUENCES.get(target_gpcr_key, GPCR_SEQUENCES["5HT2A_HUMAN"])
    
    # Simple tokenizer setup
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        use_esm_tokenizer = True
    except Exception:
        tokenizer = LocalProteinTokenizer()
        use_esm_tokenizer = False
        
    if use_esm_tokenizer:
        tokenized = tokenizer(target_seq, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
        seq_tokens = tokenized["input_ids"].to(device)
    else:
        seq_tokens = tokenizer.tokenize(target_seq, max_len=max_seq_len).unsqueeze(0).to(device)
        
    # 4. Perform Virtual Screen Inference
    results = []
    
    with torch.no_grad():
        for comp in SCREENING_LIBRARY:
            node_feats, adj_matrix = smiles_to_graph_data(comp["smiles"], max_atoms=max_atoms)
            node_feats = node_feats.unsqueeze(0).to(device)
            adj_matrix = adj_matrix.unsqueeze(0).to(device)
            
            # Predict
            pred_pkd, pred_bias = model(seq_tokens, node_feats, adj_matrix)
            
            # Convert probabilities
            probs = torch.softmax(pred_bias, dim=-1).squeeze(0).cpu().numpy()
            predicted_pkd = pred_pkd.item()
            
            # G-protein activation bias index
            prob_g_protein = probs[0]
            prob_balanced = probs[1]
            prob_arrestin = probs[2]
            
            # Calculate a compound scoring metric: G-protein selectivity and high affinity
            # Score = predicted pKd * G-protein probability
            selectivity_score = predicted_pkd * prob_g_protein
            
            # --- ADMET / Drug-likeness Calculations ---
            admet = compute_admet_properties(comp["smiles"])
            
            results.append({
                "Compound Name": comp["name"],
                "Chemical Category": comp["type"],
                "SMILES String": comp["smiles"],
                "Predicted pKd": round(predicted_pkd, 2),
                "Est. Kd (nM)": round(10**(9 - predicted_pkd), 2),
                "G-Protein Bias Prob": round(prob_g_protein * 100.0, 1),
                "Balanced Prob": round(prob_balanced * 100.0, 1),
                "β-Arrestin Bias Prob": round(prob_arrestin * 100.0, 1),
                "Selectivity Score": round(selectivity_score, 3),
                **admet
            })
            
    # 5. Format & Save Results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Selectivity Score", ascending=False).reset_index(drop=True)
    
    # Save CSV
    csv_out = os.path.join(os.path.dirname(__file__), "..", "virtual_screening_hits.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"Screening report successfully saved to: {csv_out}\n")
    
    # Print ranked output table in markdown
    print("Ranked Virtual Screening Hits (Optimized for G-protein Selectivity):")
    print(df_results[["Compound Name", "Chemical Category", "Predicted pKd", "Est. Kd (nM)", "G-Protein Bias Prob", "Selectivity Score"]].to_markdown(index=False))
    print("==================================================")
    
if __name__ == "__main__":
    run_virtual_screen()
