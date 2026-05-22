"""
Model Interpretability Module — Attention Extraction and Visualizer

Extracts GAT attention weights (which atoms are critical for ligand structure)
and Cross-Attention weights (which amino acid residues the ligand attends to
during docking/binding interaction).
"""

import os
import sys

# Configure linear algebra environments to utilize all 11 CPU cores
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["MKL_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "11"
os.environ["VECLIB_MAXIMUM_THREADS"] = "11"
os.environ["NUMEXPR_NUM_THREADS"] = "11"

import torch
# Configure PyTorch to utilize all 11 CPU cores
try:
    torch.set_num_threads(11)
    torch.set_num_interop_threads(11)
except RuntimeError:
    pass

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import GPCRBiasedSignalingModel
from src.data_loader import smiles_to_graph_data, LocalProteinTokenizer, GPCR_SEQUENCES

def get_attention_maps(model_path, smiles, gpcr_key, max_seq_len=1024, max_atoms=64):
    """Load model, run prediction, and extract all internal attention weights.
    
    Returns:
      - predicted_pkd: float
      - bias_probabilities: list of 3 floats
      - gat_attention: np.ndarray shape (num_nodes, num_nodes)
      - cross_attention: np.ndarray shape (num_nodes, seq_len)
      - atom_symbols: list of atom symbols present in the compound
    """
    # 1. Load Model
    model = GPCRBiasedSignalingModel(d_model=128)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # 2. Tokenize Protein
    tokenizer = LocalProteinTokenizer()
    seq = GPCR_SEQUENCES.get(gpcr_key, GPCR_SEQUENCES["5HT2A_HUMAN"])
    seq_tokens = tokenizer.tokenize(seq, max_len=max_seq_len)
    
    # 3. Featurize Ligand
    node_feats, adj_matrix = smiles_to_graph_data(smiles, max_atoms=max_atoms)
    
    # Get active atom count in SMILES to truncate padding
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = min(mol.GetNumAtoms(), max_atoms) if mol else 0
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()][:max_atoms] if mol else []
    
    # 4. Predict & Extract Attention
    # Add batch dimension (1, ...)
    seq_t = seq_tokens.unsqueeze(0)
    feats_t = node_feats.unsqueeze(0)
    adj_t = adj_matrix.unsqueeze(0)
    
    with torch.no_grad():
        pred_pkd, pred_bias, attn_dict = model(feats_t, feats_t, feats_t) # Placeholder call signature fallback
        # Let's use the actual forward pass arguments
        pred_pkd, pred_bias, attn_dict = model(seq_t, feats_t, adj_t, return_attention=True)

    # Extract GAT Layer 2 attention: Shape (batch, num_nodes, num_nodes) -> (num_nodes, num_nodes)
    gat_attn = attn_dict["gat_layer2"][0].cpu().numpy()
    
    # Extract Cross-Attention: Shape (batch, num_nodes, seq_len) -> (num_nodes, seq_len)
    cross_attn = attn_dict["cross_attention"][0].cpu().numpy()
    
    # Softmax probabilities
    bias_probs = torch.softmax(pred_bias, dim=-1)[0].cpu().numpy().tolist()
    
    # Truncate to actual atoms and non-padded protein sequence
    gat_attn_truncated = gat_attn[:num_atoms, :num_atoms]
    cross_attn_truncated = cross_attn[:num_atoms, :len(seq)]
    
    return {
        "pkd": float(pred_pkd[0].item()),
        "bias_probs": bias_probs,
        "gat_attn": gat_attn_truncated,
        "cross_attn": cross_attn_truncated,
        "atom_symbols": atom_symbols,
        "protein_seq": seq
    }

def plot_cross_attention(attn_data, output_path="results/cross_attention_map.png"):
    """Generate a high-fidelity visual plot mapping ligand atoms to top-attending GPCR residues."""
    cross_attn = attn_data["cross_attn"] # (num_atoms, seq_len)
    atoms = attn_data["atom_symbols"]
    seq = attn_data["protein_seq"]
    
    # Find the top 20 attending residues to keep the plot readable
    residue_importance = cross_attn.sum(axis=0) # Sum across all ligand atoms
    top_indices = np.argsort(residue_importance)[-25:] # Get indices of top 25 residues
    top_indices = sorted(top_indices) # Keep sequential order
    
    truncated_attn = cross_attn[:, top_indices]
    residue_labels = [f"{seq[i]}{i+1}" for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(truncated_attn, cmap="Blues", aspect="auto")
    
    # Set labels
    ax.set_yticks(np.arange(len(atoms)))
    ax.set_yticklabels(atoms, fontsize=10)
    ax.set_xticks(np.arange(len(residue_labels)))
    ax.set_xticklabels(residue_labels, rotation=45, ha="right", fontsize=9)
    
    # Styling
    ax.set_ylabel("Ligand Atoms", fontsize=12, fontweight="bold")
    ax.set_xlabel("GPCR Residues (Top 25 by Attention)", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Attention Binding Interface Map\n"
                 "(Highlighting which residues align to which ligand atoms)", fontsize=13, pad=15)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight Score", rotation=270, labelpad=15)
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Interpret] Cross-attention map successfully plotted: {output_path}")

if __name__ == "__main__":
    # Test stub to check structure compiles
    print("Interpretability module loaded successfully.")
