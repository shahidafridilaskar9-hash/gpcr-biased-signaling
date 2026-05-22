import torch
import pytest
import os
import sys

# Append parent directories to allow local module execution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import GPCRBiasedSignalingModel, SimpleGATLayer
from src.data_loader import LocalProteinTokenizer, smiles_to_graph_data

def test_local_tokenizer():
    tokenizer = LocalProteinTokenizer()
    seq = "MDILCEENTSLSSTTNSLMQLGD"
    tokens = tokenizer.tokenize(seq, max_len=128)
    assert tokens.shape == (128,)
    assert tokens[0] == tokenizer.vocab['<cls>']
    assert tokens[-1] == tokenizer.vocab['<pad>']

def test_smiles_to_graph():
    # Benzene ring SMILES
    smiles = "c1ccccc1"
    node_feats, adj_matrix = smiles_to_graph_data(smiles, max_atoms=32)
    assert node_feats.shape == (32, 10)
    assert adj_matrix.shape == (32, 32)
    # Check that connected carbon atoms have bond representations
    assert adj_matrix.sum() > 0

def test_gat_layer():
    batch_size = 3
    num_nodes = 16
    in_feats = 8
    out_feats = 16
    
    layer = SimpleGATLayer(in_features=in_feats, out_features=out_feats)
    h = torch.randn(batch_size, num_nodes, in_feats)
    # Fully connected adjacency with ones
    adj = torch.ones(batch_size, num_nodes, num_nodes)
    
    out = layer(h, adj)
    assert out.shape == (batch_size, num_nodes, out_feats)

def test_model_forward():
    model = GPCRBiasedSignalingModel()
    # Batch size = 2, Max sequence length = 512, Max atoms = 32
    seq = torch.randint(0, 20, (2, 512))
    feats = torch.randn(2, 32, 10)
    adj = torch.ones(2, 32, 32)
    
    aff, bias = model(seq, feats, adj)
    assert aff.shape == (2,)
    assert bias.shape == (2, 3)

def test_gradient_step():
    """Verify that model parameters update successfully on backward propagation."""
    model = GPCRBiasedSignalingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    seq = torch.randint(0, 20, (1, 512))
    feats = torch.randn(1, 32, 10)
    adj = torch.ones(1, 32, 32)
    
    aff, bias = model(seq, feats, adj)
    
    target_aff = torch.tensor([8.0], dtype=torch.float32)
    target_bias = torch.tensor([0], dtype=torch.long) # G-protein bias
    
    # Compute multi-task loss
    loss_aff = torch.nn.functional.mse_loss(aff, target_aff)
    loss_bias = torch.nn.functional.cross_entropy(bias, target_bias)
    total_loss = loss_aff + loss_bias
    
    # Backward pass
    total_loss.backward()
    
    # Check that gradients are computed for parameters that are not frozen
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    assert has_grad, "Model must compute gradients for learnable layers."
    
    # Take step
    optimizer.step()

def test_robust_clean_smiles():
    from src.data_loader import robust_clean_smiles
    # Test curly quotes removal
    assert robust_clean_smiles("“CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34”") == "CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34"
    # Test spaces/tabs and column parsing (first column is smiles)
    assert robust_clean_smiles("CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34\tLSD") == "CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34"
    assert robust_clean_smiles("  CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34   ") == "CCN(CC)C(=O)[C@@H]1CN(C)[C@@H]2Cc3c[nH]c4ccc(C2=C1)c34"
    # Test prefixes
    assert robust_clean_smiles("SMILES: c1ccccc1") == "c1ccccc1"
    assert robust_clean_smiles("smiles=c1ccccc1") == "c1ccccc1"
    # Test null/NaN safety
    assert robust_clean_smiles(None) == ""

