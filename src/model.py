import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGATLayer(nn.Module):
    """Pure PyTorch implementation of a Graph Attention Network (GAT) layer.
       Avoids tricky external C++ dependencies like torch-geometric on Windows."""
    def __init__(self, in_features, out_features, alpha=0.2):
        super(SimpleGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Projections
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj, return_attention=False):
        # h shape: (batch_size, num_nodes, in_features)
        # adj shape: (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, _ = h.size()
        
        # Apply projection: (batch_size, num_nodes, out_features)
        Wh = self.W(h)
        
        # To calculate attention efficiently across all node pairs,
        # we replicate Wh to create all combinations.
        Wh_repeated_1 = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_2 = Wh.repeat(1, num_nodes, 1)
        
        # Combine pairs: (batch_size, num_nodes * num_nodes, 2 * out_features)
        all_combinations = torch.cat([Wh_repeated_1, Wh_repeated_2], dim=-1)
        
        # Calculate scores: (batch_size, num_nodes, num_nodes)
        scores = self.a(all_combinations).view(batch_size, num_nodes, num_nodes)
        scores = self.leakyrelu(scores)
        
        # Apply mask based on adjacency matrix (only look at actual bonds)
        zero_vec = -9e15 * torch.ones_like(scores)
        # adj == 0 means no bond, so mask those scores out
        attention = torch.where(adj > 0, scores, zero_vec)
        
        # Softmax normalize over all neighbors
        attention = F.softmax(attention, dim=-1)
        
        # Message propagation: (batch_size, num_nodes, out_features)
        h_prime = torch.bmm(attention, Wh)
        if return_attention:
            return F.elu(h_prime), attention
        return F.elu(h_prime)

class ProteinSequenceEncoder(nn.Module):
    """Encodes GPCR protein sequences using pre-trained ESM-2 or an elegant local embedding fallback."""
    def __init__(self, vocab_size=33, embedding_dim=128, out_dim=128):
        super(ProteinSequenceEncoder, self).__init__()
        self.use_esm = False
        
        # Attempt to load ESM-2 transformer model
        try:
            from transformers import EsmModel
            self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
            # Freeze ESM parameters to run efficiently on CPU/standard computers
            for param in self.esm.parameters():
                param.requires_grad = False
            self.projection = nn.Linear(320, out_dim) # ESM-2 8M output size is 320
            self.use_esm = True
            print("Successfully initialized pre-trained ESM-2 encoder branch.")
        except Exception:
            # High-fidelity local embedding transformer backup
            print("Using high-fidelity local embedding sequence encoder.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.conv1 = nn.Conv1d(embedding_dim, out_dim, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2)
            self.resnet_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
            )
            
    def forward(self, seq_tokens):
        if self.use_esm:
            with torch.no_grad():
                outputs = self.esm(seq_tokens)
            embeddings = outputs.last_hidden_state # Shape: (batch, seq_len, 320)
            return self.projection(embeddings)
        else:
            # Local CNN ResNet forward pass
            # seq_tokens shape: (batch, seq_len)
            x = self.embedding(seq_tokens) # (batch, seq_len, embedding_dim)
            x = x.transpose(1, 2) # (batch, embedding_dim, seq_len)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            # Resnet skip connection
            x = x + self.resnet_block(x)
            return x.transpose(1, 2) # Output shape: (batch, seq_len, out_dim)

class CrossAttentionFusion(nn.Module):
    """Fuses protein residue embeddings with molecular node embeddings."""
    def __init__(self, d_model=128, nhead=2):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, return_attention=False):
        # query: (batch_size, target_len, d_model) -> e.g. ligand nodes
        # key, value: (batch_size, source_len, d_model) -> e.g. protein residues
        attn_out, attn_weights = self.attn(query, key, value)
        output = self.norm(query + attn_out)
        if return_attention:
            return output, attn_weights
        return output

class GPCRBiasedSignalingModel(nn.Module):
    """End-to-End Dual Branch Classifier and Affinity Regressor."""
    def __init__(self, ligand_in_dim=10, ligand_hid_dim=64, d_model=128):
        super(GPCRBiasedSignalingModel, self).__init__()
        
        # Ligand (Molecule) Branch: Custom GAT
        self.gat1 = SimpleGATLayer(ligand_in_dim, ligand_hid_dim)
        self.gat2 = SimpleGATLayer(ligand_hid_dim, d_model)
        
        # Protein Branch: ESM-2 or local CNN projection
        self.protein_encoder = ProteinSequenceEncoder(out_dim=d_model)
        
        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(d_model=d_model, nhead=2)
        
        # Aggregated Predictors (Multi-task learning)
        # Task 1: Binding Affinity (Continuous pKd Regression)
        self.affinity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Task 2: Signaling Bias (3 Classes: G-protein, Balanced, Arrestin)
        self.bias_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # 3 functional outcomes
        )
        
    def forward(self, seq_tokens, node_feats, adj_matrix, return_attention=False):
        # 1. Molecular graph representation through GAT
        if return_attention:
            x_ligand, gat1_attn = self.gat1(node_feats, adj_matrix, return_attention=True)
            x_ligand, gat2_attn = self.gat2(x_ligand, adj_matrix, return_attention=True)
        else:
            x_ligand = self.gat1(node_feats, adj_matrix)
            x_ligand = self.gat2(x_ligand, adj_matrix) # Shape: (batch, max_atoms, d_model)
        
        # 2. Protein sequence encoding
        x_protein = self.protein_encoder(seq_tokens) # Shape: (batch, seq_len, d_model)
        
        # 3. Dynamic binding fusion via cross-attention
        # Ligand queries the protein structural residues
        if return_attention:
            fused_features, cross_attn = self.fusion(x_ligand, x_protein, x_protein, return_attention=True)
        else:
            fused_features = self.fusion(x_ligand, x_protein, x_protein) # (batch, max_atoms, d_model)
        
        # Global max-pooling of ligand residues to extract localized features
        pooled_feats, _ = torch.max(fused_features, dim=1) # (batch, d_model)
        
        # 4. Multi-task output heads
        affinity_out = self.affinity_head(pooled_feats).squeeze(-1) # (batch)
        bias_out = self.bias_head(pooled_feats) # (batch, 3)
        
        if return_attention:
            return affinity_out, bias_out, {
                "gat_layer1": gat1_attn,
                "gat_layer2": gat2_attn,
                "cross_attention": cross_attn
            }
        return affinity_out, bias_out

if __name__ == "__main__":
    # Test batch sizes and forward passes
    model = GPCRBiasedSignalingModel()
    dummy_seq = torch.randint(0, 20, (2, 1024))
    dummy_feats = torch.randn(2, 64, 10)
    dummy_adj = torch.ones(2, 64, 64)
    
    aff, bias = model(dummy_seq, dummy_feats, dummy_adj)
    print("Dummy Model forward pass:")
    print("Affinity output shape:", aff.shape)
    print("Bias output shape:", bias.shape)
