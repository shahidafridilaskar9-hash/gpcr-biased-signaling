import os
# Configure linear algebra environments to utilize all 11 CPU cores
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["MKL_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "11"
os.environ["VECLIB_MAXIMUM_THREADS"] = "11"
os.environ["NUMEXPR_NUM_THREADS"] = "11"

import torch
# Configure PyTorch to utilize all 11 CPU cores for accelerated training
try:
    torch.set_num_threads(11)
    torch.set_num_interop_threads(11)
except RuntimeError:
    pass

import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add parent directory to path to allow import of model and data loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import GPCRBiasedSignalingModel
from src.data_loader import get_dataloaders

def train_model(epochs=15, batch_size=4, lr=1e-3, weight_decay=1e-4):
    print("==================================================")
    print("   Starting Class A GPCR Multi-Task Model Training ")
    print("==================================================")
    
    # 1. Fetch data loaders
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, split_ratio=0.8)
    print(f"Dataset summary: Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    
    # 2. Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target execution hardware device: {device}")
    model = GPCRBiasedSignalingModel().to(device)
    
    # 3. Setup multi-task criteria & optimizers
    # Huber loss is more robust to outliers in binding affinity
    reg_criterion = nn.HuberLoss() 
    clf_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(os.path.dirname(__file__), "..", "best_gpcr_bias_model.pt")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_clf_loss = 0.0
        
        for batch in train_loader:
            seq = batch["seq_tokens"].to(device)
            feats = batch["node_feats"].to(device)
            adj = batch["adj_matrix"].to(device)
            targets_pkd = batch["pkd"].to(device)
            targets_bias = batch["bias_label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_pkd, pred_bias = model(seq, feats, adj)
            
            # Compute losses
            loss_reg = reg_criterion(pred_pkd, targets_pkd)
            loss_clf = clf_criterion(pred_bias, targets_bias)
            
            # Multi-task weight balancing: equal weight for stability
            total_loss = loss_reg + loss_clf
            
            # Backward propagation
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * seq.size(0)
            train_reg_loss += loss_reg.item() * seq.size(0)
            train_clf_loss += loss_clf.item() * seq.size(0)
            
        # Calculate epoch averages
        num_train = len(train_loader.dataset)
        train_loss /= num_train
        train_reg_loss /= num_train
        train_clf_loss /= num_train
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        val_clf_loss = 0.0
        correct_bias = 0
        total_bias = 0
        
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq_tokens"].to(device)
                feats = batch["node_feats"].to(device)
                adj = batch["adj_matrix"].to(device)
                targets_pkd = batch["pkd"].to(device)
                targets_bias = batch["bias_label"].to(device)
                
                pred_pkd, pred_bias = model(seq, feats, adj)
                
                loss_reg = reg_criterion(pred_pkd, targets_pkd)
                loss_clf = clf_criterion(pred_bias, targets_bias)
                total_loss = loss_reg + loss_clf
                
                val_loss += total_loss.item() * seq.size(0)
                val_reg_loss += loss_reg.item() * seq.size(0)
                val_clf_loss += loss_clf.item() * seq.size(0)
                
                # Accuracy tracking
                _, predicted = torch.max(pred_bias, 1)
                correct_bias += (predicted == targets_bias).sum().item()
                total_bias += seq.size(0)
                
        num_val = len(val_loader.dataset)
        val_loss /= num_val
        val_reg_loss /= num_val
        val_clf_loss /= num_val
        bias_accuracy = (correct_bias / total_bias) * 100.0 if total_bias > 0 else 0.0
        
        scheduler.step(val_loss)
        
        # Output progress log
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} (Reg: {train_reg_loss:.3f}, Clf: {train_clf_loss:.3f}) | "
              f"Val Loss: {val_loss:.4f} (Reg: {val_reg_loss:.3f}, Clf: {val_clf_loss:.3f}) | "
              f"Bias Accuracy: {bias_accuracy:.1f}%")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved optimal model checkpoint (Val Loss: {val_loss:.4f})")
            
    print("\nTraining completed successfully! Model checkpoint saved.")
    print("==================================================")

if __name__ == "__main__":
    train_model(epochs=12, batch_size=4)
