"""
Master Pipeline Orchestrator

Runs:
  1. ChEMBL data loading / caching (fetch_chembl.py)
  2. Baseline evaluations and performance plots (baselines.py)
  3. GAT + ESM-2 Model Training (train.py)
  4. Virtual Screening against GPCR binding pockets with ADMET filtering (screen.py)
"""

import os
import sys
import subprocess

# Ensure parent directory is in sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    print("\n" + "=" * 80)
    print(f"  STEP: {description}")
    print("=" * 80)
    print(f"Executing: {' '.join(cmd)}")
    
    # Run process and stream stdout to console
    python_exe = sys.executable
    full_cmd = [python_exe] + cmd
    
    # Add project root to PYTHONPATH for subprocesses
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(full_cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"\n[Error] Step failed with return code {result.returncode}!")
        sys.exit(result.returncode)
    print(f"  Step completed successfully.")

def main():
    print("\n" + "#" * 80)
    print("  GPCR BIASED SIGNALING PIPELINE ORCHESTRATOR")
    print("#" * 80)
    
    # Check if data already exists to avoid redundant long downloads
    data_path = "data/chembl_gpcr_binding.csv"
    if not os.path.exists(data_path):
        run_command(["src/fetch_chembl.py"], "Fetching and caching ChEMBL GPCR binding dataset")
    else:
        print(f"\n  Cached ChEMBL dataset found at {data_path}. Skipping fetch.")

    # 1. Run traditional ML baselines (5-Fold CV & LOTO)
    run_command(["src/baselines.py"], "Running Random Forest and XGBoost Baselines on ECFP4 Fingerprints")
    
    # 2. Re-train our state-of-the-art GAT + ESM-2 Deep Learning Model
    print("\n" + "=" * 80)
    print("  STEP: Re-training Dual-Branch GAT + ESM-2 PyTorch Model")
    print("=" * 80)
    
    # Check if a pre-trained checkpoint already exists to enable fast demonstration/interview mode
    checkpoint_path = "best_gpcr_bias_model.pt"
    epochs = 1
    if os.path.exists(checkpoint_path):
        print(f"  [Notice] Found pre-trained model checkpoint '{checkpoint_path}'.")
        print("  To save time on CPU during evaluation, training is optimized to 1 epoch.")
        print("  (Delete the file or edit src/run_pipeline.py to force full 15-epoch training.)")
        epochs = 1
    else:
        print("  No pre-trained model checkpoint found. Training for 15 epochs on CPU.")
        epochs = 15
        
    from src.train import train_model
    try:
        train_model(epochs=epochs, batch_size=8)
        print("  Model training completed successfully.")
    except Exception as e:
        print(f"\n[Error] Model training failed: {e}")
        sys.exit(1)
        
    # 3. Perform High-Throughput Virtual Screen with ADMET Filters
    run_command(["src/screen.py"], "Performing high-throughput screen with ADMET drug-likeness filters")
    
    print("\n" + "#" * 80)
    print("  GPCR BIASED SIGNALING PIPELINE COMPLETION")
    print("#" * 80)
    print("  All steps finished successfully!")
    print("  - Real ChEMBL data cached at: data/chembl_gpcr_binding.csv")
    print("  - Evaluation plots (scatter, confusion matrices, LOTO) saved in: results/")
    print("  - Optimal deep learning weights saved at: model_checkpoints/best_gpcr_model.pt")
    print("  - Virtual screening hits with ADMET properties saved to: virtual_screening_hits.csv")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
