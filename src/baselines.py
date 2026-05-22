"""
Baseline Models & Rigorous Evaluation Module

Implements:
  1. Random Forest on Morgan fingerprints (ECFP4) for both affinity regression and bias classification
  2. XGBoost on Morgan fingerprints for comparison
  3. 5-Fold stratified cross-validation
  4. Comprehensive metrics: Pearson r, Spearman rho, RMSE, MAE, ROC-AUC, F1, confusion matrix
  5. Leave-one-target-out (LOTO) external validation
  6. Learning curve and scatter plot generation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, confusion_matrix, classification_report,
    accuracy_score
)
from rdkit import Chem
from rdkit.Chem import AllChem
from src.data_loader import robust_clean_smiles


# ─── Fingerprint Featurization ───────────────────────────────────────────────

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint (ECFP4) as a numpy array."""
    clean_s = robust_clean_smiles(smiles)
    mol = Chem.MolFromSmiles(clean_s) if clean_s else None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def prepare_fingerprint_dataset(df):
    """Convert a DataFrame with 'smiles' and 'pkd' columns into X, y arrays."""
    fps = []
    valid_indices = []
    for i, row in df.iterrows():
        fp = smiles_to_fingerprint(row["smiles"])
        if fp.sum() > 0:  # Valid molecule
            fps.append(fp)
            valid_indices.append(i)
    X = np.array(fps)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    return X, df_valid


# ─── Regression Metrics ──────────────────────────────────────────────────────

def compute_regression_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(pearson_r, 4),
        "Pearson_p": round(pearson_p, 6),
        "Spearman_rho": round(spearman_rho, 4),
        "Spearman_p": round(spearman_p, 6),
    }


# ─── Classification Metrics ─────────────────────────────────────────────────

BIAS_MAP = {"G-protein": 0, "Balanced": 1, "Arrestin": 2}
BIAS_NAMES = ["G-protein", "Balanced", "Arrestin"]

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics = {
        "Accuracy": round(acc, 4),
        "F1_macro": round(f1_macro, 4),
        "F1_weighted": round(f1_weighted, 4),
        "Confusion_matrix": cm.tolist(),
    }

    # ROC-AUC (only if probabilities are available and we have >1 class)
    if y_proba is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                metrics["ROC_AUC"] = round(roc_auc, 4)
        except Exception:
            pass

    return metrics


# ─── Baseline: Random Forest ─────────────────────────────────────────────────

def run_rf_baseline(df, results_dir="results"):
    """Run Random Forest baselines for both affinity and bias prediction."""
    print("\n" + "=" * 60)
    print("  Baseline: Random Forest on ECFP4 Morgan Fingerprints")
    print("=" * 60)

    X, df_valid = prepare_fingerprint_dataset(df)
    y_pkd = df_valid["pkd"].values.astype(np.float32)

    # --- Affinity Regression (5-fold CV) ---
    print("\n  [Regression] Binding Affinity (pKd) prediction...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_true_reg = []
    all_pred_reg = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_pkd[train_idx], y_pkd[val_idx]

        rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=11)
        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_val)

        all_true_reg.extend(y_val)
        all_pred_reg.extend(y_pred)

    reg_metrics = compute_regression_metrics(np.array(all_true_reg), np.array(all_pred_reg))
    print(f"    Pearson r = {reg_metrics['Pearson_r']}")
    print(f"    Spearman rho = {reg_metrics['Spearman_rho']}")
    print(f"    RMSE      = {reg_metrics['RMSE']}")
    print(f"    R2        = {reg_metrics['R2']}")

    # Scatter plot: predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true_reg, all_pred_reg, alpha=0.3, s=10, c="#4facfe", edgecolors="none")
    lims = [min(min(all_true_reg), min(all_pred_reg)) - 0.5,
            max(max(all_true_reg), max(all_pred_reg)) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
    ax.set_xlabel("Actual pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd", fontsize=12)
    ax.set_title(f"Random Forest - pKd Prediction (5-Fold CV)\n"
                 f"Pearson r = {reg_metrics['Pearson_r']}, RMSE = {reg_metrics['RMSE']}", fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    fig.tight_layout()
    scatter_path = os.path.join(results_dir, "rf_scatter_pkd.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"    Scatter plot saved: {scatter_path}")

    # --- Bias Classification (only on labeled data) ---
    df_labeled = df_valid[df_valid["bias"].isin(BIAS_MAP.keys())].reset_index(drop=True)
    clf_metrics = None

    if len(df_labeled) >= 20:
        print(f"\n  [Classification] Signaling Bias ({len(df_labeled)} labeled samples)...")
        X_lab, df_lab = prepare_fingerprint_dataset(df_labeled)
        y_bias = df_lab["bias"].map(BIAS_MAP).values

        skf = StratifiedKFold(n_splits=min(5, min(np.bincount(y_bias))), shuffle=True, random_state=42)

        all_true_clf = []
        all_pred_clf = []
        all_proba_clf = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_lab, y_bias), 1):
            X_train, X_val = X_lab[train_idx], X_lab[val_idx]
            y_train, y_val = y_bias[train_idx], y_bias[val_idx]

            rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=11)
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_val)
            y_proba = rf_clf.predict_proba(X_val)

            all_true_clf.extend(y_val)
            all_pred_clf.extend(y_pred)
            all_proba_clf.extend(y_proba)

        clf_metrics = compute_classification_metrics(
            np.array(all_true_clf), np.array(all_pred_clf),
            np.array(all_proba_clf) if all_proba_clf else None
        )
        print(f"    Accuracy  = {clf_metrics['Accuracy']}")
        print(f"    F1 (macro) = {clf_metrics['F1_macro']}")
        if "ROC_AUC" in clf_metrics:
            print(f"    ROC-AUC   = {clf_metrics['ROC_AUC']}")

        # Confusion matrix plot
        cm = np.array(clf_metrics["Confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set(xticks=np.arange(3), yticks=np.arange(3),
               xticklabels=BIAS_NAMES, yticklabels=BIAS_NAMES,
               ylabel='True Label', xlabel='Predicted Label',
               title='RF Bias Classification - Confusion Matrix')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        cm_path = os.path.join(results_dir, "rf_confusion_matrix.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"    Confusion matrix saved: {cm_path}")
    else:
        print(f"    [Warning] Only {len(df_labeled)} labeled samples - skipping classification baseline.")

    return {"regression": reg_metrics, "classification": clf_metrics}


# ─── XGBoost Baseline ────────────────────────────────────────────────────────

def run_xgb_baseline(df, results_dir="results"):
    """Run XGBoost baselines for affinity regression."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [Warning] XGBoost not installed. Skipping XGB baseline.")
        return None

    print("\n" + "=" * 60)
    print("  Baseline: XGBoost on ECFP4 Morgan Fingerprints")
    print("=" * 60)

    X, df_valid = prepare_fingerprint_dataset(df)
    y_pkd = df_valid["pkd"].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_pkd[train_idx], y_pkd[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", verbosity=0, n_jobs=11
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        all_true.extend(y_val)
        all_pred.extend(y_pred)

    metrics = compute_regression_metrics(np.array(all_true), np.array(all_pred))
    print(f"    Pearson r = {metrics['Pearson_r']}")
    print(f"    Spearman rho = {metrics['Spearman_rho']}")
    print(f"    RMSE      = {metrics['RMSE']}")
    print(f"    R2        = {metrics['R2']}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_pred, alpha=0.3, s=10, c="#ff6b6b", edgecolors="none")
    lims = [min(min(all_true), min(all_pred)) - 0.5,
            max(max(all_true), max(all_pred)) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
    ax.set_xlabel("Actual pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd", fontsize=12)
    ax.set_title(f"XGBoost - pKd Prediction (5-Fold CV)\n"
                 f"Pearson r = {metrics['Pearson_r']}, RMSE = {metrics['RMSE']}", fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    fig.tight_layout()
    scatter_path = os.path.join(results_dir, "xgb_scatter_pkd.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"    Scatter plot saved: {scatter_path}")

    return {"regression": metrics}


# ─── Leave-One-Target-Out (LOTO) Validation ──────────────────────────────────

def run_loto_validation(df, results_dir="results"):
    """Hold out each GPCR target entirely and evaluate generalization."""
    print("\n" + "=" * 60)
    print("  Leave-One-Target-Out (LOTO) External Validation")
    print("=" * 60)

    X, df_valid = prepare_fingerprint_dataset(df)
    y_pkd = df_valid["pkd"].values.astype(np.float32)

    targets = df_valid["gpcr"].unique()
    loto_results = {}

    for holdout_target in targets:
        mask = df_valid["gpcr"] == holdout_target
        train_idx = np.where(~mask)[0]
        test_idx = np.where(mask)[0]

        if len(test_idx) < 5:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_pkd[train_idx], y_pkd[test_idx]

        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=11)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = compute_regression_metrics(y_test, y_pred)
        loto_results[holdout_target] = metrics
        print(f"\n  Held out: {holdout_target} ({len(test_idx)} compounds)")
        print(f"    Pearson r = {metrics['Pearson_r']}, RMSE = {metrics['RMSE']}")

    # Summary comparison bar chart
    if loto_results:
        targets = list(loto_results.keys())
        pearson_vals = [loto_results[t]["Pearson_r"] for t in targets]
        rmse_vals = [loto_results[t]["RMSE"] for t in targets]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        colors = ["#4facfe", "#00f2fe", "#ff6b6b", "#ffc107"]

        ax1.bar(range(len(targets)), pearson_vals, color=colors[:len(targets)])
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels([t.split("_")[0] for t in targets], rotation=30)
        ax1.set_ylabel("Pearson r")
        ax1.set_title("LOTO: Cross-Target Pearson Correlation")
        ax1.set_ylim(0, 1)

        ax2.bar(range(len(targets)), rmse_vals, color=colors[:len(targets)])
        ax2.set_xticks(range(len(targets)))
        ax2.set_xticklabels([t.split("_")[0] for t in targets], rotation=30)
        ax2.set_ylabel("RMSE (pKd)")
        ax2.set_title("LOTO: Cross-Target RMSE")

        fig.tight_layout()
        loto_path = os.path.join(results_dir, "loto_validation.png")
        fig.savefig(loto_path, dpi=150)
        plt.close(fig)
        print(f"\n  LOTO bar chart saved: {loto_path}")

    return loto_results


# ─── Custom JSON Encoder for NumPy Types ─────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ─── Master Evaluation Runner ────────────────────────────────────────────────

def run_full_evaluation(data_path="data/chembl_gpcr_binding.csv", results_dir="results"):
    """Run all baselines and evaluations, save results to JSON."""
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} records from {data_path}")
    else:
        # Fallback to curated data
        from src.data_loader import LIGAND_DATA
        df = pd.DataFrame(LIGAND_DATA)
        print(f"  Using curated dataset ({len(df)} records). Run fetch_chembl.py first for full evaluation.")

    all_results = {}

    # Random Forest baseline
    rf_results = run_rf_baseline(df, results_dir)
    all_results["RandomForest"] = rf_results

    # XGBoost baseline
    xgb_results = run_xgb_baseline(df, results_dir)
    if xgb_results:
        all_results["XGBoost"] = xgb_results

    # Leave-one-target-out
    loto_results = run_loto_validation(df, results_dir)
    all_results["LOTO"] = loto_results

    # Save all results as JSON
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  All evaluation results saved to: {results_path}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY (pKd Regression)")
    print("=" * 60)
    print(f"  {'Model':<20} {'Pearson r':>10} {'Spearman rho':>14} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*58}")
    for model_name in ["RandomForest", "XGBoost"]:
        if model_name in all_results and all_results[model_name]:
            m = all_results[model_name]["regression"]
            print(f"  {model_name:<20} {m['Pearson_r']:>10} {m['Spearman_rho']:>12} {m['RMSE']:>8} {m['R2']:>8}")
    print(f"  {'GAT+ESM-2 (ours)':<20} {'TBD':>10} {'TBD':>12} {'TBD':>8} {'TBD':>8}")
    print(f"  {'-'*58}")
    print("  Note: GAT+ESM-2 metrics will be filled in after re-training on ChEMBL data.\n")

    return all_results


if __name__ == "__main__":
    run_full_evaluation()
