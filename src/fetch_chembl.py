"""
ChEMBL Data Fetcher — Pulls real GPCR binding affinity data from the ChEMBL database.

Targets:
  - 5-HT2A (HTR2A)    -> CHEMBL224
  - Mu-opioid (OPRM1)  -> CHEMBL233
  - Dopamine D2 (DRD2) -> CHEMBL217
  - Beta-2 (ADRB2)     -> CHEMBL210

Fetches Ki/Kd measurements, converts to pKi/pKd, and assigns functional bias
labels from published literature where available.
"""

import os
import sys
import math
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ChEMBL Target IDs for our 4 Class A GPCRs
GPCR_TARGETS = {
    "5HT2A_HUMAN": {"chembl_id": "CHEMBL224", "name": "Serotonin 5-HT2A receptor"},
    "OPRM1_HUMAN": {"chembl_id": "CHEMBL233", "name": "Mu-opioid receptor"},
    "DRD2_HUMAN":  {"chembl_id": "CHEMBL217", "name": "Dopamine D2 receptor"},
    "ADRB2_HUMAN": {"chembl_id": "CHEMBL210", "name": "Beta-2 adrenergic receptor"},
}

# Known biased ligands from published BRET/TANGO assay literature
# Sources: Kroeze et al. 2015, Olsen et al. 2020, Schmid et al. 2017
KNOWN_BIAS_LABELS = {
    # 5-HT2A
    "LSD":           "G-protein",
    "LISURIDE":      "G-protein",
    "ERGOTAMINE":    "Arrestin",
    "SEROTONIN":     "Balanced",
    "5-HT":          "Balanced",
    "DOI":           "Balanced",
    "RISPERIDONE":   "Arrestin",
    "CLOZAPINE":     "Balanced",
    "PSILOCIN":      "G-protein",
    "DMT":           "G-protein",
    "MESCALINE":     "G-protein",
    # Mu-opioid
    "MORPHINE":      "Balanced",
    "FENTANYL":      "Arrestin",
    "OLICERIDINE":   "G-protein",
    "TRV130":        "G-protein",
    "BUPRENORPHINE": "G-protein",
    "NALOXONE":      "Balanced",
    "METHADONE":     "Arrestin",
    "CARFENTANIL":   "Arrestin",
    "SUFENTANIL":    "Arrestin",
    "OXYCODONE":     "Balanced",
    # DRD2
    "DOPAMINE":      "Balanced",
    "HALOPERIDOL":   "Arrestin",
    "ARIPIPRAZOLE":  "G-protein",
    "CABERGOLINE":   "G-protein",
    "BROMOCRIPTINE": "G-protein",
    "QUINPIROLE":    "Balanced",
    "ROPINIROLE":    "G-protein",
    "PRAMIPEXOLE":   "G-protein",
    "RISPERIDONE":   "Arrestin",
    "CHLORPROMAZINE":"Arrestin",
    # ADRB2
    "EPINEPHRINE":   "Balanced",
    "ADRENALINE":    "Balanced",
    "ISOPRENALINE":  "Balanced",
    "ISOPROTERENOL": "Balanced",
    "SALBUTAMOL":    "G-protein",
    "ALBUTEROL":     "G-protein",
    "SALMETEROL":    "G-protein",
    "FORMOTEROL":    "Balanced",
    "CARVEDILOL":    "Arrestin",
    "PROPRANOLOL":   "Arrestin",
    "ALPRENOLOL":    "Arrestin",
}


def fetch_chembl_data(target_key, target_info, max_records=2000):
    """Fetch binding affinity data from ChEMBL for a single GPCR target."""
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    chembl_id = target_info["chembl_id"]

    print(f"  Fetching data for {target_info['name']} ({chembl_id})...")

    # Query for Ki and Kd activities with nM units
    results = activity.filter(
        target_chembl_id=chembl_id,
        standard_type__in=["Ki", "Kd", "IC50"],
        standard_units="nM",
        standard_relation="=",
        assay_type="B"  # Binding assays only
    ).only([
        'molecule_chembl_id', 'canonical_smiles', 'molecule_pref_name',
        'standard_type', 'standard_value', 'standard_units',
        'target_chembl_id', 'assay_type'
    ])

    records = []
    count = 0

    for r in results:
        if count >= max_records:
            break

        smiles = r.get('canonical_smiles')
        value = r.get('standard_value')
        name = r.get('molecule_pref_name') or r.get('molecule_chembl_id')

        if not smiles or not value:
            continue

        try:
            value_float = float(value)
            if value_float <= 0:
                continue
            # Convert nM to pKi/pKd: pKi = -log10(Ki_in_M) = 9 - log10(Ki_in_nM)
            pkd = 9.0 - math.log10(value_float)
            if pkd < 3.0 or pkd > 12.0:  # Filter unrealistic values
                continue
        except (ValueError, TypeError):
            continue

        # Assign bias label from known literature, default to "Unknown"
        name_upper = (name or "").upper().strip()
        bias = KNOWN_BIAS_LABELS.get(name_upper, "Unknown")

        records.append({
            "name": name or r.get('molecule_chembl_id'),
            "smiles": smiles,
            "gpcr": target_key,
            "pkd": round(pkd, 2),
            "bias": bias,
            "source": "ChEMBL",
            "measurement_type": r.get('standard_type'),
        })
        count += 1

    print(f"    -> Retrieved {len(records)} valid binding records.")
    return records


def fetch_all_targets(output_path="data/chembl_gpcr_binding.csv", max_per_target=400):
    """Fetch binding data for all 4 GPCR targets and save to CSV."""
    print("=" * 60)
    print("  ChEMBL GPCR Binding Data Fetcher")
    print("=" * 60)

    all_records = []

    for target_key, target_info in GPCR_TARGETS.items():
        try:
            records = fetch_chembl_data(target_key, target_info, max_records=max_per_target)
            all_records.extend(records)
        except Exception as e:
            print(f"    [Warning] Error fetching {target_key}: {e}")
            continue

    if not all_records:
        print("Error: No records fetched from ChEMBL. Check internet connection.")
        return None

    df = pd.DataFrame(all_records)

    # Remove exact duplicate SMILES per target
    df = df.drop_duplicates(subset=["smiles", "gpcr"], keep="first")

    # For compounds with multiple measurements, take the median pKd
    df_agg = df.groupby(["smiles", "gpcr", "name", "bias", "source"]).agg(
        pkd=("pkd", "median"),
    ).reset_index()

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_agg.to_csv(output_path, index=False)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"  Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total unique ligand-target pairs: {len(df_agg)}")
    for gpcr in GPCR_TARGETS:
        subset = df_agg[df_agg["gpcr"] == gpcr]
        labeled = subset[subset["bias"] != "Unknown"]
        print(f"  {gpcr}: {len(subset)} compounds ({len(labeled)} with bias labels)")
    print(f"\n  Saved to: {os.path.abspath(output_path)}")
    print(f"{'=' * 60}")

    return df_agg


if __name__ == "__main__":
    df = fetch_all_targets()
    if df is not None:
        print(f"\nSample entries:")
        print(df.head(10).to_string(index=False))
