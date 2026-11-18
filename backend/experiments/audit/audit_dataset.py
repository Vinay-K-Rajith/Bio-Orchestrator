"""Is the CDK2 training set actually CDK2?

Pull every IC50 ChEMBL holds for CHEMBL301 (human CDK2) and join it against the
training CSV. Three questions:
  1. How many training compounds have NO CDK2 record at all?
  2. Where a record exists, does the training label match it?
  3. How many labels are censored (relation '>') but stored as exact?
"""
import sys, time
from pathlib import Path
import httpx, numpy as np, pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}
HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]

rows, offset = [], 0
with httpx.Client(headers=H, timeout=180.0) as c:
    while True:
        r = c.get(f"{BASE}/activity", params={
            "target_chembl_id": "CHEMBL301", "standard_type": "IC50",
            "format": "json", "limit": 1000, "offset": offset})
        r.raise_for_status()
        acts = r.json().get("activities", [])
        if not acts: break
        for a in acts:
            rows.append({"molecule_id": a.get("molecule_chembl_id"),
                         "value": a.get("standard_value"),
                         "units": a.get("standard_units"),
                         "relation": a.get("standard_relation")})
        offset += len(acts)
        print(f"  fetched {offset}", flush=True)
        if len(acts) < 1000: break

ref = pd.DataFrame(rows).dropna(subset=["molecule_id", "value"])
ref = ref[ref["units"] == "nM"].copy()
ref["value"] = ref["value"].astype(float)
ref.to_csv(HERE / "chembl_cdk2_truth.csv", index=False)
print(f"\nChEMBL CDK2 IC50 records: {len(ref)} over {ref.molecule_id.nunique()} molecules")
print(f"  censored (relation '>' or '>='): "
      f"{ref.relation.isin(['>', '>=']).sum()} "
      f"({100*ref.relation.isin(['>','>=']).mean():.1f}%)")

train = pd.read_csv(ROOT / "backend/data/cdk2_chembl_labeled.csv")
g = ref.groupby("molecule_id")
truth = g["value"].median()
rel_any_censored = g["relation"].apply(lambda s: bool(s.isin([">", ">="]).any()))

train["chembl_cdk2_median_nM"] = train["molecule_id"].map(truth)
train["chembl_censored"] = train["molecule_id"].map(rel_any_censored)
have = train["chembl_cdk2_median_nM"].notna()

print(f"\n--- training set audit (n={len(train)}) ---")
print(f"  molecules WITH a ChEMBL CDK2 IC50 record : {have.sum()} ({100*have.mean():.1f}%)")
print(f"  molecules with NO CDK2 record at all     : {(~have).sum()} ({100*(~have).mean():.1f}%)")

sub = train[have].copy()
sub["log_err"] = np.abs(np.log10(sub["IC50_nM"]) - np.log10(sub["chembl_cdk2_median_nM"]))
for tol, lbl in ((0.301, "within 2x"), (1.0, "within 10x")):
    print(f"  label {lbl} of ChEMBL CDK2 median      : "
          f"{(sub.log_err <= tol).sum()} / {len(sub)} ({100*(sub.log_err<=tol).mean():.1f}%)")
print(f"  label off by >10x from ChEMBL CDK2       : {(sub.log_err > 1.0).sum()}")

cens = train[have & train["chembl_censored"].fillna(False)]
print(f"\n  ChEMBL marks censored ('>') but CSV stores '=': {len(cens)} "
      f"({100*len(cens)/len(train):.1f}% of training set)")
print(f"    (HANDOFF's round-number heuristic estimated 22.6%)")

print("\n--- worst 15 label mismatches ---")
w = sub.nlargest(15, "log_err")[["molecule_id", "IC50_nM", "chembl_cdk2_median_nM", "log_err"]]
for _, r in w.iterrows():
    print(f"  {r.molecule_id:<16} csv={r.IC50_nM:>11.1f} nM   chembl={r.chembl_cdk2_median_nM:>11.1f} nM"
          f"   {r.log_err:.2f} log")
train.to_csv(HERE / "train_audited.csv", index=False)
