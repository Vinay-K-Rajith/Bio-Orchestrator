"""Decisive test: is cdk2_chembl_labeled.csv actually EGFR (CHEMBL203) data?

Same join as the CDK2 audit, against EGFR. If coverage is high AND the stored
IC50 values match EGFR's records, the file is mislabelled at the target level.
"""
from pathlib import Path
import httpx, numpy as np, pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}
HERE = Path(__file__).parent; ROOT = Path(__file__).resolve().parents[3]
train = pd.read_csv(ROOT / "backend/data/cdk2_chembl_labeled.csv")

rows, off = [], 0
with httpx.Client(headers=H, timeout=180.0) as c:
    while True:
        r = c.get(f"{BASE}/activity", params={"target_chembl_id": "CHEMBL203",
                  "standard_type": "IC50", "format": "json", "limit": 1000, "offset": off})
        r.raise_for_status()
        a = r.json().get("activities", [])
        if not a: break
        rows += [{"molecule_id": x.get("molecule_chembl_id"), "value": x.get("standard_value"),
                  "units": x.get("standard_units"), "relation": x.get("standard_relation")}
                 for x in a]
        off += len(a)
        if off % 5000 == 0: print(f"  fetched {off}", flush=True)
        if len(a) < 1000: break

ref = pd.DataFrame(rows).dropna(subset=["molecule_id", "value"])
ref = ref[ref["units"] == "nM"].copy(); ref["value"] = ref["value"].astype(float)
print(f"\nEGFR (CHEMBL203) IC50 records: {len(ref)} over {ref.molecule_id.nunique()} molecules")

g = ref.groupby("molecule_id")
train["egfr_median_nM"] = train["molecule_id"].map(g["value"].median())
train["egfr_censored"]  = train["molecule_id"].map(
    g["relation"].apply(lambda s: bool(s.isin([">", ">="]).any())))
have = train["egfr_median_nM"].notna()

print(f"\n--- training set (n={len(train)}) vs EGFR ---")
print(f"  covered by EGFR IC50 records : {have.sum()} ({100*have.mean():.1f}%)")
sub = train[have].copy()
sub["log_err"] = np.abs(np.log10(sub["IC50_nM"]) - np.log10(sub["egfr_median_nM"]))
for tol, lbl in ((0.0001, "EXACT"), (0.301, "within 2x"), (1.0, "within 10x")):
    print(f"  label {lbl:<10} of EGFR median   : {(sub.log_err<=tol).sum()} / {len(sub)} "
          f"({100*(sub.log_err<=tol).mean():.1f}%)")

print("\n--- side by side: CDK2 vs EGFR as the true target ---")
print(f"  {'':<26}{'CDK2 (CHEMBL301)':>20}{'EGFR (CHEMBL203)':>20}")
print(f"  {'coverage of training set':<26}{'2.4%':>20}{f'{100*have.mean():.1f}%':>20}")
print(f"  {'label within 2x (of covered)':<26}{'27.7%':>20}"
      f"{f'{100*(sub.log_err<=0.301).mean():.1f}%':>20}")

cens = train[have & train["egfr_censored"].fillna(False)]
print(f"\n  EGFR records marked censored ('>') but CSV stores '=': {len(cens)} "
      f"({100*len(cens)/len(train):.1f}% of training set)")
print(f"  (HANDOFF round-number heuristic estimated 22.6% censored)")
train.to_csv(HERE / "train_egfr_audit.csv", index=False)
