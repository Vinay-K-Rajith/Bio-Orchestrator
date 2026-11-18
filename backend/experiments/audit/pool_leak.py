"""The pool was built to exclude 'CDK2' compounds. But the label file is EGFR,
so the exclusion list was an EGFR list. Question: does the screening pool
contain EGFR-labelled chemistry the model was effectively trained on?
"""
from pathlib import Path
import httpx, pandas as pd
HERE = Path(__file__).parent; ROOT = Path(__file__).resolve().parents[3]
BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}

train = pd.read_csv(ROOT / "backend/data/cdk2_chembl_labeled.csv")
pool  = pd.read_csv(ROOT / "backend/models/pool_cache/pool_index.csv")
print(f"pool={len(pool)}  train={len(train)}")
print(f"direct molecule_id overlap pool<->train: "
      f"{len(set(pool.molecule_id) & set(train.molecule_id))}")

egfr = pd.read_csv(HERE / "train_egfr_audit.csv")  # has egfr_median_nM
# how much of the pool has its own EGFR record in ChEMBL?
rows, off = [], 0
with httpx.Client(headers=H, timeout=180.0) as c:
    while True:
        r = c.get(f"{BASE}/activity", params={"target_chembl_id": "CHEMBL203",
                  "standard_type": "IC50", "format": "json", "limit": 1000, "offset": off})
        a = r.json().get("activities", [])
        if not a: break
        rows += [x.get("molecule_chembl_id") for x in a]
        off += len(a)
        if len(a) < 1000: break
egfr_mols = set(x for x in rows if x)
inpool = pool.molecule_id.isin(egfr_mols)
print(f"\npool compounds WITH a ChEMBL EGFR IC50 record: {inpool.sum()} "
      f"({100*inpool.mean():.1f}% of pool)")
print("  -> these are EGFR-labelled molecules the 'CDK2' exclusion never removed")
pool.loc[inpool, ["molecule_id"]].to_csv(HERE / "pool_egfr_leak.csv", index=False)
