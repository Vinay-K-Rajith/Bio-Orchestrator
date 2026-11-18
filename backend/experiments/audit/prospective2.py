"""Redo the prospective join locally against the FULL EGFR table (no batch API
limits), so the held-out sample is as large as the data allows."""
import sys
from pathlib import Path
import httpx, numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from backend.app.pipeline import get_pipeline
from backend.app.bio_engine import AD_SIMILARITY_FLOOR
HERE = Path(__file__).parent
BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}

F = HERE / "egfr_full.csv"
if not F.exists():
    rows, off = [], 0
    with httpx.Client(headers=H, timeout=180.0) as c:
        while True:
            a = c.get(f"{BASE}/activity", params={"target_chembl_id": "CHEMBL203",
                "standard_type": "IC50", "format": "json", "limit": 1000,
                "offset": off}).json().get("activities", [])
            if not a: break
            rows += [{"molecule_id": x.get("molecule_chembl_id"), "nM": x.get("standard_value"),
                      "units": x.get("standard_units"), "rel": x.get("standard_relation")} for x in a]
            off += len(a)
            if len(a) < 1000: break
    d = pd.DataFrame(rows).dropna(subset=["molecule_id", "nM"])
    d = d[d.units == "nM"]; d["nM"] = d["nM"].astype(float)
    d.to_csv(F, index=False)
d = pd.read_csv(F)
print(f"EGFR IC50 table: {len(d)} records / {d.molecule_id.nunique()} molecules")

pipe = get_pipeline()
pool = pipe.pool_index.copy()
pool["pic50"] = pipe.pool_pic50
pool["sim"] = [p.max_similarity for p in pipe.pool_profiles]
pool["in_domain"] = [p.in_domain for p in pipe.pool_profiles]
print(f"\n--- applicability domain over the 10,000-compound pool ---")
print(f"  in-domain (maxTc >= {AD_SIMILARITY_FLOOR}): {pool.in_domain.sum()} "
      f"({100*pool.in_domain.mean():.1f}%)")
print(f"  median max-Tanimoto to training set: {pool.sim.median():.3f}")

g = d.groupby("molecule_id")
truth = g["nM"].median()
allcens = g["rel"].apply(lambda s: bool(s.isin([">", ">="]).all()))
m = pool[pool.molecule_id.isin(truth.index)].copy()
m["true_nM"] = m.molecule_id.map(truth)
m["cens"] = m.molecule_id.map(allcens).fillna(False)
m["true_pic50"] = 9 - np.log10(m.true_nM)
clean = m[~m.cens]
print(f"\n--- PROSPECTIVE SET: {len(m)} pool compounds with measured EGFR IC50, "
      f"{len(clean)} uncensored ---")
r = clean.true_pic50 - clean.pic50
ss = ((clean.true_pic50 - clean.true_pic50.mean())**2).sum()
print(f"  Spearman(pred, measured) = {spearmanr(clean.true_pic50, clean.pic50).statistic:+.3f}"
      f"   (scaffold-CV claim: 0.844)")
print(f"  R2                       = {1 - (r@r)/ss:+.3f}   (scaffold-CV claim: 0.707)")
print(f"  RMSE                     = {np.sqrt((r**2).mean()):.3f}   (claim: 0.812)")
print(f"  bias (true - pred)       = {r.mean():+.3f} log units")
print(f"  predicted range {clean.pic50.min():.2f}-{clean.pic50.max():.2f} vs "
      f"measured {clean.true_pic50.min():.2f}-{clean.true_pic50.max():.2f}")

print(f"\n  conformal 90% interval coverage on this held-out set:")
lo = np.interp(clean.index, pool.index, pipe.pool_lo[pool.index.get_indexer(pool.index)]) if False else None
idx = pool.index.get_indexer(clean.index)
cov = ((clean.true_pic50 >= pipe.pool_lo[idx]) & (clean.true_pic50 <= pipe.pool_hi[idx])).mean()
print(f"    {cov*100:.1f}%   (claimed 90.1%)")

pot = clean[clean.true_pic50 >= 7]
print(f"\n  measured-potent (<=100 nM): {len(pot)} of {len(clean)}")
if len(pot):
    print(f"    model predicts them at mean {pot.pic50.mean():.2f} (max {pot.pic50.max():.2f})")
    ranks = pool.pic50.rank(ascending=False)
    print(f"    their median rank among 10,000: "
          f"{int(ranks[pot.index].median())}  (random would be 5000)")
clean.to_csv(HERE / "prospective_clean.csv", index=False)
