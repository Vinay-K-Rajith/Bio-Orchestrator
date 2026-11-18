"""Validate the surrogate against its ACTUAL target, EGFR.

In-training compounds are scored with their scaffold-held-out (OOF) prediction,
so nothing here is a memorised label.
"""
import json, sys
from pathlib import Path
import httpx, numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from backend.app.featurize import canonical_smiles
from backend.app.predictor import AffinityPredictor
from backend.app.bio_engine import ApplicabilityDomain, profile_batch, annotate_domain

BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}
HERE = Path(__file__).parent

PANEL = ["gefitinib", "erlotinib", "lapatinib", "afatinib", "osimertinib",
         "neratinib", "dacomitinib", "icotinib", "vandetanib", "brigatinib",
         "imatinib", "palbociclib", "dinaciclib",
         "aspirin", "caffeine", "metformin", "paracetamol"]
TRUTH_HINT = {  # literature EGFR class, for reading the table
    "gefitinib": "EGFR potent", "erlotinib": "EGFR potent", "lapatinib": "EGFR/HER2 potent",
    "afatinib": "EGFR covalent, potent", "osimertinib": "EGFR T790M, potent",
    "neratinib": "EGFR/HER2 potent", "dacomitinib": "EGFR potent",
    "icotinib": "EGFR potent", "vandetanib": "VEGFR/EGFR mixed",
    "brigatinib": "ALK, weak EGFR", "imatinib": "BCR-ABL, not EGFR",
    "palbociclib": "CDK4/6, not EGFR", "dinaciclib": "CDK, not EGFR",
    "aspirin": "negative", "caffeine": "negative", "metformin": "negative",
    "paracetamol": "negative"}

info = {}
with httpx.Client(headers=H, timeout=120.0) as c:
    for n in PANEL:
        r = c.get(f"{BASE}/molecule/search", params={"q": n, "format": "json"})
        mols = r.json().get("molecules", [])
        hit = next((m for m in mols if (m.get("pref_name") or "").lower() == n.lower()),
                   mols[0] if mols else None)
        if not hit: continue
        smi = (hit.get("molecule_structures") or {}).get("canonical_smiles")
        if not smi: continue
        mid = hit["molecule_chembl_id"]
        a = c.get(f"{BASE}/activity", params={"molecule_chembl_id": mid,
                  "target_chembl_id": "CHEMBL203", "standard_type": "IC50",
                  "format": "json", "limit": 200}).json().get("activities", [])
        vals = [float(x["standard_value"]) for x in a
                if x.get("standard_value") and x.get("standard_units") == "nM"]
        info[n] = {"id": mid, "smiles": smi,
                   "egfr_nM": float(np.median(vals)) if vals else None, "n_rec": len(vals)}

comp = pd.read_csv(ROOT / "backend/experiments/blocks/compounds.csv")
Y = np.load(ROOT / "backend/experiments/blocks/y.npy")
lut = {s: i for i, s in enumerate(comp["smiles"])}
OOF = np.load(HERE / "oof.npy")
oof_xgb, oof_blend = OOF[0].mean(0), (0.5*OOF[1] + 0.5*OOF[3]).mean(0)

pred = AffinityPredictor(); dom = ApplicabilityDomain.load()
names = list(info); smis = [info[n]["smiles"] for n in names]
res, _ = pred.predict_smiles(smis)
profs, fps, _ = profile_batch(smis); annotate_domain(profs, fps, dom)

print(f"{'drug':<13}{'ChEMBL EGFR':>13}{'true':>7}{'in-tr':>7}{'score':>7}{'src':>5}"
      f"{'maxTc':>7}  class")
print("-" * 84)
rows = []
for i, n in enumerate(names):
    ci = lut.get(canonical_smiles(smis[i]))
    true_p = (9 - np.log10(info[n]["egfr_nM"])) if info[n]["egfr_nM"] else None
    if ci is not None: sc, src = float(oof_xgb[ci]), "OOF"
    else:              sc, src = float(res["pic50"][i]), "ext"
    tp = f"{true_p:.2f}" if true_p else "  -  "
    en = f"{info[n]['egfr_nM']:.1f}" if info[n]["egfr_nM"] else "none"
    print(f"{n:<13}{en:>13}{tp:>7}{('YES' if ci is not None else 'no'):>7}"
          f"{sc:>7.2f}{src:>5}{profs[i].max_similarity:>7.2f}  {TRUTH_HINT[n]}")
    rows.append(dict(name=n, true=true_p, score=sc, src=src, cls=TRUTH_HINT[n]))

pairs = [(r["true"], r["score"]) for r in rows if r["true"] is not None]
t, s = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
print(f"\n  n with measured EGFR IC50 = {len(pairs)}")
print(f"  Spearman(pred, measured)  = {spearmanr(t, s).statistic:.3f}")
print(f"  MAE                       = {np.mean(np.abs(t-s)):.2f} log units")

pos = [r["score"] for r in rows if "potent" in r["cls"]]
neg = [r["score"] for r in rows if r["cls"] in ("negative", "BCR-ABL, not EGFR",
                                                "CDK4/6, not EGFR", "CDK, not EGFR")]
print(f"\n  potent-EGFR mean {np.mean(pos):.2f} (min {np.min(pos):.2f})")
print(f"  non-EGFR    mean {np.mean(neg):.2f} (max {np.max(neg):.2f})")
print(f"  clean separation? {'YES' if np.min(pos) > np.max(neg) else 'NO'}")
