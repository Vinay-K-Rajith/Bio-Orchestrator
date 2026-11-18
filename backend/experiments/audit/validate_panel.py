"""Known-drug validation. Does the surrogate rank real CDK inhibitors correctly?

The trap this avoids: most famous CDK inhibitors are IN the ChEMBL CDK2 training
set, so a plain predict() would be scored on memorised labels. For any panel
member found in training we instead report its *out-of-fold* prediction from the
scaffold-CV run -- the number the model produces when its whole scaffold series
was held out. That is the only honest figure for a compound the model has seen.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from backend.app.featurize import canonical_smiles, morgan_bits, mol_from_smiles
from backend.app.predictor import AffinityPredictor
from backend.app.bio_engine import ApplicabilityDomain, profile_batch, annotate_domain

HERE = Path(__file__).parent
panel = json.loads((HERE / "panel.json").read_text())
names = list(panel)

# training set, in the exact order build_blocks.py used (== oof column order)
comp = pd.read_csv(ROOT / "backend/experiments/blocks/compounds.csv")
Ytrue = np.load(ROOT / "backend/experiments/blocks/y.npy")
assert len(comp) == len(Ytrue)
train_lookup = {s: i for i, s in enumerate(comp["smiles"])}

# oof.npy: [model, seed, compound]; models were (xgb_base, xgb_fused, krr_m2, krr_multi)
OOF = np.load(HERE / "oof.npy")
oof_xgb  = OOF[0].mean(0)            # mean over the 3 seeds
oof_blend = (0.5 * OOF[1] + 0.5 * OOF[3]).mean(0)

pred = AffinityPredictor()
dom = ApplicabilityDomain.load()
smiles = [panel[n]["smiles"] for n in names]
canon = [canonical_smiles(s) for s in smiles]

res, kept = pred.predict_smiles(smiles)
profiles, fps, kept2 = profile_batch(smiles)
annotate_domain(profiles, fps, dom)
assert kept == kept2 == list(range(len(smiles))), "a panel SMILES failed to parse"

print(f"{'drug':<14}{'MW':>7}{'in-train':>9}{'true':>7}{'pred':>7}{'OOF':>7}"
      f"{'blend':>7}{'90% interval':>16}{'maxTc':>7}")
print("-" * 88)
rows = []
for i, n in enumerate(names):
    p = profiles[i]
    ti = train_lookup.get(canon[i])
    true = f"{Ytrue[ti]:.2f}" if ti is not None else "  -  "
    oofv = f"{oof_xgb[ti]:.2f}" if ti is not None else "  -  "
    blv  = f"{oof_blend[ti]:.2f}" if ti is not None else "  -  "
    band = f"[{res['pic50_lo'][i]:.2f}, {res['pic50_hi'][i]:.2f}]"
    print(f"{n:<14}{p.mw:>7.1f}{('YES' if ti is not None else 'no'):>9}{true:>7}"
          f"{res['pic50'][i]:>7.2f}{oofv:>7}{blv:>7}{band:>16}{p.max_similarity:>7.2f}")
    rows.append(dict(name=n, in_train=ti is not None,
                     true=(float(Ytrue[ti]) if ti is not None else None),
                     pred=float(res["pic50"][i]),
                     oof=(float(oof_xgb[ti]) if ti is not None else None),
                     blend=(float(oof_blend[ti]) if ti is not None else None),
                     note=panel[n]["note"], sim=p.max_similarity,
                     in_domain=p.in_domain, herg=None))

# ---- the ordering test -------------------------------------------------
print("\n--- ranking by honest score (OOF where in-train, else prediction) ---")
def honest(r):  return r["oof"] if r["oof"] is not None else r["pred"]
for r in sorted(rows, key=lambda r: -honest(r)):
    tag = "OOF" if r["oof"] is not None else "ext"
    print(f"  {honest(r):>6.2f} ({tag})  {r['name']:<14} {r['note']}")

CDK_ACTIVE = {"dinaciclib", "alvocidib", "seliciclib", "AT7519", "milciclib", "riviciclib"}
NEGATIVE   = {"aspirin", "caffeine", "paracetamol", "metformin"}
act = [honest(r) for r in rows if r["name"] in CDK_ACTIVE]
neg = [honest(r) for r in rows if r["name"] in NEGATIVE]
print(f"\n  CDK-active mean  {np.mean(act):.2f}  (min {np.min(act):.2f})")
print(f"  negative   mean  {np.mean(neg):.2f}  (max {np.max(neg):.2f})")
print(f"  separation       {np.mean(act) - np.mean(neg):.2f} log units")
print(f"  clean split?     {'YES' if np.min(act) > np.max(neg) else 'NO'}")
json.dump(rows, open(HERE / "panel_results.json", "w"), indent=2)
