"""How much of the model's error is spent on censored 'inactive' labels?

A round-number IC50 at or above 1 uM is almost always a reporting convention
for 'no activity at the top assay concentration', written into ChEMBL with an
'=' relation. Fitting a precise pIC50 to those points is fitting a number that
was never measured.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from bench import build, Y, SCAFFOLDS, BASE
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.splits import scaffold_kfold

y = Y.astype(np.float64)
ic50 = 10 ** (9 - y)
# round mantissas used by assay panels for a top-concentration readout
MANT = np.array([1.0, 2.0, 2.5, 5.0])
def is_round(v):
    e = np.floor(np.log10(v))
    m = v / 10 ** e
    return np.any(np.isclose(m, MANT, rtol=2e-3))
censored = np.array([v >= 1000 and is_round(v) for v in ic50])
print(f"suspected censored (round IC50 >= 1 uM): {censored.sum()} "
      f"({censored.mean():.1%})   pIC50 range {y[censored].min():.2f}-{y[censored].max():.2f}")
print(f"  their mean pIC50 {y[censored].mean():.2f} vs clean {y[~censored].mean():.2f}")
np.save(Path(__file__).parent / "blocks/censored.npy", censored)

X = build(["morgan2_2048", "desc26"])
oof = np.full(len(y), np.nan)
for tr, te in scaffold_kfold(SCAFFOLDS, 5, seed=0):
    m = xgb.XGBRegressor(**BASE); m.fit(X[tr], Y[tr]); oof[te] = m.predict(X[te])

def rep(mask, name):
    r = y[mask] - oof[mask]
    ss = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    print(f"  {name:<34} n={mask.sum():<5} RMSE={np.sqrt((r**2).mean()):.3f}  "
          f"MAE={np.abs(r).mean():.3f}  bias={r.mean():+.3f}  "
          f"R2={1-float(np.sum(r**2))/ss:.3f}")

print("\nbaseline OOF error, split by label provenance:")
rep(np.ones(len(y), bool), "ALL")
rep(censored, "censored (round, >=1uM)")
rep(~censored, "clean (real measurements)")

tot = float(np.sum((y - oof) ** 2))
print(f"\n  share of total squared error spent on censored points: "
      f"{float(np.sum((y[censored]-oof[censored])**2))/tot:.1%} "
      f"of error from {censored.mean():.1%} of data")

print("\nerror by potency band (what the screener actually cares about):")
for lo, hi, nm in [(-9, 5, "pIC50 < 5      (inactive)"), (5, 6, "5-6"), (6, 7, "6-7"),
                   (7, 8, "7-8   (hit range)"), (8, 20, ">= 8  (potent)")]:
    m = (y >= lo) & (y < hi)
    if m.sum(): rep(m, nm)

print("\nerror vs similarity to training chemistry:")
sim = np.load(Path(__file__).parent / "blocks/sim_to_train.npy")
for lo, hi in [(0, .4), (.4, .55), (.55, .7), (.7, .85), (.85, 1.01)]:
    m = (sim >= lo) & (sim < hi)
    if m.sum(): rep(m, f"max-Tanimoto {lo:.2f}-{hi:.2f}")
np.save(Path(__file__).parent / "blocks/oof_baseline.npy", oof)
