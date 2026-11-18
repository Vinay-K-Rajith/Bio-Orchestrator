"""First-principles structure analysis of the EGFR set.

Not model tuning -- asking what the data can support at all, so effort goes
where the ceiling actually is.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
RDLogger.DisableLog("rdApp.*")

B = Path(__file__).parent / "blocks"
y = np.load(B / "y.npy").astype(np.float64)
scaf = json.loads((B / "scaffolds.json").read_text())
df = pd.read_csv(B / "compounds.csv")

print("=" * 72)
print("1. LABEL DISTRIBUTION  -- looking for assay censoring artefacts")
print("=" * 72)
print(f"n={len(y)}  mean={y.mean():.3f}  sd={y.std():.3f}  min={y.min():.2f} max={y.max():.2f}")
vc = Counter(np.round(y, 4))
print("\nmost repeated exact pIC50 values (a spike = a censored '>' cutoff "
      "recorded as '='):")
for v, c in vc.most_common(12):
    ic50 = 10 ** (9 - v)
    print(f"   pIC50={v:<8} n={c:<5} ({c/len(y):5.1%})  == IC50 {ic50:>12,.0f} nM")
# round-number IC50 pile-ups
ic50 = 10 ** (9 - y)
for cut in [10000, 100000, 1000, 50000, 20000]:
    n = int(np.sum(np.isclose(ic50, cut, rtol=1e-3)))
    if n:
        print(f"   IC50 exactly {cut:>7,} nM : {n:4d} ({n/len(y):.1%})")

print()
print("=" * 72)
print("2. VARIANCE DECOMPOSITION -- is the signal 'which series' or "
      "'which analog'?")
print("=" * 72)
g = pd.DataFrame({"y": y, "s": scaf})
sizes = g.groupby("s").size()
grand = y.mean()
means = g.groupby("s")["y"].mean()
between = float((sizes * (means - grand) ** 2).sum())
within = float(g.groupby("s")["y"].apply(lambda v: ((v - v.mean()) ** 2).sum()).sum())
total = between + within
print(f"  total SS       {total:9.1f}")
print(f"  between-scaffold {between:9.1f}  ({between/total:.1%})")
print(f"  within-scaffold  {within:9.1f}  ({within/total:.1%})")
print(f"\n  scaffolds: {len(sizes)}  singletons: {int((sizes==1).sum())} "
      f"({(sizes==1).mean():.0%})")
print(f"  largest series: {sorted(sizes.values)[-8:]}")
print(f"  compounds in singleton scaffolds: {int(sizes[sizes==1].sum())} "
      f"({sizes[sizes==1].sum()/len(y):.1%})")
# ceiling for a model that only knows the scaffold
print(f"\n  R2 of an oracle that predicts each scaffold's OWN mean: "
      f"{between/total:.3f}")
print("  (i.e. even perfect series-level knowledge leaves "
      f"{within/total:.0%} of variance to substituent effects)")

print()
print("=" * 72)
print("3. ACTIVITY CLIFFS -- irreducible error for any similarity learner")
print("=" * 72)
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(m) for m in mols]
n = len(fps)
sims_max, nn_dy, nn_idx = np.zeros(n), np.zeros(n), np.zeros(n, int)
cliff_pairs = 0
tot_pairs = 0
for i in range(n):
    s = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
    s[i] = -1
    j = int(np.argmax(s))
    sims_max[i], nn_idx[i] = s[j], j
    nn_dy[i] = abs(y[i] - y[j])
    hi = s >= 0.7
    tot_pairs += int(hi.sum())
    cliff_pairs += int((hi & (np.abs(y - y[i]) >= 2.0)).sum())
print(f"  nearest-neighbour Tanimoto: median={np.median(sims_max):.3f} "
      f"mean={sims_max.mean():.3f}")
print(f"  |dpIC50| to nearest neighbour: median={np.median(nn_dy):.3f} "
      f"mean={nn_dy.mean():.3f}")
print(f"  pairs with Tanimoto>=0.70: {tot_pairs//2}, of which "
      f"|dpIC50|>=2.0 (cliffs): {cliff_pairs//2} ({cliff_pairs/max(tot_pairs,1):.1%})")
print(f"\n  RMSE of a 1-NN predictor (upper bound on 'just look it up'): "
      f"{np.sqrt(np.mean((y - y[nn_idx])**2)):.3f}")
np.save(B / "nn_sim.npy", sims_max)
np.save(B / "nn_idx.npy", nn_idx)

print()
print("=" * 72)
print("4. HOW FAR IS A HELD-OUT SCAFFOLD FROM TRAINING CHEMISTRY?")
print("=" * 72)
from backend.app.splits import scaffold_kfold
sim_to_train = np.zeros(n)
for tr, te in scaffold_kfold(scaf, 5, seed=0):
    trs = [fps[i] for i in tr]
    for i in te:
        sim_to_train[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], trs))
np.save(B / "sim_to_train.npy", sim_to_train)
print(f"  under scaffold-CV, max Tanimoto to any training compound:")
for q in [10, 25, 50, 75, 90]:
    print(f"    p{q:<3} = {np.percentile(sim_to_train, q):.3f}")
print(f"  fraction with <0.4 similarity to anything in train: "
      f"{(sim_to_train < 0.4).mean():.1%}")
