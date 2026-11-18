"""Do XGBoost and the Tanimoto-kernel model make *different* mistakes?

If yes, the blend beats both by more than any feature-engineering delta found
so far. Same folds, same seeds, so the comparison is paired.
"""
import json, sys, time
from pathlib import Path
import numpy as np, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from backend.app.splits import scaffold_kfold

B = ROOT / "backend/experiments/blocks"
Y = np.load(B / "y.npy").astype(np.float64)
SCAF = json.loads((B / "scaffolds.json").read_text())
L = lambda n: np.load(B / f"{n}.npy").astype(np.float64)
BASE = dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=3, reg_lambda=2.0, reg_alpha=0.1,
            tree_method="hist", n_jobs=-1, random_state=42)

def score(y, p):
    r = y - p
    return dict(r2=float(1-r@r/((y-y.mean())@(y-y.mean()))),
                rmse=float(np.sqrt(np.mean(r**2))),
                spearman=float(spearmanr(y, p).statistic))

def tanimoto(X):
    G = X @ X.T; n = np.diag(G).copy()
    return G / (n[:,None] + n[None,:] - G + 1e-12)

# feature matrices
X_base  = np.hstack([L("morgan2_2048"), L("desc26")]).astype(np.float32)
X_fused = np.hstack([L("morgan2_2048"), L("fcfp2_2048"), L("torsion_2048"),
                     L("desc_full")]).astype(np.float32)
X_fused = np.nan_to_num(X_fused, nan=0.0, posinf=0.0, neginf=0.0)

M2b   = (np.expm1(L("morgan2_2048")) > 0).astype(np.float64)
FCb   = (np.expm1(L("fcfp2_2048"))   > 0).astype(np.float64)
TOb   = (np.expm1(L("torsion_2048")) > 0).astype(np.float64)
K_m2, K_fc, K_to = tanimoto(M2b), tanimoto(FCb), tanimoto(TOb)
K_multi = (K_m2 + K_fc + K_to) / 3

SEEDS = (0, 1, 2)
def oof_all():
    """Out-of-fold predictions from every model on identical folds."""
    out = {k: np.full((len(SEEDS), len(Y)), np.nan) for k in
           ("xgb_base", "xgb_fused", "krr_m2", "krr_multi")}
    for si, s in enumerate(SEEDS):
        for tr, te in scaffold_kfold(SCAF, 5, seed=s):
            m = xgb.XGBRegressor(**BASE); m.fit(X_base[tr], Y[tr])
            out["xgb_base"][si, te] = m.predict(X_base[te])
            m = xgb.XGBRegressor(**BASE); m.fit(X_fused[tr], Y[tr])
            out["xgb_fused"][si, te] = m.predict(X_fused[te])
            for nm, K in (("krr_m2", K_m2), ("krr_multi", K_multi)):
                k = KernelRidge(alpha=0.03, kernel="precomputed")
                k.fit(K[np.ix_(tr, tr)], Y[tr])
                out[nm][si, te] = k.predict(K[np.ix_(te, tr)])
        print(f"  seed {s} done", flush=True)
    return out

t0 = time.time(); O = oof_all(); print(f"OOF in {time.time()-t0:.0f}s\n")

def rep(name, P):
    rows = [score(Y, P[i]) for i in range(len(SEEDS))]
    a = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])) for k in rows[0]}
    print(f"{name:<40} R2={a['r2'][0]:.4f}+-{a['r2'][1]:.4f} "
          f"RMSE={a['rmse'][0]:.4f} rho={a['spearman'][0]:.4f}", flush=True)
    return a['r2'][0]

print("--- individuals ---")
for k in O: rep(k, O[k])

print("\n--- residual correlation (seed 0) ---")
R = {k: Y - O[k][0] for k in O}
ks = list(O)
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        print(f"  corr({ks[i]}, {ks[j]}) = {np.corrcoef(R[ks[i]], R[ks[j]])[0,1]:.3f}")

print("\n--- blends ---")
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    rep(f"{w:.1f}*xgb_base + {1-w:.1f}*krr_multi", w*O["xgb_base"] + (1-w)*O["krr_multi"])
print()
for w in (0.4, 0.5, 0.6):
    rep(f"{w:.1f}*xgb_fused + {1-w:.1f}*krr_multi", w*O["xgb_fused"] + (1-w)*O["krr_multi"])
print()
rep("mean(xgb_base, xgb_fused, krr_m2, krr_multi)",
    (O["xgb_base"]+O["xgb_fused"]+O["krr_m2"]+O["krr_multi"])/4)
rep("0.6*xgb_fused + 0.2*krr_multi + 0.2*krr_m2",
    0.6*O["xgb_fused"] + 0.2*O["krr_multi"] + 0.2*O["krr_m2"])

# does the blend help where it matters -- ranking among actives?
print("\n--- Spearman among actives (pIC50 >= 6) ---")
act = Y >= 6.0
for nm, P in (("xgb_base", O["xgb_base"]), ("xgb_fused", O["xgb_fused"]),
              ("krr_multi", O["krr_multi"]),
              ("0.5 blend", 0.5*O["xgb_fused"]+0.5*O["krr_multi"])):
    v = [spearmanr(Y[act], P[i][act]).statistic for i in range(len(SEEDS))]
    print(f"  {nm:<38} rho_actives={np.mean(v):.4f}+-{np.std(v):.4f}  (n={act.sum()})")
np.save(Path(__file__).parent/"oof.npy", np.array([O[k] for k in ks]))
