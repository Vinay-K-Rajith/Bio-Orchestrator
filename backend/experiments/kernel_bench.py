"""Kernel methods vs XGBoost on the EGFR surrogate, same repeated scaffold CV.

First principles: XGBoost splits on individual fingerprint bits. It has to
*rebuild* the notion of "similar molecule" out of conjunctions of bits, which
costs labels. A Tanimoto kernel hands it that notion for free -- the right
prior for n=2765.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from backend.app.splits import scaffold_kfold

B = ROOT / "backend/experiments/blocks"
Y = np.load(B / "y.npy").astype(np.float64)
SCAF = json.loads((B / "scaffolds.json").read_text())
L = lambda n: np.load(B / f"{n}.npy").astype(np.float64)

def score(y, p):
    r = y - p
    return dict(r2=float(1 - r@r/((y-y.mean())@(y-y.mean()))),
                rmse=float(np.sqrt(np.mean(r**2))),
                spearman=float(spearmanr(y, p).statistic))

def tanimoto(X):
    """K(x,y) = <x,y> / (|x|^2 + |y|^2 - <x,y>).  Jaccard for binary; PSD."""
    G = X @ X.T
    n = np.diag(G).copy()
    return G / (n[:, None] + n[None, :] - G + 1e-12)

def rbf(X):
    """Median-heuristic RBF on rank-normalised descriptors (y-independent)."""
    r = np.argsort(np.argsort(X, axis=0), axis=0) / (len(X) - 1.0)   # -> [0,1]
    D = ((r**2).sum(1)[:, None] + (r**2).sum(1)[None, :] - 2*r@r.T).clip(0)
    med = np.median(D[D > 0])
    return np.exp(-D / med)

# ---- kernels -----------------------------------------------------------
M2 = np.expm1(L("morgan2_2048"))          # blocks are log1p counts -> raw counts
K_tan_cnt = tanimoto(M2)
K_tan_bin = tanimoto((M2 > 0).astype(np.float64))
K_fcfp    = tanimoto(np.expm1(L("fcfp2_2048")))
K_tors    = tanimoto(np.expm1(L("torsion_2048")))
K_desc    = rbf(L("desc_full"))
print("kernels built", flush=True)

def cv_kernel(K, model, seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        oof = np.full(len(Y), np.nan)
        for tr, te in scaffold_kfold(SCAF, 5, seed=s):
            m = model()
            m.fit(K[np.ix_(tr, tr)], Y[tr])
            oof[te] = m.predict(K[np.ix_(te, tr)])
        rows.append(score(Y, oof))
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
            for k in rows[0]}, rows

def show(name, agg, t):
    print(f"{name:<44} R2={agg['r2'][0]:.4f}+-{agg['r2'][1]:.4f} "
          f"RMSE={agg['rmse'][0]:.4f} rho={agg['spearman'][0]:.4f}  ({t:.0f}s)", flush=True)

print("\n--- Kernel ridge, Tanimoto(Morgan2 counts), alpha sweep ---")
for a in (0.003, 0.01, 0.03, 0.1, 0.3):
    t = time.time(); agg, _ = cv_kernel(K_tan_cnt, lambda a=a: KernelRidge(alpha=a, kernel="precomputed"))
    show(f"KRR tanimoto-count alpha={a}", agg, time.time()-t)

print("\n--- binary vs count fingerprint ---")
t = time.time(); agg, _ = cv_kernel(K_tan_bin, lambda: KernelRidge(alpha=0.03, kernel="precomputed"))
show("KRR tanimoto-binary alpha=0.03", agg, time.time()-t)

print("\n--- SVR (MoleculeACE's best classical model) ---")
for C in (3.0, 10.0, 30.0):
    t = time.time(); agg, _ = cv_kernel(K_tan_cnt, lambda C=C: SVR(kernel="precomputed", C=C, epsilon=0.1))
    show(f"SVR tanimoto-count C={C}", agg, time.time()-t)

print("\n--- multi-kernel sums (fp similarity + descriptor RBF) ---")
combos = {
  "0.5*M2 + 0.5*desc":            0.5*K_tan_cnt + 0.5*K_desc,
  "0.7*M2 + 0.3*desc":            0.7*K_tan_cnt + 0.3*K_desc,
  "M2+FCFP+Tors+desc (equal)":    (K_tan_cnt + K_fcfp + K_tors + K_desc)/4,
  "M2+FCFP+Tors (equal)":         (K_tan_cnt + K_fcfp + K_tors)/3,
}
for nm, K in combos.items():
    for a in (0.01, 0.03):
        t = time.time(); agg, _ = cv_kernel(K, lambda a=a: KernelRidge(alpha=a, kernel="precomputed"))
        show(f"KRR {nm} a={a}", agg, time.time()-t)
np.save(Path(__file__).parent/"K_best.npy", (K_tan_cnt + K_fcfp + K_tors + K_desc)/4)
