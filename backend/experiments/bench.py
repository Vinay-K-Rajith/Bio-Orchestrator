"""Scaffold-CV benchmark over cached feature blocks.

Repeated (multi-seed) scaffold K-fold so a reported delta can be compared
against its own seed-to-seed spread. A gain smaller than the spread is noise.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from backend.app.splits import scaffold_kfold, random_kfold

B = Path(__file__).parent / "blocks"
Y = np.load(B / "y.npy")
SCAFFOLDS = json.loads((B / "scaffolds.json").read_text())
_cache: dict[str, np.ndarray] = {}

BASE = dict(n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.5, min_child_weight=3, reg_lambda=2.0, reg_alpha=0.1,
            tree_method="hist", n_jobs=-1, random_state=42)


def load(name: str) -> np.ndarray:
    if name not in _cache:
        _cache[name] = np.load(B / f"{name}.npy")
    return _cache[name]


def build(blocks, min_count: int = 0) -> np.ndarray:
    """Concatenate blocks; optionally drop columns active in < min_count molecules.

    Sparse fingerprint blocks are mostly dead columns on a 2.7k dataset: a bit
    firing in two molecules cannot support a split that generalises, it can only
    memorise. Pruning them is free dimensionality reduction, not selection bias,
    because the criterion never looks at y.
    """
    X = np.hstack([load(b) for b in blocks])
    if min_count > 0:
        keep = (np.nan_to_num(X, nan=0.0) != 0).sum(0) >= min_count
        X = X[:, keep]
    return np.ascontiguousarray(X, dtype=np.float32)


def score(y, p):
    r = y - p
    return dict(r2=float(1 - r @ r / ((y - y.mean()) @ (y - y.mean()))),
                rmse=float(np.sqrt(np.mean(r ** 2))), mae=float(np.mean(np.abs(r))),
                spearman=float(spearmanr(y, p).statistic))


def run(blocks, params=None, min_count=0, seeds=(0, 1, 2), label=None, split="scaffold",
        weights=None, verbose=True):
    X = build(blocks, min_count)
    P = dict(BASE, **(params or {}))
    rows = []
    t0 = time.time()
    for s in seeds:
        oof = np.full(len(Y), np.nan)
        folds = (scaffold_kfold(SCAFFOLDS, 5, seed=s) if split == "scaffold"
                 else random_kfold(len(Y), 5, seed=s))
        for tr, te in folds:
            m = xgb.XGBRegressor(**P)
            m.fit(X[tr], Y[tr], sample_weight=None if weights is None else weights[tr])
            oof[te] = m.predict(X[te])
        rows.append(score(Y, oof))
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
           for k in rows[0]}
    name = label or "+".join(blocks)
    if verbose:
        print(f"{name:<46} d={X.shape[1]:<6} R2={agg['r2'][0]:.4f}±{agg['r2'][1]:.4f} "
              f"RMSE={agg['rmse'][0]:.4f} rho={agg['spearman'][0]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return agg, X.shape[1]
