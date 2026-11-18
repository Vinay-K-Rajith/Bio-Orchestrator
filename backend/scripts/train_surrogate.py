"""Train the EGFR affinity surrogate (Model 1).

Produces three things:
  1. A point regressor for pIC50.
  2. Conformalized quantile regressors giving a calibrated 90% interval per
     molecule -- so downstream ranking can reason about uncertainty instead of
     trusting a bare point estimate.
  3. Reference fingerprints defining the applicability domain.

Reports scaffold-split and random-split metrics side by side. The gap between
them is the honest measure of how much the random number is inflated by
congeneric analogs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.featurize import (  # noqa: E402
    FEATURE_VERSION, canonical_smiles, featurize_many, murcko_scaffold, spec_dict,
)
from backend.app.splits import random_kfold, scaffold_kfold, scaffold_split  # noqa: E402

DATA_CSV = ROOT / "backend" / "data" / "egfr_chembl_labeled.csv"
MODEL_DIR = ROOT / "backend" / "models"
ALPHA = 0.10  # target 90% prediction interval

POINT_PARAMS = dict(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
    reg_lambda=2.0, reg_alpha=0.1,
    tree_method="hist", n_jobs=-1, random_state=42,
)
QUANTILE_PARAMS = dict(
    POINT_PARAMS, objective="reg:quantileerror", n_estimators=400, learning_rate=0.05,
)


def load_clean() -> pd.DataFrame:
    """Load, canonicalize, and collapse duplicate measurements.

    ChEMBL reports the same compound from multiple papers with different IC50s.
    Median-aggregating per canonical SMILES prevents the same molecule from
    appearing on both sides of a split with two different labels.
    """
    df = pd.read_csv(DATA_CSV)
    df["canonical"] = [canonical_smiles(s) for s in df["smiles"]]
    before = len(df)
    df = df.dropna(subset=["canonical", "pIC50"])
    agg = (
        df.groupby("canonical")
        .agg(pIC50=("pIC50", "median"), n_measurements=("pIC50", "size"),
             molecule_id=("molecule_id", "first"))
        .reset_index()
        .rename(columns={"canonical": "smiles"})
    )
    print(f"  loaded {before} rows -> {len(agg)} unique compounds "
          f"({before - len(agg)} duplicate/invalid collapsed)")
    print(f"  pIC50 range {agg.pIC50.min():.2f}-{agg.pIC50.max():.2f}, "
          f"mean {agg.pIC50.mean():.2f}, sd {agg.pIC50.std():.2f}")
    return agg


def metrics(y_true, y_pred) -> dict:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "n": int(len(y_true)),
    }


def cross_val(X, y, folds, label: str):
    """Out-of-fold predictions plus aggregate metrics."""
    oof = np.full(len(y), np.nan)
    for i, (tr, te) in enumerate(folds):
        model = xgb.XGBRegressor(**POINT_PARAMS)
        model.fit(X[tr], y[tr], verbose=False)
        oof[te] = model.predict(X[te])
        print(f"    fold {i + 1} done ({len(tr)} train / {len(te)} test)", flush=True)
    mask = ~np.isnan(oof)
    m = metrics(y[mask], oof[mask])
    print(f"  [{label}] R2={m['r2']:.3f}  RMSE={m['rmse']:.3f}  "
          f"MAE={m['mae']:.3f}  Spearman={m['spearman']:.3f}")
    return m, oof


def fit_cqr(X, y, scaffolds):
    """Conformalized Quantile Regression (Romano et al., 2019).

    Raw quantile regression is not calibrated -- its nominal 90% band covers
    whatever it happens to cover. CQR fixes this: fit lo/hi quantile models
    out-of-fold, measure how far the true value falls outside the predicted
    band, and take the (1-alpha) quantile of that error as a global correction.
    The result has finite-sample coverage guarantees under exchangeability, and
    the band still widens where the molecule is genuinely harder.
    """
    lo_q, hi_q = ALPHA / 2, 1.0 - ALPHA / 2
    oof_lo = np.full(len(y), np.nan)
    oof_hi = np.full(len(y), np.nan)

    for i, (tr, te) in enumerate(scaffold_kfold(scaffolds, n_splits=5, seed=7)):
        m_lo = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=lo_q)
        m_hi = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=hi_q)
        m_lo.fit(X[tr], y[tr], verbose=False)
        m_hi.fit(X[tr], y[tr], verbose=False)
        oof_lo[te] = m_lo.predict(X[te])
        oof_hi[te] = m_hi.predict(X[te])
        print(f"    cqr fold {i + 1} done", flush=True)

    mask = ~np.isnan(oof_lo)
    # Conformity score: signed distance outside the band (negative = inside).
    scores = np.maximum(oof_lo[mask] - y[mask], y[mask] - oof_hi[mask])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    correction = float(np.quantile(scores, level, method="higher"))

    covered = np.mean(
        (y[mask] >= oof_lo[mask] - correction) & (y[mask] <= oof_hi[mask] + correction)
    )
    raw_cov = np.mean((y[mask] >= oof_lo[mask]) & (y[mask] <= oof_hi[mask]))
    width = float(np.mean(oof_hi[mask] - oof_lo[mask] + 2 * correction))
    print(f"  [conformal] raw coverage={raw_cov:.3f} -> calibrated={covered:.3f} "
          f"(target {1 - ALPHA:.2f}), correction=+/-{correction:.3f} log units, "
          f"mean width={width:.2f}")
    return correction, {
        "alpha": ALPHA, "correction": correction,
        "raw_coverage": float(raw_cov), "calibrated_coverage": float(covered),
        "mean_interval_width": width,
    }


def main() -> int:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== STEP 1: load & clean ===")
    df = load_clean()

    print("\n=== STEP 2: featurize ===")
    X, fps, kept = featurize_many(df["smiles"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    y = df["pIC50"].to_numpy(dtype=np.float32)
    print(f"  X={X.shape} (2048 Morgan counts + 26 descriptors), fingerprints={fps.shape}")

    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    print(f"  {len(set(scaffolds))} unique Bemis-Murcko scaffolds "
          f"({len(df) / len(set(scaffolds)):.1f} analogs per scaffold)")

    print("\n=== STEP 3: honest evaluation ===")
    print("  random 5-fold (optimistic -- analogs leak across folds):")
    rand_m, _ = cross_val(X, y, list(random_kfold(len(y), 5, seed=0)), "random-cv")
    print("  scaffold 5-fold (realistic -- whole series held out):")
    scaf_m, oof = cross_val(X, y, list(scaffold_kfold(scaffolds, 5, seed=0)), "scaffold-cv")
    gap = rand_m["r2"] - scaf_m["r2"]
    print(f"  >>> optimism gap: R2 drops {gap:.3f} ({rand_m['r2']:.3f} -> "
          f"{scaf_m['r2']:.3f}) when analogs are removed.")

    print("\n=== STEP 4: conformal calibration ===")
    correction, conformal_meta = fit_cqr(X, y, scaffolds)

    print("\n=== STEP 5: fit production models on all data ===")
    point = xgb.XGBRegressor(**POINT_PARAMS)
    point.fit(X, y, verbose=False)
    lo = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=ALPHA / 2)
    hi = xgb.XGBRegressor(**QUANTILE_PARAMS, quantile_alpha=1 - ALPHA / 2)
    lo.fit(X, y, verbose=False)
    hi.fit(X, y, verbose=False)

    point.save_model(MODEL_DIR / "egfr_xgboost.json")
    lo.save_model(MODEL_DIR / "egfr_quantile_lo.json")
    hi.save_model(MODEL_DIR / "egfr_quantile_hi.json")
    np.save(MODEL_DIR / "reference_fps.npy", fps)
    np.save(MODEL_DIR / "reference_pic50.npy", y)

    holdout_tr, holdout_te = scaffold_split(scaffolds, 0.15, seed=0)
    meta = {
        "feature_version": FEATURE_VERSION,
        "feature_spec": spec_dict(),
        "target": "pIC50 (human EGFR, ChEMBL target CHEMBL203)",
        "n_compounds": int(len(df)),
        "n_scaffolds": int(len(set(scaffolds))),
        "label_stats": {"mean": float(y.mean()), "std": float(y.std()),
                        "min": float(y.min()), "max": float(y.max())},
        "metrics": {"random_cv": rand_m, "scaffold_cv": scaf_m,
                    "optimism_gap_r2": float(gap)},
        "conformal": conformal_meta,
        "holdout_scaffold_split": {"n_train": int(len(holdout_tr)),
                                   "n_test": int(len(holdout_te))},
        "hyperparameters": {"point": {k: v for k, v in POINT_PARAMS.items()},
                            "quantile": {k: v for k, v in QUANTILE_PARAMS.items()}},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (MODEL_DIR / "surrogate_metadata.json").write_text(json.dumps(meta, indent=2))

    df[["molecule_id", "smiles", "pIC50", "n_measurements"]].to_csv(
        MODEL_DIR / "reference_compounds.csv", index=False)

    print(f"\n=== DONE in {time.time() - t0:.1f}s ===")
    print(f"  artifacts -> {MODEL_DIR}")
    for f in sorted(MODEL_DIR.iterdir()):
        print(f"    {f.name:<32} {f.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
