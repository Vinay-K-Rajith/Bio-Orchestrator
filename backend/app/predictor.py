"""Affinity surrogate inference (Stage 1).

Wraps the trained XGBoost artifacts behind one object that returns, per
molecule, a point pIC50 *and* a calibrated 90% interval. Callers should treat
the interval as the real output: a point estimate of 7.1 with a band of
[5.9, 8.3] is a very different decision from the same point with [6.9, 7.3].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from .featurize import FEATURE_VERSION, featurize_many

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def pic50_to_ic50_nm(pic50: np.ndarray | float):
    """pIC50 -> IC50 in nM.  pIC50 = -log10(IC50 in M), so IC50_nM = 10^(9-pIC50).

    Deliberately named IC50, not Kd. The training labels are IC50 values;
    converting them to a dissociation constant would require the Cheng-Prusoff
    relation and knowledge of the assay ATP concentration, which is not in the
    dataset. Reporting an IC50 as a Kd is a category error.
    """
    return np.power(10.0, 9.0 - np.asarray(pic50, dtype=np.float64))


class AffinityPredictor:
    """Loads the point model, both quantile models, and the conformal width."""

    def __init__(self, model_dir: Path | None = None):
        self.dir = Path(model_dir or MODEL_DIR)
        meta_path = self.dir / "surrogate_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained surrogate at {self.dir}. "
                "Run: python backend/scripts/train_surrogate.py"
            )
        self.meta = json.loads(meta_path.read_text())

        got = self.meta.get("feature_version")
        if got != FEATURE_VERSION:
            raise RuntimeError(
                f"Feature version mismatch: model was trained with v{got}, "
                f"code produces v{FEATURE_VERSION}. Retrain the surrogate."
            )

        self.point = xgb.XGBRegressor()
        self.point.load_model(self.dir / "egfr_xgboost.json")
        self.lo = xgb.XGBRegressor()
        self.lo.load_model(self.dir / "egfr_quantile_lo.json")
        self.hi = xgb.XGBRegressor()
        self.hi.load_model(self.dir / "egfr_quantile_hi.json")

        conf = self.meta.get("conformal", {})
        self.correction = float(conf.get("correction", 0.0))
        self.alpha = float(conf.get("alpha", 0.10))
        self.coverage = float(conf.get("calibrated_coverage", float("nan")))

    @property
    def scaffold_r2(self) -> float:
        return float(self.meta["metrics"]["scaffold_cv"]["r2"])

    @property
    def scaffold_rmse(self) -> float:
        return float(self.meta["metrics"]["scaffold_cv"]["rmse"])

    def predict_features(self, X: np.ndarray) -> dict:
        """Predict from an already-built feature matrix."""
        if len(X) == 0:
            empty = np.zeros(0)
            return {"pic50": empty, "pic50_lo": empty, "pic50_hi": empty,
                    "ic50_nm": empty, "ic50_nm_lo": empty, "ic50_nm_hi": empty,
                    "interval_width": empty}

        point = self.point.predict(X).astype(np.float64)
        lo = self.lo.predict(X).astype(np.float64) - self.correction
        hi = self.hi.predict(X).astype(np.float64) + self.correction

        # Quantile models are fit independently and can cross on hard molecules;
        # enforce ordering and keep the point estimate inside its own interval.
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        lo = np.minimum(lo, point)
        hi = np.maximum(hi, point)

        return {
            "pic50": point,
            "pic50_lo": lo,
            "pic50_hi": hi,
            "ic50_nm": pic50_to_ic50_nm(point),
            "ic50_nm_lo": pic50_to_ic50_nm(hi),   # high pIC50 = low IC50
            "ic50_nm_hi": pic50_to_ic50_nm(lo),
            "interval_width": hi - lo,
        }

    def predict_smiles(self, smiles_list: list[str]) -> tuple[dict, list[int]]:
        """Predict from SMILES. Also returns which input rows survived parsing."""
        X, _, kept = featurize_many(smiles_list)
        return self.predict_features(X), kept


if __name__ == "__main__":
    import time

    p = AffinityPredictor()
    print(f"loaded surrogate: scaffold R2={p.scaffold_r2:.3f} "
          f"RMSE={p.scaffold_rmse:.3f}, conformal coverage={p.coverage:.3f}")

    known = {
        "imatinib (in training set)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "roscovitine (CDK inhibitor)": "CCC(CO)Nc1nc(NCc2ccccc2)c2c(n1)n(C(C)C)cn2",
        "caffeine (not an EGFR drug)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "ethanol (nonsense control)": "CCO",
    }
    res, kept = p.predict_smiles(list(known.values()))
    print(f"\n{'molecule':<30}{'pIC50':>8}{'90% interval':>18}{'IC50 nM':>12}")
    print("-" * 68)
    for i, name in enumerate(known):
        band = f"[{res['pic50_lo'][i]:.2f}, {res['pic50_hi'][i]:.2f}]"
        print(f"{name:<30}{res['pic50'][i]:>8.2f}{band:>18}{res['ic50_nm'][i]:>12.1f}")

    n = 10000
    big = [known["imatinib (in training set)"]] * n
    t0 = time.time()
    X, _, _ = featurize_many(big)
    t_feat = time.time() - t0
    t0 = time.time()
    p.predict_features(X)
    t_pred = time.time() - t0
    print(f"\nthroughput on {n:,} molecules: featurize {t_feat:.2f}s, "
          f"predict {t_pred:.2f}s, total {t_feat + t_pred:.2f}s")
