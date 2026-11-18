"""Train the hERG cardiotoxicity classifier.

hERG (KCNH2) blockade causes QT prolongation and is one of the most common
reasons a potent compound dies in preclinical development. The original plan
listed a "hERG flag" with nothing behind it; this trains a real classifier on
ChEMBL CHEMBL240 measurements.

Evaluated by scaffold split and AUPRC alongside ROC-AUC. On an imbalanced
screening task ROC-AUC flatters a model badly -- AUPRC against the positive
base rate is the number that says whether the flag is actually informative.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.featurize import (  # noqa: E402
    FEATURE_VERSION, featurize_many, murcko_scaffold, spec_dict,
)
from backend.app.splits import scaffold_kfold  # noqa: E402

DATA_CSV = ROOT / "backend" / "data" / "herg_chembl.csv"
MODEL_DIR = ROOT / "backend" / "models"

PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
    reg_lambda=2.0, eval_metric="aucpr",
    tree_method="hist", n_jobs=-1, random_state=42,
)


def main() -> int:
    t0 = time.time()
    if not DATA_CSV.exists():
        print(f"ERROR: {DATA_CSV} not found.\n"
              "Run first: python backend/scripts/fetch_chembl.py herg")
        return 1

    print("=== STEP 1: load ===")
    df = pd.read_csv(DATA_CSV).dropna(subset=["smiles", "is_blocker"])
    print(f"  {len(df)} compounds, {int(df.is_blocker.sum())} blockers "
          f"({df.is_blocker.mean():.1%} positive)")

    print("\n=== STEP 2: featurize ===")
    X, _, kept = featurize_many(df["smiles"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    y = df["is_blocker"].to_numpy(dtype=int)
    print(f"  X={X.shape}")

    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    print(f"  {len(set(scaffolds))} unique scaffolds")

    print("\n=== STEP 3: scaffold-split cross-validation ===")
    oof = np.full(len(y), np.nan)
    for i, (tr, te) in enumerate(scaffold_kfold(scaffolds, 5, seed=0)):
        # Rebalance by class ratio inside each fold rather than globally.
        pos, neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        model = xgb.XGBClassifier(**PARAMS,
                                  scale_pos_weight=(neg / pos) if pos else 1.0)
        model.fit(X[tr], y[tr], verbose=False)
        oof[te] = model.predict_proba(X[te])[:, 1]
        print(f"    fold {i + 1} done", flush=True)

    mask = ~np.isnan(oof)
    base_rate = float(y[mask].mean())
    auc = float(roc_auc_score(y[mask], oof[mask]))
    auprc = float(average_precision_score(y[mask], oof[mask]))
    brier = float(brier_score_loss(y[mask], oof[mask]))
    print(f"  ROC-AUC={auc:.3f}  AUPRC={auprc:.3f} (base rate {base_rate:.3f}, "
          f"lift {auprc / base_rate:.2f}x)  Brier={brier:.3f}")

    # Pick the threshold that maximises F1 on out-of-fold predictions, so the
    # binary flag is tuned rather than defaulted to 0.5.
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (oof[mask] >= thr).astype(int)
        tp = int(((pred == 1) & (y[mask] == 1)).sum())
        fp = int(((pred == 1) & (y[mask] == 0)).sum())
        fn = int(((pred == 0) & (y[mask] == 1)).sum())
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"  best F1={best_f1:.3f} at threshold {best_thr:.2f}")

    print("\n=== STEP 4: fit production model ===")
    pos, neg = int(y.sum()), int((1 - y).sum())
    final = xgb.XGBClassifier(**PARAMS, scale_pos_weight=(neg / pos) if pos else 1.0)
    final.fit(X, y, verbose=False)
    final.save_model(MODEL_DIR / "herg_xgboost.json")

    meta = {
        "feature_version": FEATURE_VERSION,
        "feature_spec": spec_dict(),
        "target": "hERG (KCNH2, CHEMBL240) blockade, pChEMBL > 5 = blocker",
        "n_compounds": int(len(df)),
        "n_scaffolds": int(len(set(scaffolds))),
        "positive_rate": base_rate,
        "metrics": {"roc_auc": auc, "auprc": auprc,
                    "auprc_lift_over_base_rate": auprc / base_rate,
                    "brier": brier, "best_f1": best_f1},
        "decision_threshold": best_thr,
        "validation": "scaffold-split 5-fold cross-validation",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (MODEL_DIR / "herg_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved herg_xgboost.json + herg_metadata.json")
    print(f"\n=== DONE in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
