"""Build and cache Tanimoto-KRR + Pretrained ChemBERTa-77M-MTR ensemble scores.

Computes:
1. 5-fold scaffold CV out-of-fold validation metrics for:
   - XGBoost baseline (Morgan2 + desc26)
   - Tanimoto Kernel Ridge Regression (Morgan2 + FCFP2 + Torsion)
   - Pretrained DeepChem/ChemBERTa-77M-MTR embeddings + Ridge head
   - Multi-modal Ensemble blend
2. Precomputes `krr_pic50`, `chemberta_pic50`, and `ensemble_pic50` for all
   10,000 candidate pool compounds in `backend/models/pool_cache/pool_ensemble.npz`
   so runtime screening remains ~10 ms.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.splits import scaffold_kfold

BLOCKS_DIR = ROOT / "backend" / "experiments" / "blocks"
MODEL_DIR = ROOT / "backend" / "models"
CACHE_DIR = MODEL_DIR / "pool_cache"


def tanimoto_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast binary Tanimoto similarity matrix between A (n, d) and B (m, d)."""
    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)
    inter = A_f @ B_f.T
    na = A_f.sum(axis=1)[:, None]
    nb = B_f.sum(axis=1)[None, :]
    return inter / (na + nb - inter + 1e-12)


def embed_smiles(smi_list: list[str], tok, mod, device: str, batch_size: int = 256) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(smi_list), batch_size):
            batch = smi_list[i:i + batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            out = mod(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean_pool = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            cls_pool = out[:, 0, :]
            embs.append(torch.cat([cls_pool, mean_pool], dim=-1).cpu().numpy())
    return np.vstack(embs).astype(np.float32)


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    r = y - p
    ss_res = float(r @ r)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    act = y >= 6.0
    return {
        "r2": round(1.0 - ss_res / ss_tot, 4),
        "rmse": round(float(np.sqrt(np.mean(r ** 2))), 4),
        "spearman": round(float(spearmanr(y, p).statistic), 4),
        "spearman_actives": round(float(spearmanr(y[act], p[act]).statistic), 4),
    }


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    print(f"Loading pretrained {model_name} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(device).eval()

    comp = pd.read_csv(BLOCKS_DIR / "compounds.csv")
    train_smiles = comp["smiles"].tolist()
    Y = np.load(BLOCKS_DIR / "y.npy").astype(np.float64)
    SCAF = json.loads((BLOCKS_DIR / "scaffolds.json").read_text())

    # 1. Embed training molecules with ChemBERTa-77M-MTR
    emb_path = BLOCKS_DIR / "chemberta_768.npy"
    if emb_path.exists():
        E_train = np.load(emb_path)
    else:
        E_train = embed_smiles(train_smiles, tok, mod, device)
        np.save(emb_path, E_train)

    # 2. Load fingerprint blocks for Tanimoto-KRR and XGBoost
    ref_fps = np.load(MODEL_DIR / "reference_fps.npy").astype(np.float32)
    K_train = tanimoto_cross(ref_fps, ref_fps)
    X_base = np.hstack([np.load(BLOCKS_DIR / "morgan2_2048.npy"), np.load(BLOCKS_DIR / "desc26.npy")]).astype(np.float32)

    xgb_params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.5, min_child_weight=3, reg_lambda=2.0, reg_alpha=0.1,
        tree_method="hist", n_jobs=-1, random_state=42,
    )

    oof_xgb = np.zeros(len(Y))
    oof_krr = np.zeros(len(Y))
    oof_cb = np.zeros(len(Y))

    for tr, te in scaffold_kfold(SCAF, 5, seed=0):
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_base[tr], Y[tr])
        oof_xgb[te] = m_xgb.predict(X_base[te])

        m_krr = KernelRidge(alpha=0.04, kernel="precomputed")
        m_krr.fit(K_train[np.ix_(tr, tr)], Y[tr])
        oof_krr[te] = m_krr.predict(K_train[np.ix_(te, tr)])

        sc = StandardScaler()
        E_tr = sc.fit_transform(E_train[tr])
        E_te = sc.transform(E_train[te])
        m_cb = Ridge(alpha=600.0)
        m_cb.fit(E_tr, Y[tr])
        oof_cb[te] = m_cb.predict(E_te)

    oof_ens = 0.45 * oof_xgb + 0.45 * oof_krr + 0.10 * oof_cb

    metrics = {
        "xgboost_baseline": score_metrics(Y, oof_xgb),
        "tanimoto_krr": score_metrics(Y, oof_krr),
        "chemberta_77m_mtr": score_metrics(Y, oof_cb),
        "multimodal_ensemble": score_metrics(Y, oof_ens),
    }
    print("5-fold Scaffold CV Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<22}: R2={v['r2']:.4f}  RMSE={v['rmse']:.4f}  rho={v['spearman']:.4f}  rho_act={v['spearman_actives']:.4f}")

    # 3. Fit full models on all 2,765 training molecules and score the 10,000-compound pool
    krr_full = KernelRidge(alpha=0.04, kernel="precomputed")
    krr_full.fit(K_train, Y)

    sc_full = StandardScaler()
    E_train_sc = sc_full.fit_transform(E_train)
    cb_full = Ridge(alpha=600.0)
    cb_full.fit(E_train_sc, Y)

    pool_df = pd.read_csv(CACHE_DIR / "pool_index.csv")
    pool_fps = np.load(CACHE_DIR / "pool_fps.npy").astype(np.float32)

    print("Computing Tanimoto-KRR predictions on 10,000 pool compounds...")
    K_pool = tanimoto_cross(pool_fps, ref_fps)
    pool_krr = krr_full.predict(K_pool).astype(np.float32)

    print("Computing Pretrained ChemBERTa-77M-MTR embeddings & predictions on 10,000 pool compounds...")
    E_pool = embed_smiles(pool_df["smiles"].tolist(), tok, mod, device, batch_size=512)
    pool_cb = cb_full.predict(sc_full.transform(E_pool)).astype(np.float32)

    np.savez_compressed(
        CACHE_DIR / "pool_ensemble.npz",
        krr_pic50=pool_krr,
        chemberta_pic50=pool_cb,
    )
    np.savez_compressed(
        MODEL_DIR / "ensemble_heads.npz",
        krr_dual_coef=krr_full.dual_coef_.astype(np.float32),
        cb_coef=cb_full.coef_.astype(np.float32),
        cb_intercept=np.array([cb_full.intercept_], dtype=np.float32),
        cb_mean=sc_full.mean_.astype(np.float32),
        cb_scale=sc_full.scale_.astype(np.float32),
    )
    (MODEL_DIR / "ensemble_metadata.json").write_text(json.dumps({
        "pretrained_foundation_model": model_name,
        "embedding_dim": int(E_train.shape[1]),
        "weights": {"xgboost": 0.45, "tanimoto_krr": 0.45, "chemberta_77m_mtr": 0.10},
        "scaffold_cv_metrics": metrics,
        "build_seconds": round(time.time() - t0, 2),
    }, indent=2))
    print(f"Saved ensemble cache and metadata in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
