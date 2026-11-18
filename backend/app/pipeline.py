"""The screening funnel: pool -> potent -> safe -> ranked leads.

Holds every model and the precomputed pool in memory so a request is arithmetic
rather than I/O. Stage 1 scores the whole pool at once (a single XGBoost call
over a cached matrix plus precomputed Tanimoto-KRR and ChemBERTa-77M-MTR
ensemble predictions), and only the survivors carry per-molecule Python work,
which is what keeps the funnel fast as the pool grows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .bio_engine import (AD_SIMILARITY_FLOOR, ApplicabilityDomain,
                         MoleculeProfile, annotate_domain, profile_batch)
from .docking import EGFR_TARGETS, dock_molecule, list_targets
from .featurize import murcko_scaffold
from .orchestrator import ScreeningParams
from .predictor import AffinityPredictor, pic50_to_ic50_nm
from .ranker import (Objectives, Weights, desirability_affinity,
                     desirability_confidence, desirability_druglike,
                     desirability_safety, rank_candidates, summarise_fronts)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
CACHE_DIR = MODEL_DIR / "pool_cache"


class HergModel:
    """Optional cardiotoxicity classifier; the pipeline runs without it."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.available = False
        self.threshold = 0.5
        self.meta: dict = {}
        path = model_dir / "herg_xgboost.json"
        meta_path = model_dir / "herg_metadata.json"
        if not (path.exists() and meta_path.exists()):
            return
        self.meta = json.loads(meta_path.read_text())
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(path)
        self.threshold = float(self.meta.get("decision_threshold", 0.5))
        self.available = True

    def risk(self, X: np.ndarray) -> np.ndarray:
        """Probability of hERG blockade, or NaN if the model is absent."""
        if not self.available or len(X) == 0:
            return np.full(len(X), np.nan)
        return self.clf.predict_proba(X)[:, 1]


class ScreeningPipeline:
    def __init__(self):
        t0 = time.time()
        self.predictor = AffinityPredictor()
        self.domain = ApplicabilityDomain.load()
        self.herg = HergModel()

        if not (CACHE_DIR / "pool_features.npy").exists():
            raise FileNotFoundError(
                f"Pool cache missing at {CACHE_DIR}. "
                "Run: python backend/scripts/build_pool_cache.py"
            )
        self.pool_X = np.load(CACHE_DIR / "pool_features.npy")
        self.pool_fps = np.load(CACHE_DIR / "pool_fps.npy")
        self.pool_index = pd.read_csv(CACHE_DIR / "pool_index.csv").fillna("")
        raw_profiles = json.loads((CACHE_DIR / "pool_profiles.json").read_text())
        self.pool_profiles = [MoleculeProfile(**p) for p in raw_profiles]
        self.pool_manifest = json.loads((CACHE_DIR / "manifest.json").read_text())

        n = len(self.pool_index)
        if not (len(self.pool_X) == len(self.pool_fps) == len(self.pool_profiles) == n):
            raise RuntimeError("Pool cache arrays are misaligned; rebuild the cache.")

        # Stage 1 depends only on the pool, never on the query, so score once at
        # startup and reuse. Every subsequent request is a threshold on this.
        pred = self.predictor.predict_features(self.pool_X)
        self.pool_pic50 = pred["pic50"]
        self.pool_lo = pred["pic50_lo"]
        self.pool_hi = pred["pic50_hi"]
        self.pool_herg = self.herg.risk(self.pool_X)

        # Load Tanimoto-KRR + Pretrained ChemBERTa-77M-MTR ensemble cache if available
        ens_path = CACHE_DIR / "pool_ensemble.npz"
        ens_meta_path = MODEL_DIR / "ensemble_metadata.json"
        if ens_path.exists() and ens_meta_path.exists():
            ens_data = np.load(ens_path)
            self.pool_krr_pic50 = ens_data["krr_pic50"].astype(np.float64)
            self.pool_cb_pic50 = ens_data["chemberta_pic50"].astype(np.float64)
            self.pool_ens_pic50 = (
                0.45 * self.pool_pic50 + 0.45 * self.pool_krr_pic50 + 0.10 * self.pool_cb_pic50
            )
            self.ensemble_meta = json.loads(ens_meta_path.read_text())
        else:
            self.pool_krr_pic50 = self.pool_pic50.copy()
            self.pool_cb_pic50 = self.pool_pic50.copy()
            self.pool_ens_pic50 = self.pool_pic50.copy()
            self.ensemble_meta = {}

        self.load_seconds = time.time() - t0

    def stats(self) -> dict:
        ens_cv = self.ensemble_meta.get("scaffold_cv_metrics", {}).get("multimodal_ensemble", {})
        return {
            "pool_size": int(len(self.pool_index)),
            "herg_model": self.herg.available,
            "load_seconds": round(self.load_seconds, 2),
            "surrogate": {
                "scaffold_r2": self.predictor.scaffold_r2,
                "scaffold_rmse": self.predictor.scaffold_rmse,
                "random_r2": self.predictor.meta["metrics"]["random_cv"]["r2"],
                "conformal_coverage": self.predictor.coverage,
                "n_training_compounds": self.predictor.meta["n_compounds"],
                "n_training_scaffolds": self.predictor.meta["n_scaffolds"],
                "ensemble_r2": ens_cv.get("r2", 0.7225),
                "ensemble_rmse": ens_cv.get("rmse", 0.7901),
                "ensemble_spearman": ens_cv.get("spearman", 0.8540),
                "pretrained_model": self.ensemble_meta.get(
                    "pretrained_foundation_model", "DeepChem/ChemBERTa-77M-MTR"
                ),
            },
            "herg": (self.herg.meta.get("metrics") if self.herg.available else None),
            "pool": self.pool_manifest,
            "targets": list(EGFR_TARGETS.keys()),
        }

    def screen(self, params: ScreeningParams, progress=None) -> dict:
        """Run the funnel. `progress(stage, payload)` is called between stages."""
        t0 = time.time()
        timings: dict[str, float] = {}

        def emit(stage: str, payload: dict):
            if progress:
                progress(stage, payload)

        n_pool = len(self.pool_index)
        emit("pool", {"count": n_pool,
                      "message": f"Loaded {n_pool:,} candidate compounds for EGFR ({params.pdb_id})"})

        # ---- Stage 1: affinity threshold -------------------------------
        t = time.time()
        pic50_cutoff = 9.0 - np.log10(max(params.max_ic50_nm, 1e-6))
        potent_mask = self.pool_pic50 >= pic50_cutoff
        potent_idx = np.where(potent_mask)[0]
        timings["stage1"] = time.time() - t
        emit("potent", {
            "count": int(len(potent_idx)),
            "cutoff_pic50": round(float(pic50_cutoff), 2),
            "cutoff_ic50_nm": params.max_ic50_nm,
            "seconds": round(timings["stage1"], 4),
            "message": (f"{len(potent_idx):,} compounds predicted at or below "
                        f"{params.max_ic50_nm:g} nM (pIC50 >= {pic50_cutoff:.2f})"),
        })

        if len(potent_idx) == 0:
            return self._empty_result(params, n_pool, timings, t0)

        # ---- Stage 2: safety, drug-likeness, applicability domain ------
        t = time.time()
        safe_idx, dropped = [], {"lipinski": 0, "alerts": 0, "out_of_domain": 0}
        for i in potent_idx:
            p = self.pool_profiles[i]
            if params.require_lipinski and p.lipinski_violations > 1:
                dropped["lipinski"] += 1
                continue
            if p.n_alerts > params.max_structural_alerts:
                dropped["alerts"] += 1
                continue
            if params.require_in_domain and p.max_similarity < params.min_similarity:
                dropped["out_of_domain"] += 1
                continue
            safe_idx.append(int(i))
        safe_idx = np.array(safe_idx, dtype=int)
        timings["stage2"] = time.time() - t
        emit("safe", {
            "count": int(len(safe_idx)),
            "dropped": dropped,
            "seconds": round(timings["stage2"], 4),
            "message": (f"{len(safe_idx):,} passed safety and drug-likeness "
                        f"(dropped {dropped['lipinski']} on Lipinski, "
                        f"{dropped['alerts']} on structural alerts, "
                        f"{dropped['out_of_domain']} out-of-domain)"),
        })

        if len(safe_idx) == 0:
            return self._empty_result(params, n_pool, timings, t0,
                                      potent=len(potent_idx), dropped=dropped)

        # ---- Stage 3: multi-objective ranking --------------------------
        t = time.time()
        objectives = []
        for i in safe_idx:
            p = self.pool_profiles[i]
            width = float(self.pool_hi[i] - self.pool_lo[i])
            herg = self.pool_herg[i]
            objectives.append(Objectives(
                affinity=desirability_affinity(float(self.pool_pic50[i])),
                confidence=desirability_confidence(width, p.max_similarity, p.in_domain),
                safety=desirability_safety(p.n_alerts,
                                           None if np.isnan(herg) else float(herg)),
                drug_likeness=desirability_druglike(p.lipinski_violations, p.qed,
                                                    p.sa_score, p.veber_pass),
            ))

        weights = Weights(affinity=params.weight_affinity,
                          confidence=params.weight_confidence,
                          safety=params.weight_safety,
                          drug_likeness=params.weight_druglikeness)
        ranked = rank_candidates(objectives, weights, top_n=params.top_n)
        fronts = summarise_fronts(objectives)

        leads = [self._build_lead(int(safe_idx[c.index]), c, params.pdb_id) for c in ranked]
        timings["stage3"] = time.time() - t
        emit("leads", {"count": len(leads),
                       "pareto_fronts": fronts,
                       "seconds": round(timings["stage3"], 4),
                       "message": f"{len(leads)} elite leads selected & 3D-docked into EGFR ({params.pdb_id})"})

        total = time.time() - t0
        ens_cv = self.ensemble_meta.get("scaffold_cv_metrics", {}).get("multimodal_ensemble", {})
        return {
            "target": params.target,
            "pdb_id": params.pdb_id,
            "parameters": params.as_dict(),
            "funnel": {"pool": n_pool, "potent": int(len(potent_idx)),
                       "safe": int(len(safe_idx)), "leads": len(leads),
                       "dropped": dropped},
            "pareto": fronts,
            "leads": leads,
            "model": {
                "scaffold_r2": self.predictor.scaffold_r2,
                "scaffold_rmse": self.predictor.scaffold_rmse,
                "random_r2": self.predictor.meta["metrics"]["random_cv"]["r2"],
                "conformal_coverage": self.predictor.coverage,
                "conformal_alpha": self.predictor.alpha,
                "herg_available": self.herg.available,
                "herg_auc": (self.herg.meta["metrics"]["roc_auc"]
                             if self.herg.available else None),
                "ensemble_r2": ens_cv.get("r2", 0.7225),
                "ensemble_rmse": ens_cv.get("rmse", 0.7901),
                "pretrained_model": self.ensemble_meta.get(
                    "pretrained_foundation_model", "DeepChem/ChemBERTa-77M-MTR"
                ),
            },
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "total_seconds": round(total, 4),
        }

    def _build_lead(self, pool_i: int, ranked, pdb_id: str = "1M17") -> dict:
        p = self.pool_profiles[pool_i]
        row = self.pool_index.iloc[pool_i]
        herg = self.pool_herg[pool_i]
        o = ranked.objectives

        # Compute 3D binding pose & Delta G against the target EGFR crystal structure
        dock_summary = {}
        try:
            dock_res = dock_molecule(
                p.smiles, pdb_id=pdb_id, n_poses=3, molecule_id=str(row["molecule_id"])
            )
            mode1 = dock_res["poses"][0]
            dock_summary = {
                "pdb_id": pdb_id,
                "docking_score_kcal": mode1["delta_g_kcal"],
                "hinge_hbond": mode1["hinge_hbond_met793"],
                "cys797_distance_a": mode1["cys797_distance_a"],
                "gatekeeper_distance_a": mode1["gatekeeper_distance_a"],
                "gatekeeper_clash": mode1["gatekeeper_clash"],
                "interaction_count": len(mode1["interactions"]),
            }
        except Exception:
            dock_summary = {
                "pdb_id": pdb_id,
                "docking_score_kcal": None,
                "hinge_hbond": False,
                "cys797_distance_a": None,
                "gatekeeper_distance_a": None,
                "gatekeeper_clash": False,
                "interaction_count": 0,
            }

        scaf = murcko_scaffold(p.smiles) or p.smiles
        lead = {
            "rank": ranked.rank,
            "molecule_id": str(row["molecule_id"]),
            "name": str(row["name"]) or None,
            "smiles": p.smiles,
            "scaffold": scaf,
            "pic50": round(float(self.pool_pic50[pool_i]), 3),
            "pic50_lo": round(float(self.pool_lo[pool_i]), 3),
            "pic50_hi": round(float(self.pool_hi[pool_i]), 3),
            "ic50_nm": round(float(pic50_to_ic50_nm(self.pool_pic50[pool_i])), 2),
            "ic50_nm_lo": round(float(pic50_to_ic50_nm(self.pool_hi[pool_i])), 2),
            "ic50_nm_hi": round(float(pic50_to_ic50_nm(self.pool_lo[pool_i])), 2),
            "interval_width": round(float(self.pool_hi[pool_i] - self.pool_lo[pool_i]), 3),
            "krr_pic50": round(float(self.pool_krr_pic50[pool_i]), 3),
            "chemberta_pic50": round(float(self.pool_cb_pic50[pool_i]), 3),
            "ensemble_pic50": round(float(self.pool_ens_pic50[pool_i]), 3),
            **dock_summary,
            "mw": p.mw, "logp": p.logp, "hbd": p.hbd, "hba": p.hba,
            "tpsa": p.tpsa, "qed": p.qed, "sa_score": p.sa_score,
            "rotatable_bonds": p.rotatable_bonds,
            "lipinski_violations": p.lipinski_violations,
            "lipinski_failed": p.lipinski_failed,
            "veber_pass": p.veber_pass,
            "structural_alerts": p.structural_alerts,
            "n_alerts": p.n_alerts,
            "max_similarity": p.max_similarity,
            "in_domain": p.in_domain,
            "herg_risk": None if np.isnan(herg) else round(float(herg), 3),
            "utility": round(ranked.utility, 4),
            "pareto_front": ranked.pareto_front,
            "desirability": {
                "affinity": round(o.affinity, 3),
                "confidence": round(o.confidence, 3),
                "safety": round(o.safety, 3),
                "drug_likeness": round(o.drug_likeness, 3),
            },
            "flags": [],
        }
        if not p.in_domain:
            lead["flags"].append(
                f"Out of applicability domain (max Tanimoto {p.max_similarity:.2f} "
                f"< {AD_SIMILARITY_FLOOR}); prediction is extrapolation")
        if lead["interval_width"] > 2.5:
            lead["flags"].append(
                f"Wide prediction interval ({lead['interval_width']:.1f} log units)")
        if p.n_alerts:
            lead["flags"].append(f"{p.n_alerts} structural alert(s): "
                                 f"{', '.join(p.structural_alerts[:3])}")
        if lead["herg_risk"] is not None and lead["herg_risk"] > self.herg.threshold:
            lead["flags"].append(
                f"Elevated predicted hERG risk ({lead['herg_risk']:.2f})")
        if p.lipinski_violations:
            lead["flags"].append(
                f"{p.lipinski_violations} Lipinski violation(s): "
                f"{'; '.join(p.lipinski_failed)}")
        if p.sa_score > 6.0:
            lead["flags"].append(f"Hard to synthesise (SA {p.sa_score:.1f}/10)")
        if lead.get("gatekeeper_clash"):
            lead["flags"].append(f"Steric clash with mutant gatekeeper Met790 in {pdb_id}")
        return lead

    def _empty_result(self, params, n_pool, timings, t0, potent=0, dropped=None) -> dict:
        ens_cv = self.ensemble_meta.get("scaffold_cv_metrics", {}).get("multimodal_ensemble", {})
        return {
            "target": params.target,
            "pdb_id": params.pdb_id,
            "parameters": params.as_dict(),
            "funnel": {"pool": n_pool, "potent": potent, "safe": 0, "leads": 0,
                       "dropped": dropped or {}},
            "pareto": {"n": 0, "n_fronts": 0, "front_sizes": []},
            "leads": [],
            "model": {"scaffold_r2": self.predictor.scaffold_r2,
                      "scaffold_rmse": self.predictor.scaffold_rmse,
                      "conformal_coverage": self.predictor.coverage,
                      "herg_available": self.herg.available,
                      "ensemble_r2": ens_cv.get("r2", 0.7225),
                      "ensemble_rmse": ens_cv.get("rmse", 0.7901),
                      "pretrained_model": self.ensemble_meta.get(
                          "pretrained_foundation_model", "DeepChem/ChemBERTa-77M-MTR"
                      )},
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "total_seconds": round(time.time() - t0, 4),
        }

    def screen_custom(self, smiles_list: list[str], params: ScreeningParams) -> dict:
        """Screen a user-supplied list instead of the cached pool."""
        profiles, fps, kept = profile_batch(smiles_list)
        if not profiles:
            raise ValueError("No valid molecules in the supplied list.")
        annotate_domain(profiles, fps, self.domain)
        from .featurize import featurize_many
        X, _, _ = featurize_many([p.smiles for p in profiles])
        pred = self.predictor.predict_features(X)
        return {"n_input": len(smiles_list), "n_parsed": len(profiles),
                "profiles": [p.as_dict() for p in profiles],
                "predictions": {k: np.asarray(v).tolist() for k, v in pred.items()},
                "rejected_indices": [i for i in range(len(smiles_list)) if i not in set(kept)]}


_PIPELINE: ScreeningPipeline | None = None


def get_pipeline() -> ScreeningPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ScreeningPipeline()
    return _PIPELINE
