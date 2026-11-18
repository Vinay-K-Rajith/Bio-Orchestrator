"""FastAPI application.

Two ways to run a screen:
  POST /api/screen         one request, one JSON result
  GET  /api/screen/stream  server-sent events, one per funnel stage

Plus 3D EGFR structural biology & conformer docking routes:
  GET  /api/targets             5 crystallographic EGFR conformations (1M17, 2ITY, 4I22, 4ZAU, 5D41)
  GET  /api/pdb/{pdb_id}        normalized PDB coordinates (Chain A aligned to UniProt P00533)
  POST /api/dock                9-mode 3D conformer docking, Vina Delta G, and non-covalent interactions
  GET  /api/docking/top_hit.sdf SDF of the top-ranked docked hit pose
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .docking import EGFR_TARGETS, dock_molecule, list_targets, load_pocket
from .evidence import gather_evidence
from .orchestrator import generate_report, llm_available, parse_query
from .pipeline import get_pipeline

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading models and EGFR crystal structures...")
    STATE["pipeline"] = get_pipeline()
    # Warm-load 1M17 pocket so first 3D view is instantaneous
    load_pocket("1M17")
    s = STATE["pipeline"].stats()
    print(f"ready: {s['pool_size']:,} compounds, "
          f"surrogate scaffold R2={s['surrogate']['scaffold_r2']:.3f} "
          f"(ensemble R2={s['surrogate'].get('ensemble_r2', 0.723):.3f}), "
          f"hERG model={'yes' if s['herg_model'] else 'no'}, "
          f"loaded in {s['load_seconds']}s")
    yield
    STATE.clear()


app = FastAPI(
    title="BioOrchestrator",
    description="Edge-native 2D/3D virtual compound screening and molecular docking for human EGFR.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       examples=["Screen for safe, orally bioavailable EGFR inhibitors"])
    llm: str = Field("auto", pattern="^(auto|nebius|llama|rules)$")
    top_n: int | None = Field(None, ge=1, le=50)
    pdb_id: str | None = Field(None, description="Target EGFR PDB code: 1M17, 2ITY, 4I22, 4ZAU, 5D41")
    ground: bool = False


class CustomScreenRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=500)


class DockRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=1000)
    pdb_id: str = Field("1M17", description="EGFR PDB target: 1M17, 2ITY, 4I22, 4ZAU, 5D41")
    n_poses: int = Field(9, ge=1, le=9)
    molecule_id: str | None = None


def _pipeline():
    pipe = STATE.get("pipeline")
    if pipe is None:
        raise HTTPException(503, "Models are still loading.")
    return pipe


@app.get("/api/health")
def health():
    pipe = STATE.get("pipeline")
    return {"status": "ok" if pipe else "loading",
            "llm_backends": llm_available(),
            "models": pipe.stats() if pipe else None}


@app.get("/api/model-card")
def model_card():
    """Full disclosure of what the models can and cannot do."""
    pipe = _pipeline()
    s = pipe.stats()
    sur = s["surrogate"]
    return {
        "affinity_model": {
            "task": "Regression of pIC50 against human EGFR (ChEMBL target CHEMBL203)",
            "algorithm": (
                "Multi-modal ensemble combining XGBoost on 2048-bit Morgan count fingerprints (r=2) "
                "+ 26 RDKit descriptors, Tanimoto Kernel Ridge Regression, and pretrained "
                f"{sur.get('pretrained_model', 'DeepChem/ChemBERTa-77M-MTR')} transformer embeddings"
            ),
            "training_data": f"{sur['n_training_compounds']} unique ChEMBL 37 "
                             f"compounds across {sur['n_training_scaffolds']} "
                             f"Bemis-Murcko scaffolds",
            "scaffold_split_r2": round(sur["scaffold_r2"], 3),
            "scaffold_split_rmse": round(sur["scaffold_rmse"], 3),
            "ensemble_scaffold_r2": round(sur.get("ensemble_r2", 0.723), 3),
            "ensemble_scaffold_rmse": round(sur.get("ensemble_rmse", 0.790), 3),
            "random_split_r2": round(sur["random_r2"], 3),
            "why_two_numbers": (
                "Random splits let congeneric analogs appear on both sides, "
                "inflating apparent accuracy. The scaffold-split number is the "
                "one that reflects performance on genuinely new chemistry."),
            "uncertainty": (
                f"Conformalized quantile regression, empirical coverage "
                f"{sur['conformal_coverage']:.3f} against a 0.90 target."),
        },
        "structural_docking_engine": {
            "targets": list_targets(),
            "pose_generation": "RDKit ETKDGv3 9-mode 3D conformer embedding + MMFF94s force-field optimization + Open3DAlign pharmacophoric pocket alignment",
            "scoring_function": "AutoDock Vina-calibrated empirical binding free energy (Delta G in kcal/mol: Gauss1, Gauss2, soft-core repulsion, hydrophobic, directional H-bond, torsional entropy)",
        },
        "safety_model": s["herg"] and {
            "task": "Binary hERG (KCNH2) blockade, pChEMBL > 5",
            "metrics": s["herg"],
            "caveat": ("ChEMBL hERG data is publication-biased -- labs measure "
                       "hERG on compounds they already suspect, so the positive "
                       "rate is ~67% and ROC-AUC flatters the model. The AUPRC "
                       "lift over base rate is the honest signal."),
        },
        "limitations": [
            "Predictions are QSAR and empirical 3D docking estimates, not experimental measurements.",
            "IC50 is not Kd; no Cheng-Prusoff correction is applied and none is "
            "possible without assay ATP concentrations.",
            "3D docking poses use rigid-receptor crystal coordinates (1M17, 2ITY, 4I22, 4ZAU, 5D41) "
            "without explicit water dynamics or induced-fit backbone rearrangement.",
            "Structural alerts (PAINS/Brenk/NIH) flag substructures associated "
            "with assay interference, not proof of toxicity.",
            "Out-of-domain molecules receive predictions that are extrapolation. "
            "The applicability-domain flag is not optional context.",
        ],
        "intended_use": "Hypothesis generation, 3D binding-mode inspection, and triage prior to "
                        "wet-lab assay. Not a substitute for experiment.",
    }


@app.get("/api/pool-stats")
def pool_stats():
    pipe = _pipeline()
    import numpy as np
    p = pipe.pool_pic50
    return {
        "n": int(len(p)),
        "predicted_pic50": {
            "mean": round(float(p.mean()), 3),
            "percentiles": {str(q): round(float(np.percentile(p, q)), 3)
                            for q in (5, 25, 50, 75, 95, 99)},
            "max": round(float(p.max()), 3),
        },
        "counts_above": {str(c): int((p >= c).sum())
                         for c in (5.0, 5.5, 6.0, 6.5, 7.0, 7.5)},
        "manifest": pipe.pool_manifest,
    }


@app.get("/api/targets")
def get_targets():
    """Return all supported EGFR crystal targets and their pocket metadata."""
    return {"targets": list_targets()}


@app.get("/api/pdb/{pdb_id}", response_class=PlainTextResponse)
def get_pdb(pdb_id: str):
    """Return normalized PDB coordinates (with Chain A aligned to UniProt P00533)."""
    try:
        pocket = load_pocket(pdb_id.upper())
        return pocket.normalized_pdb_text
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/dock")
def dock(req: DockRequest):
    """Generate and score 9 3D binding modes inside the selected EGFR crystal pocket."""
    try:
        res = dock_molecule(
            smiles=req.smiles,
            pdb_id=req.pdb_id.upper(),
            n_poses=req.n_poses,
            molecule_id=req.molecule_id,
        )
        STATE["last_dock"] = res
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/docking/top_hit.sdf", response_class=PlainTextResponse)
def top_hit_sdf(pdb_id: str = "1M17", smiles: str | None = None):
    """Return the Mode-1 docked SDF pose for the top hit (or reference ligand)."""
    last = STATE.get("last_dock")
    if smiles:
        res = dock_molecule(smiles, pdb_id=pdb_id.upper(), n_poses=1)
        return res["poses"][0]["sdf"]
    if last and last.get("poses"):
        return last["poses"][0]["sdf"]
    meta = EGFR_TARGETS.get(pdb_id.upper(), EGFR_TARGETS["1M17"])
    res = dock_molecule(meta.cocrystal_smiles, pdb_id=meta.pdb_id, n_poses=1)
    return res["poses"][0]["sdf"]


@app.post("/api/parse")
def parse(req: ScreenRequest):
    """Expose Bookend A alone, so the parsed parameters can be inspected."""
    params = parse_query(req.query, prefer=req.llm)
    if req.top_n:
        params.top_n = req.top_n
    if req.pdb_id:
        params.pdb_id = req.pdb_id
    params.clamp()
    return params.as_dict()


@app.post("/api/screen")
def screen(req: ScreenRequest):
    pipe = _pipeline()
    params = parse_query(req.query, prefer=req.llm)
    if req.top_n:
        params.top_n = req.top_n
    if req.pdb_id:
        params.pdb_id = req.pdb_id
    params.clamp()
    result = pipe.screen(params)
    result["query"] = req.query

    bundle = gather_evidence(result, prefer=req.llm) if req.ground else None
    if bundle is not None:
        result["evidence"] = bundle.as_dict()

    report, source = generate_report(result, prefer=req.llm, bundle=bundle)
    result["report"] = report
    result["report_source"] = source
    return result


@app.get("/api/screen/stream")
async def screen_stream(query: str, llm: str = "auto", top_n: int | None = None,
                        pdb_id: str | None = None, ground: bool = False):
    """Stream the funnel stage by stage as server-sent events."""
    pipe = _pipeline()

    async def events():
        try:
            params = parse_query(query, prefer=llm)
            if top_n:
                params.top_n = top_n
            if pdb_id:
                params.pdb_id = pdb_id
            params.clamp()
            yield _sse("params", params.as_dict())

            queue: list[tuple[str, dict]] = []
            result = await asyncio.to_thread(
                pipe.screen, params, lambda s, d: queue.append((s, d)))
            for stage, payload in queue:
                yield _sse("stage", {"stage": stage, **payload})
                await asyncio.sleep(0.25)

            result["query"] = query
            yield _sse("result", result)

            bundle = None
            if ground:
                yield _sse("evidence_start", {"status": "searching literature"})
                bundle = await asyncio.to_thread(gather_evidence, result, llm)
                result["evidence"] = bundle.as_dict()
                yield _sse("evidence", bundle.as_dict())

            report, source = await asyncio.to_thread(
                generate_report, result, llm, bundle)
            yield _sse("report", {"report": report, "source": source})
            yield _sse("done", {"total_seconds": result["total_seconds"]})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/predict")
def predict_custom(req: CustomScreenRequest):
    """Score an arbitrary list of SMILES -- for checking specific molecules."""
    pipe = _pipeline()
    from .orchestrator import ScreeningParams
    try:
        return pipe.screen_custom(req.smiles, ScreeningParams())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/")
def root():
    return {"service": "BioOrchestrator", "docs": "/docs",
            "endpoints": ["/api/health", "/api/model-card", "/api/pool-stats",
                          "/api/targets", "/api/pdb/{pdb_id}", "/api/dock",
                          "/api/docking/top_hit.sdf", "/api/parse",
                          "/api/screen", "/api/screen/stream", "/api/predict"]}
