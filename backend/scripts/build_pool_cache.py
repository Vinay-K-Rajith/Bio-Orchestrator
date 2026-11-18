"""Precompute the screening pool into a memory-mappable cache.

Measured on this machine: featurizing 10,000 SMILES costs ~29s, while XGBoost
inference over the same 10,000 costs ~0.14s. RDKit parsing, not the model, is
the bottleneck -- so the plan's "<1 second for 10,000 compounds" is only
achievable if parsing happens once, offline, rather than per request.

This script does that parsing once and writes:
  pool_features.npy  float32 (N, 2074)  model input
  pool_fps.npy       uint8   (N, 2048)  binary fingerprints for Tanimoto
  pool_profiles.json                    physicochemical + alert profiles
  pool_index.csv                        id / name / SMILES, row-aligned

Every array is row-aligned with pool_index.csv. Nothing downstream may reorder
one without the others.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.bio_engine import ApplicabilityDomain, annotate_domain  # noqa: E402
from backend.app.featurize import (  # noqa: E402
    FEATURE_VERSION, featurize_mol, mol_from_smiles, morgan_bits,
)
from backend.app.bio_engine import profile_molecule  # noqa: E402

DATA_DIR = ROOT / "backend" / "data"
CACHE_DIR = ROOT / "backend" / "models" / "pool_cache"
POOL_CSV = DATA_DIR / "candidate_pool.csv"


def main() -> int:
    t0 = time.time()
    if not POOL_CSV.exists():
        print(f"ERROR: {POOL_CSV} missing.\n"
              "Run first: python backend/scripts/fetch_chembl.py pool")
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(POOL_CSV)
    print(f"=== featurizing {len(df)} candidates (one-time cost) ===")

    feats, bits, profiles, rows = [], [], [], []
    for i, rec in enumerate(df.itertuples(index=False)):
        mol = mol_from_smiles(rec.smiles)
        if mol is None:
            continue
        prof = profile_molecule(rec.smiles, mol)
        if prof is None:
            continue
        feats.append(featurize_mol(mol))
        bits.append(morgan_bits(mol))
        profiles.append(prof)
        rows.append({"molecule_id": rec.molecule_id,
                     "name": getattr(rec, "name", "") or "",
                     "smiles": prof.smiles})
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(df)}  ({rate:.0f} mol/s)", flush=True)

    X = np.vstack(feats).astype(np.float32)
    fps = np.vstack(bits).astype(np.uint8)
    index = pd.DataFrame(rows)
    print(f"  parsed {len(index)} / {len(df)} "
          f"({len(df) - len(index)} rejected) in {time.time() - t0:.1f}s")

    print("\n=== applicability domain vs EGFR training set ===")
    domain = ApplicabilityDomain.load()
    annotate_domain(profiles, fps, domain)
    sims = np.array([p.max_similarity for p in profiles])
    n_in = int(sum(p.in_domain for p in profiles))
    print(f"  max-Tanimoto: mean={sims.mean():.3f} median={np.median(sims):.3f} "
          f"p95={np.percentile(sims, 95):.3f}")
    print(f"  in-domain (>=0.35): {n_in} / {len(profiles)} ({n_in / len(profiles):.1%})")

    np.save(CACHE_DIR / "pool_features.npy", X)
    np.save(CACHE_DIR / "pool_fps.npy", fps)
    index.to_csv(CACHE_DIR / "pool_index.csv", index=False)
    (CACHE_DIR / "pool_profiles.json").write_text(
        json.dumps([p.as_dict() for p in profiles]))

    alerts = np.array([p.n_alerts for p in profiles])
    ro5 = np.array([p.lipinski_violations for p in profiles])
    manifest = {
        "feature_version": FEATURE_VERSION,
        "n_candidates": int(len(index)),
        "n_rejected": int(len(df) - len(index)),
        "feature_shape": list(X.shape),
        "in_domain_fraction": n_in / len(profiles),
        "similarity_stats": {"mean": float(sims.mean()),
                             "median": float(np.median(sims)),
                             "p95": float(np.percentile(sims, 95))},
        "alert_free_fraction": float((alerts == 0).mean()),
        "ro5_clean_fraction": float((ro5 == 0).mean()),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  alert-free: {(alerts == 0).mean():.1%}   "
          f"zero Ro5 violations: {(ro5 == 0).mean():.1%}")
    print(f"\n=== cache built in {time.time() - t0:.1f}s -> {CACHE_DIR} ===")
    for f in sorted(CACHE_DIR.iterdir()):
        print(f"    {f.name:<24} {f.stat().st_size / 1024 / 1024:7.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
