"""End-to-end verification.

Checks the claims this system makes about itself. Every assertion is a property
that would matter if it broke: leakage in the splits, a miscalibrated interval,
a ranker that ignores its weights, a funnel that silently drops molecules, and
the 3D EGFR docking & multi-modal ensemble pipeline.

Run: python backend/scripts/verify.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = ""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    t0 = time.time()

    print("\n=== featurization ===")
    from backend.app.featurize import (N_FEATURES, canonical_smiles,
                                       featurize_many, murcko_scaffold)
    X, fps, kept = featurize_many(["CCO", "c1ccccc1", "!!!invalid!!!", "CC(=O)O"])
    check("invalid SMILES dropped, indices tracked", kept == [0, 1, 3], f"kept={kept}")
    check("feature width is stable", X.shape[1] == N_FEATURES, f"{X.shape[1]}")
    check("fingerprints are binary", set(np.unique(fps)) <= {0, 1})
    a = canonical_smiles("C(C)O")
    b = canonical_smiles("CCO")
    check("canonicalisation is order-independent", a == b, f"{a} == {b}")
    check("scaffold strips side chains",
          murcko_scaffold("CCCCc1ccccc1") == "c1ccccc1")

    print("\n=== splits (leakage) ===")
    import pandas as pd
    from backend.app.splits import scaffold_kfold, scaffold_split
    df = pd.read_csv(ROOT / "backend" / "data" / "egfr_chembl_labeled.csv")
    scafs = np.array([murcko_scaffold(s) for s in df["smiles"]])
    tr, te = scaffold_split(scafs, 0.15, seed=0)
    overlap = set(scafs[tr]) & set(scafs[te])
    check("scaffold split leaks no scaffold", len(overlap) == 0,
          f"{len(overlap)} shared")
    check("split covers every molecule", len(tr) + len(te) == len(scafs))
    check("test fraction is respected", len(te) / len(scafs) <= 0.16,
          f"{len(te) / len(scafs):.3f}")

    for name, synthetic in [
        ("two large scaffolds", ["A"] * 90 + ["B"] * 10),
        ("one dominant scaffold", ["A"] * 99 + ["B"]),
        ("all singletons", [str(i) for i in range(100)]),
        ("a single scaffold", ["A"] * 50),
    ]:
        s = np.array(synthetic)
        a, b = scaffold_split(s, 0.15, seed=0)
        ok = (len(a) > 0
              and len(a) + len(b) == len(s)
              and len(b) / len(s) <= 0.16
              and not (set(s[a]) & set(s[b]) if len(b) else set()))
        check(f"split never inverts: {name}", ok,
              f"train={len(a)} test={len(b)}")
    for i, (ktr, kte) in enumerate(scaffold_kfold(scafs, 5, seed=0)):
        ov = set(scafs[ktr]) & set(scafs[kte])
        check(f"kfold {i + 1} leaks no scaffold", len(ov) == 0, f"{len(ov)} shared")

    print("\n=== affinity model ===")
    from backend.app.predictor import AffinityPredictor, pic50_to_ic50_nm
    pred = AffinityPredictor()
    check("scaffold R2 is reported and positive", pred.scaffold_r2 > 0.5,
          f"R2={pred.scaffold_r2:.3f}")
    check("conformal coverage is near nominal",
          abs(pred.coverage - 0.90) < 0.03, f"coverage={pred.coverage:.3f}")
    check("pIC50 9 converts to 1 nM", abs(pic50_to_ic50_nm(9.0) - 1.0) < 1e-6)
    check("pIC50 6 converts to 1000 nM", abs(pic50_to_ic50_nm(6.0) - 1000.0) < 1e-3)

    imatinib = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
    r, _ = pred.predict_smiles([imatinib])
    check("imatinib recovers its known-weak EGFR label",
          abs(r["pic50"][0] - 4.0) < 1.0, f"predicted {r['pic50'][0]:.2f} vs actual 4.0")
    check("interval brackets the point estimate",
          r["pic50_lo"][0] <= r["pic50"][0] <= r["pic50_hi"][0])
    check("IC50 bounds are inverted correctly",
          r["ic50_nm_lo"][0] < r["ic50_nm"][0] < r["ic50_nm_hi"][0])

    print("\n=== chemistry engine ===")
    from backend.app.bio_engine import profile_molecule
    asp = profile_molecule("CC(=O)Oc1ccccc1C(=O)O")
    check("aspirin MW is correct", abs(asp.mw - 180.16) < 0.1, f"{asp.mw}")
    check("aspirin passes Lipinski", asp.lipinski_violations == 0)
    azo = profile_molecule("O=C(O)c1ccccc1N=Nc1ccc(O)cc1")
    check("azo dye trips structural alerts", azo.n_alerts >= 2,
          f"{azo.n_alerts} alerts")
    big = profile_molecule("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O")
    check("very lipophilic chain violates Ro5", big.lipinski_violations >= 1,
          f"LogP={big.logp}")

    print("\n=== ranker ===")
    from backend.app.ranker import (Objectives, Weights, non_dominated_sort,
                                    rank_candidates)
    dominated = np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    f = non_dominated_sort(dominated)
    check("clear domination is detected", f[0] == 0 and f[1] == 1, f"fronts={f.tolist()}")
    tie = non_dominated_sort(np.array([[1.0, 0.0], [0.0, 1.0]]))
    check("trade-offs share the front", tie[0] == tie[1] == 0)

    toxic = Objectives(affinity=0.95, confidence=0.9, safety=0.05, drug_likeness=0.2)
    balanced = Objectives(affinity=0.70, confidence=0.8, safety=0.90, drug_likeness=0.85)
    objs = [toxic, balanced]
    safety_first = rank_candidates(
        objs, Weights(affinity=0.1, confidence=0.1, safety=0.7, drug_likeness=0.1), 2)
    check("safety weighting demotes the toxic molecule",
          safety_first[0].index == 1, f"winner index {safety_first[0].index}")
    potency_first = rank_candidates(
        objs, Weights(affinity=0.85, confidence=0.05, safety=0.05, drug_likeness=0.05), 2)
    check("potency weighting promotes the potent molecule",
          potency_first[0].index == 0, f"winner index {potency_first[0].index}")

    print("\n=== orchestrator ===")
    from backend.app.orchestrator import _rule_based_parse
    p = _rule_based_parse("Screen for safe HER2 hits")
    check("'HER2' is not misread as 'top 2'", p.top_n == 5, f"top_n={p.top_n}")
    check("'safe' raises the safety weight", p.weight_safety > 0.35,
          f"{p.weight_safety:.2f}")
    p2 = _rule_based_parse("very potent, I don't care about toxicity")
    check("explicit dismissal lowers the safety weight", p2.weight_safety < 0.2,
          f"{p2.weight_safety:.2f}")
    p3 = _rule_based_parse("inhibitors under 25 nM, top 8 leads")
    check("explicit cutoff is parsed", p3.max_ic50_nm == 25.0, f"{p3.max_ic50_nm}")
    check("explicit top_n is parsed", p3.top_n == 8, f"{p3.top_n}")
    p4 = _rule_based_parse("anything at all")
    check("weights always normalise to 1",
          abs(p4.weight_affinity + p4.weight_confidence + p4.weight_safety
              + p4.weight_druglikeness - 1.0) < 1e-9)

    print("\n=== pipeline ===")
    from backend.app.pipeline import get_pipeline
    pipe = get_pipeline()
    n = len(pipe.pool_index)
    check("pool arrays stay row-aligned",
          len(pipe.pool_X) == len(pipe.pool_fps) == len(pipe.pool_profiles) == n,
          f"n={n}")

    from backend.app.orchestrator import ScreeningParams
    params = ScreeningParams(max_ic50_nm=5000.0, top_n=5).clamp()
    t = time.time()
    res = pipe.screen(params)
    elapsed = time.time() - t
    fn = res["funnel"]
    check("funnel is monotonically narrowing",
          fn["pool"] >= fn["potent"] >= fn["safe"] >= fn["leads"],
          f"{fn['pool']} -> {fn['potent']} -> {fn['safe']} -> {fn['leads']}")
    check("screen completes under 2 s", elapsed < 2.0, f"{elapsed * 1000:.0f} ms")
    check("leads are returned", fn["leads"] > 0)
    check("leads are rank-ordered",
          [l["rank"] for l in res["leads"]] == list(range(1, len(res["leads"]) + 1)))
    check("Pareto fronts never worsen down the ranking",
          all(res["leads"][i]["pareto_front"] <= res["leads"][i + 1]["pareto_front"]
              for i in range(len(res["leads"]) - 1)))
    check("every out-of-domain lead is flagged",
          all(any("applicability domain" in f for f in l["flags"])
              for l in res["leads"] if not l["in_domain"]))

    strict = ScreeningParams(max_ic50_nm=5000.0, require_in_domain=True,
                             min_similarity=0.35, top_n=5).clamp()
    strict_res = pipe.screen(strict)
    check("require_in_domain admits only in-domain leads",
          all(l["in_domain"] for l in strict_res["leads"]),
          f"{len(strict_res['leads'])} leads")

    print("\n=== report ===")
    from backend.app.orchestrator import _deterministic_report
    rep = _deterministic_report(res)
    check("report states these are predictions",
          "not experimental measurements" in rep)
    check("report recommends a next step", "docking" in rep.lower())
    check("report is substantial", len(rep) > 200, f"{len(rep)} chars")

    print("\n=== 3D EGFR docking & multi-modal ensemble ===")
    from backend.app.docking import EGFR_TARGETS, dock_molecule, load_pocket
    check("all 5 EGFR crystal conformations configured",
          set(EGFR_TARGETS.keys()) == {"1M17", "2ITY", "4I22", "4ZAU", "5D41"})
    p_1m17 = load_pocket("1M17")
    check("1M17 chain A normalized to canonical Thr790/Met793/Cys797",
          all(len(p_1m17.key_residue_atoms.get(r, [])) > 0 for r in (790, 793, 797)),
          f"pocket atoms={len(p_1m17.pocket_atoms)}")
    d_res = dock_molecule(
        EGFR_TARGETS["1M17"].cocrystal_smiles, pdb_id="1M17", n_poses=9, molecule_id="Erlotinib"
    )
    check("9 binding modes generated and sorted by Delta G",
          d_res["n_poses"] == 9
          and d_res["poses"][0]["delta_g_kcal"] <= d_res["poses"][-1]["delta_g_kcal"],
          f"best dG={d_res['best_delta_g_kcal']} kcal/mol")
    check("non-covalent 3D interactions extracted",
          len(d_res["poses"][0]["interactions"]) > 0,
          f"{len(d_res['poses'][0]['interactions'])} contacts")
    check("leads include 3D docking score, scaffold, and ensemble pIC50",
          all("docking_score_kcal" in l and "scaffold" in l and "ensemble_pic50" in l
              for l in res["leads"]))

    total = len(PASS) + len(FAIL)
    print(f"\n{'=' * 60}")
    print(f"  {len(PASS)}/{total} checks passed in {time.time() - t0:.1f}s")
    if FAIL:
        print(f"  FAILED: {', '.join(FAIL)}")
    print(f"{'=' * 60}\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
