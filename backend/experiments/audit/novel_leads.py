"""Novelty run: hunt novel EGFR inhibitor candidates for NSCLC, then audit the
hits against ChEMBL -- are they genuinely novel, and are they actually active?
"""
import sys
from pathlib import Path
import httpx, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from backend.app.orchestrator import ScreeningParams, generate_report
from backend.app.pipeline import get_pipeline
HERE = Path(__file__).parent
BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}

pipe = get_pipeline()
params = ScreeningParams(target="EGFR", max_ic50_nm=1000.0, top_n=8,
                         require_lipinski=True, max_structural_alerts=0,
                         require_in_domain=True,   # only where the model has standing
                         weight_affinity=0.35, weight_confidence=0.25,
                         weight_safety=0.25, weight_druglikeness=0.15).clamp()
res = pipe.screen(params)
f = res["funnel"]
print(f"funnel: {f['pool']:,} -> potent {f['potent']} -> safe {f['safe']} "
      f"-> leads {f['leads']}  ({res['total_seconds']*1000:.0f} ms)")
print(f"        dropped: {f['dropped']}\n")

ids = [l["molecule_id"] for l in res["leads"]]
meta = {}
with httpx.Client(headers=H, timeout=120.0) as c:
    for mid in ids:
        m = c.get(f"{BASE}/molecule/{mid}", params={"format": "json"}).json()
        a = c.get(f"{BASE}/activity", params={"molecule_chembl_id": mid,
            "target_chembl_id": "CHEMBL203", "standard_type": "IC50",
            "format": "json", "limit": 100}).json().get("activities", [])
        vals = [float(x["standard_value"]) for x in a
                if x.get("standard_value") and x.get("standard_units") == "nM"]
        meta[mid] = {"name": m.get("pref_name"),
                     "phase": m.get("max_phase"),
                     "egfr_nM": float(np.median(vals)) if vals else None,
                     "n": len(vals)}

print(f"{'#':<3}{'molecule':<15}{'pred':>6}{'IC50 nM':>9}{'maxTc':>7}{'QED':>6}"
      f"{'SA':>5}{'hERG':>6}  {'known EGFR IC50':<17}{'name/phase'}")
print("-" * 104)
for l in res["leads"]:
    m = meta[l["molecule_id"]]
    kn = f"{m['egfr_nM']:.1f} nM (n={m['n']})" if m["egfr_nM"] else "none in ChEMBL"
    nm = f"{m['name'] or '-'}"
    if m["phase"]: nm += f" / phase {m['phase']}"
    print(f"{l['rank']:<3}{l['molecule_id']:<15}{l['pic50']:>6.2f}{l['ic50_nm']:>9.1f}"
          f"{l['max_similarity']:>7.2f}{l['qed']:>6.2f}{l['sa_score']:>5.1f}"
          f"{(l['herg_risk'] or 0):>6.2f}  {kn:<17}{nm}")
    print(f"   {l['smiles']}")
    for fl in l["flags"]:
        print(f"   ! {fl}")

hits = [m for m in meta.values() if m["egfr_nM"]]
print(f"\n--- novelty audit of the {len(ids)} leads ---")
print(f"  with ANY measured EGFR IC50 in ChEMBL : {len(hits)}")
if hits:
    for m in hits:
        print(f"    {m['name'] or '(unnamed)':<28} measured {m['egfr_nM']:.1f} nM")
print(f"  genuinely unmeasured against EGFR     : {len(ids)-len(hits)}")

report, src = generate_report(res, prefer="rules")
print(f"\n--- generated report ({src}) ---")
print(report[:1400])
