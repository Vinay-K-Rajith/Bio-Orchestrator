"""Audit suspicious training labels against ChEMBL's own activity records.

For each molecule, ask ChEMBL what IC50s it actually holds for CDK2 (CHEMBL301)
and compare with the label the training CSV assigns.
"""
import httpx, pandas as pd
BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}
CHECK = {"CHEMBL939": "gefitinib (label 28 nM)",
         "CHEMBL428690": "alvocidib (label 14800 nM)",
         "CHEMBL941": "imatinib (label 100000 nM)",
         "CHEMBL2103840": "dinaciclib (absent from training)",
         "CHEMBL445813": "AT-7519 (absent from training)"}

with httpx.Client(headers=H, timeout=120.0) as c:
    for mid, desc in CHECK.items():
        print(f"\n=== {desc}  [{mid}] ===")
        for tgt, tname in (("CHEMBL301", "CDK2"), (None, "ANY TARGET")):
            params = {"molecule_chembl_id": mid, "standard_type": "IC50",
                      "format": "json", "limit": 60}
            if tgt: params["target_chembl_id"] = tgt
            r = c.get(f"{BASE}/activity", params=params)
            acts = r.json().get("activities", [])
            if tgt:
                if not acts:
                    print(f"  CDK2 (CHEMBL301): NO IC50 RECORDS IN CHEMBL")
                else:
                    vals = [(a.get("standard_value"), a.get("standard_units"),
                             a.get("standard_relation"), a.get("assay_chembl_id"))
                            for a in acts]
                    print(f"  CDK2 (CHEMBL301): {len(acts)} records")
                    for v in vals[:6]: print(f"     {v}")
            else:
                # which targets DO have an IC50 near the training label?
                seen = {}
                for a in acts:
                    t = a.get("target_pref_name"); v = a.get("standard_value")
                    if t and v: seen.setdefault(t, []).append(float(v))
                print(f"  targets with IC50 records (top 6 by count):")
                for t, vs in sorted(seen.items(), key=lambda kv: -len(kv[1]))[:6]:
                    print(f"     {t[:52]:<54} n={len(vs):<4} median={pd.Series(vs).median():.1f} nM")
