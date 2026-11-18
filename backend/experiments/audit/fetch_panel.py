"""Pull canonical SMILES for a known-drug validation panel straight from ChEMBL.

Typing SMILES from memory is how silent errors enter a validation set; the
structures come from the same authority that produced the training labels.
"""
import json, sys
from pathlib import Path
import httpx

BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}

PANEL = {
    # pan-CDK / CDK2-active -- should score HIGH
    "dinaciclib":   "pan-CDK, CDK2 IC50 ~1 nM",
    "alvocidib":    "flavopiridol, pan-CDK, CDK2 ~100 nM",
    "seliciclib":   "roscovitine, CDK2 ~700 nM",
    "AT7519":       "pan-CDK, CDK2 ~47 nM",
    "milciclib":    "CDK2 ~45 nM",
    "riviciclib":   "P276-00, pan-CDK",
    # CDK4/6-selective -- should score LOWER on CDK2
    "palbociclib":  "CDK4/6-selective, weak CDK2",
    "ribociclib":   "CDK4/6-selective, weak CDK2",
    "abemaciclib":  "CDK4/6, some CDK2 activity",
    # other kinase inhibitors -- off-target controls
    "imatinib":     "BCR-ABL, not a CDK2 drug",
    "gefitinib":    "EGFR, not a CDK2 drug",
    # non-kinase negative controls -- should score LOW
    "aspirin":      "NSAID, negative control",
    "caffeine":     "xanthine, negative control",
    "paracetamol":  "analgesic, negative control",
    "metformin":    "biguanide, negative control",
}

out = {}
with httpx.Client(headers=H, timeout=60.0) as c:
    for name, note in PANEL.items():
        r = c.get(f"{BASE}/molecule/search", params={"q": name, "format": "json"})
        if r.status_code != 200:
            print(f"  {name:<14} HTTP {r.status_code}"); continue
        mols = r.json().get("molecules", [])
        hit = None
        for m in mols:  # prefer an exact preferred-name match
            if (m.get("pref_name") or "").lower() == name.lower():
                hit = m; break
        hit = hit or (mols[0] if mols else None)
        if not hit:
            print(f"  {name:<14} NOT FOUND"); continue
        smi = (hit.get("molecule_structures") or {}).get("canonical_smiles")
        if not smi:
            print(f"  {name:<14} no structure"); continue
        out[name] = {"chembl_id": hit["molecule_chembl_id"],
                     "pref_name": hit.get("pref_name"), "smiles": smi, "note": note}
        print(f"  {name:<14} {hit['molecule_chembl_id']:<16} {hit.get('pref_name')}")

Path(__file__).with_name("panel.json").write_text(json.dumps(out, indent=2))
print(f"\n{len(out)}/{len(PANEL)} resolved")
