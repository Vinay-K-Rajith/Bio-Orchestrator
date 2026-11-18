"""Rule out the obvious false alarm: CDK2 activity split across complex targets.

ChEMBL stores 'Cyclin-dependent kinase 2' and 'CDK2/cyclin E' as separate
targets. If the training set was built from the complexes, my CHEMBL301-only
join would wrongly report it as contaminated. So: enumerate every CDK2-ish
target, redo the join across all of them, and then ask what targets the
still-missing molecules are actually measured against.
"""
from pathlib import Path
import httpx, numpy as np, pandas as pd
from collections import Counter

BASE = "https://www.ebi.ac.uk/chembl/api/data"
H = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}
HERE = Path(__file__).parent; ROOT = Path(__file__).resolve().parents[3]
train = pd.read_csv(ROOT / "backend/data/cdk2_chembl_labeled.csv")

with httpx.Client(headers=H, timeout=180.0) as c:
    # 1. every target whose name mentions CDK2 / cyclin-dependent kinase 2
    tids = {}
    for q in ("Cyclin-dependent kinase 2", "CDK2"):
        r = c.get(f"{BASE}/target/search", params={"q": q, "format": "json", "limit": 100})
        for t in r.json().get("targets", []):
            nm = t.get("pref_name") or ""
            if "2" in nm and ("cyclin" in nm.lower() or "cdk2" in nm.lower()):
                tids[t["target_chembl_id"]] = f'{nm} [{t.get("organism")}]'
    print(f"CDK2-related targets found: {len(tids)}")
    for k, v in tids.items(): print(f"  {k:<16} {v}")

    # 2. pull every activity (any standard_type) for all of them
    recs = []
    for tid in tids:
        off = 0
        while True:
            r = c.get(f"{BASE}/activity", params={"target_chembl_id": tid, "format": "json",
                                                  "limit": 1000, "offset": off})
            a = r.json().get("activities", [])
            if not a: break
            recs += [{"molecule_id": x.get("molecule_chembl_id"), "target": tid,
                      "type": x.get("standard_type")} for x in a]
            off += len(a)
            if len(a) < 1000: break
    allcdk2 = pd.DataFrame(recs).dropna(subset=["molecule_id"])
    print(f"\nactivities across all CDK2 targets: {len(allcdk2)} "
          f"over {allcdk2.molecule_id.nunique()} molecules (any assay type)")

    covered = set(allcdk2["molecule_id"])
    hit = train["molecule_id"].isin(covered)
    print(f"\n--- training set vs ALL CDK2 targets, ANY assay type ---")
    print(f"  covered     : {hit.sum()} / {len(train)} ({100*hit.mean():.1f}%)")
    print(f"  NOT covered : {(~hit).sum()} ({100*(~hit).mean():.1f}%)")

    # 3. what ARE the uncovered molecules measured against?
    missing = train.loc[~hit, "molecule_id"].tolist()
    rng = np.random.default_rng(0)
    sample = list(rng.choice(missing, size=min(120, len(missing)), replace=False))
    tgt_counter, n_with_any = Counter(), 0
    for i in range(0, len(sample), 20):
        batch = ",".join(sample[i:i+20])
        r = c.get(f"{BASE}/activity", params={"molecule_chembl_id__in": batch,
                                              "standard_type": "IC50",
                                              "format": "json", "limit": 1000})
        acts = r.json().get("activities", [])
        seen_mol = set()
        for a in acts:
            t = a.get("target_pref_name")
            if t: tgt_counter[t] += 1; seen_mol.add(a.get("molecule_chembl_id"))
        n_with_any += len(seen_mol)
    print(f"\n--- what {len(sample)} sampled 'missing' molecules are ACTUALLY measured on ---")
    print(f"  had any IC50 record anywhere in ChEMBL: {n_with_any}/{len(sample)}")
    for t, n in tgt_counter.most_common(12):
        print(f"    {t[:58]:<60} {n}")
