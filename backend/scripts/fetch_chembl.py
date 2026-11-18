"""Download external data from the ChEMBL REST API.

Two datasets, both keyless and public:

  * herg   -- CHEMBL240 (human hERG / KCNH2) bioactivities, used to train a real
              cardiotoxicity classifier instead of hand-waving a "toxicity flag".
  * pool   -- a screening library of drug-like molecules with no measured EGFR
              activity, so the funnel screens genuinely unlabeled chemistry
              rather than replaying its own training set.

The API paginates at 1000 records. Requests are retried with backoff because a
partial download that silently truncates is worse than a loud failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.featurize import canonical_smiles, mol_from_smiles  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

BASE = "https://www.ebi.ac.uk/chembl/api/data"
DATA_DIR = ROOT / "backend" / "data"
PAGE_SIZE = 1000
HEADERS = {"Accept": "application/json", "User-Agent": "BioOrchestrator/1.0"}


def paged_get(client: httpx.Client, path: str, params: dict, max_records: int):
    """Walk ChEMBL pagination, yielding record batches."""
    params = dict(params, limit=PAGE_SIZE, offset=0, format="json")
    fetched = 0
    while fetched < max_records:
        for attempt in range(4):
            try:
                r = client.get(f"{BASE}/{path}", params=params, timeout=120.0)
                r.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                if attempt == 3:
                    raise RuntimeError(f"ChEMBL request failed after 4 tries: {exc}")
                wait = 2 ** attempt
                print(f"    retry {attempt + 1} in {wait}s ({type(exc).__name__})")
                time.sleep(wait)

        payload = r.json()
        key = next(k for k in payload if k != "page_meta")
        batch = payload[key]
        if not batch:
            return
        yield batch
        fetched += len(batch)
        print(f"    {fetched} records", flush=True)
        if len(batch) < PAGE_SIZE:
            return
        params["offset"] += PAGE_SIZE


def fetch_herg(max_records: int = 20000) -> Path:
    """hERG blockade data: IC50/Ki against CHEMBL240.

    Labelled by the standard cardiotoxicity convention -- IC50 < 10 uM
    (pIC50 > 5) counts as a blocker. That threshold is the one used across the
    hERG QSAR literature, not an arbitrary cut.
    """
    print("\n=== hERG (CHEMBL240) ===")
    rows = []
    params = {
        "target_chembl_id": "CHEMBL240",
        "standard_type__in": "IC50,Ki,Kd",
        "standard_relation": "=",
        "standard_units": "nM",
        "pchembl_value__isnull": "false",
    }
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for batch in paged_get(client, "activity", params, max_records):
            for a in batch:
                smi = a.get("canonical_smiles")
                pch = a.get("pchembl_value")
                if not smi or pch is None:
                    continue
                rows.append({"molecule_id": a.get("molecule_chembl_id"),
                             "smiles": smi, "pchembl": float(pch),
                             "standard_type": a.get("standard_type")})

    df = pd.DataFrame(rows)
    print(f"  {len(df)} raw activities")
    df["smiles"] = [canonical_smiles(s) for s in df["smiles"]]
    df = df.dropna(subset=["smiles"])
    agg = (df.groupby("smiles")
             .agg(pchembl=("pchembl", "median"), n=("pchembl", "size"),
                  molecule_id=("molecule_id", "first"))
             .reset_index())
    agg["is_blocker"] = (agg["pchembl"] > 5.0).astype(int)
    out = DATA_DIR / "herg_chembl.csv"
    agg.to_csv(out, index=False)
    pos = int(agg.is_blocker.sum())
    print(f"  {len(agg)} unique compounds -> {pos} blockers / "
          f"{len(agg) - pos} non-blockers ({pos / len(agg):.1%} positive)")
    print(f"  saved {out}")
    return out


def _drug_like(smiles: str) -> bool:
    """Coarse pre-filter so the pool is not full of peptides and dyes."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    if not (150.0 <= mw <= 600.0):
        return False
    if Descriptors.NumRotatableBonds(mol) > 12:
        return False
    if mol.GetNumHeavyAtoms() < 12:
        return False
    # Reject anything with elements outside the usual medicinal-chemistry set.
    allowed = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"}
    return all(a.GetSymbol() in allowed for a in mol.GetAtoms())


def fetch_pool(target_size: int = 10000, max_records: int = 60000) -> Path:
    """Build the unlabeled screening library.

    Pulled from ChEMBL molecules that passed at least phase-0 development, so
    they are real, synthesizable chemistry. Any compound already present in the
    EGFR training set is removed -- screening your own training data and
    reporting the hits would be meaningless.
    """
    print("\n=== screening pool ===")
    train_csv = DATA_DIR / "egfr_chembl_labeled.csv"
    known = set()
    if train_csv.exists():
        tdf = pd.read_csv(train_csv)
        known = {c for c in (canonical_smiles(s) for s in tdf["smiles"]) if c}
        print(f"  {len(known)} known EGFR compounds will be excluded")

    rows, seen = [], set()
    params = {
        "molecule_properties__mw_freebase__gte": 200,
        "molecule_properties__mw_freebase__lte": 550,
        "molecule_properties__num_ro5_violations__lte": 1,
        "molecule_type": "Small molecule",
        "only": "molecule_chembl_id,molecule_structures,molecule_properties,pref_name",
    }
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for batch in paged_get(client, "molecule", params, max_records):
            for m in batch:
                struct = m.get("molecule_structures") or {}
                smi = struct.get("canonical_smiles")
                if not smi:
                    continue
                canon = canonical_smiles(smi)
                if not canon or canon in seen or canon in known:
                    continue
                if not _drug_like(canon):
                    continue
                seen.add(canon)
                rows.append({"molecule_id": m.get("molecule_chembl_id"),
                             "name": m.get("pref_name") or "",
                             "smiles": canon})
            if len(rows) >= target_size:
                break

    df = pd.DataFrame(rows[:target_size])
    out = DATA_DIR / "candidate_pool.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} unique drug-like candidates (none overlap training set)")
    print(f"  saved {out}")
    return out


# Diverse, well-populated human protein kinases.
#
# WARNING -- unresolved: this list was written when the training set was
# believed to be CDK2, so CDK2 (CHEMBL301) was excluded and EGFR (CHEMBL203)
# included. The training set is in fact EGFR, which inverts the intent: the
# trained target is now *in* the pool's source list. The per-compound exclusion
# below still removes every training molecule, so there is no literal overlap
# (audited: 0), but a pool seeded from the trained target's own chemistry is
# not the neutral library this was meant to be.
#
# Fixing it means dropping EGFR here and rebuilding the pool cache, which
# changes every downstream result -- so it is left as an explicit decision
# rather than a silent edit. See experiments/audit/AUDIT.md.
KINASE_TARGETS = {
    "EGFR": "CHEMBL203", "ABL1": "CHEMBL1862", "KDR/VEGFR2": "CHEMBL279",
    "p38alpha": "CHEMBL260", "SRC": "CHEMBL267", "GSK3B": "CHEMBL262",
    "AURKA": "CHEMBL4722", "PLK1": "CHEMBL3024", "CHK1": "CHEMBL4630",
    "JAK2": "CHEMBL2971", "CDK1": "CHEMBL308", "CDK4": "CHEMBL331",
    "CDK6": "CHEMBL3650", "MET": "CHEMBL3717", "LCK": "CHEMBL258",
    "PKA": "CHEMBL4101", "ROCK1": "CHEMBL3231", "PIM1": "CHEMBL2147",
    "FLT3": "CHEMBL1974", "BTK": "CHEMBL5251",
}


def fetch_kinase_pool(target_size: int = 10000, per_target: int = 4000) -> Path:
    """Build a kinase-focused screening library.

    Screening a random drug-like library against a kinase finds nothing, and
    that is not a modelling failure -- ATP-competitive inhibitors share hinge-
    binding motifs that random chemistry simply does not have. Real virtual
    screening campaigns use focused libraries for exactly this reason.

    So the pool is drawn from compounds with measured activity against other
    kinases. Every training-set compound is excluded, which keeps the screen
    honest while making it ask a genuinely useful question: which known kinase
    chemistry might cross-react with EGFR?

    See the WARNING on KINASE_TARGETS -- EGFR is still one of the source
    targets, which weakens that framing until the pool is rebuilt.
    """
    print("\n=== kinase-focused screening pool ===")
    train_csv = DATA_DIR / "egfr_chembl_labeled.csv"
    known = set()
    if train_csv.exists():
        tdf = pd.read_csv(train_csv)
        known = {c for c in (canonical_smiles(s) for s in tdf["smiles"]) if c}
        print(f"  excluding {len(known)} EGFR-labelled training compounds")

    rows, seen, degraded = [], set(), []
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for name, tid in KINASE_TARGETS.items():
            print(f"  {name} ({tid})")
            params = {
                "target_chembl_id": tid,
                "standard_type__in": "IC50,Ki,Kd",
                "standard_relation": "=",
                "standard_units": "nM",
                "pchembl_value__isnull": "false",
            }
            got, n_records, n_no_structure = 0, 0, 0
            try:
                for batch in paged_get(client, "activity", params, per_target):
                    for a in batch:
                        n_records += 1
                        smi = a.get("canonical_smiles")
                        if not smi:
                            n_no_structure += 1
                            continue
                        canon = canonical_smiles(smi)
                        if not canon or canon in seen or canon in known:
                            continue
                        if not _drug_like(canon):
                            continue
                        seen.add(canon)
                        got += 1
                        rows.append({
                            "molecule_id": a.get("molecule_chembl_id"),
                            "name": "",
                            "smiles": canon,
                            "source_kinase": name,
                            "source_pchembl": a.get("pchembl_value"),
                        })
            except RuntimeError as exc:
                print(f"    skipped ({exc})")
            print(f"    +{got} new compounds (pool now {len(rows)})")
            # A degraded ChEMBL response can return activity records with the
            # structure field null. Those silently contribute nothing, which
            # looks identical to "this target had no drug-like actives" unless
            # it is called out. Observed live: one target returned 1000 records
            # and 7 usable compounds while its neighbour returned 2100.
            if n_records >= 200 and n_no_structure > 0.5 * n_records:
                degraded.append(name)
                print(f"    WARNING: {n_no_structure}/{n_records} records had no "
                      f"structure -- {name} likely hit a degraded response. "
                      f"Re-run to backfill.")

            # Checkpoint after every target. ChEMBL outages can stall a single
            # request for minutes, and a twenty-target run that only writes at
            # the end throws away everything it collected if it dies partway.
            if rows:
                pd.DataFrame(rows).to_csv(DATA_DIR / "candidate_pool.partial.csv",
                                          index=False)
            if len(rows) >= target_size:
                break

    df = pd.DataFrame(rows[:target_size])
    out = DATA_DIR / "candidate_pool.csv"
    df.to_csv(out, index=False)
    print(f"\n  {len(df)} unique kinase-active compounds, zero training overlap")
    print(f"  drawn from {df.source_kinase.nunique()} distinct kinases")
    if degraded:
        print(f"  NOTE: degraded responses for {', '.join(degraded)}; "
              f"re-run to backfill those targets.")
    print(f"  saved {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["herg", "pool", "kinase_pool", "all"],
                    default="all", nargs="?")
    ap.add_argument("--pool-size", type=int, default=10000)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True) as c:
        status = c.get(f"{BASE}/status.json", timeout=30.0).json()
    print(f"ChEMBL {status.get('chembl_db_version')} reachable "
          f"({status.get('activities'):,} activities)")

    manifest = {"chembl_version": status.get("chembl_db_version"),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.what in ("herg", "all"):
        fetch_herg()
        manifest["herg"] = "herg_chembl.csv"
    if args.what == "pool":
        fetch_pool(args.pool_size)
        manifest["pool"] = "candidate_pool.csv"
    if args.what in ("kinase_pool", "all"):
        fetch_kinase_pool(args.pool_size)
        manifest["pool"] = "candidate_pool.csv"
        manifest["pool_type"] = "kinase-focused"
    (DATA_DIR / "chembl_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nall downloads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
