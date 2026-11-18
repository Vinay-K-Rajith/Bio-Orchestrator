"""Compute every candidate feature block once and cache to .npy.

Ablation then costs a column-concat instead of an RDKit pass.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Avalon import pyAvalonTools
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.app.featurize import canonical_smiles, murcko_scaffold, DESCRIPTOR_NAMES

OUT = Path(__file__).parent / "blocks"
OUT.mkdir(parents=True, exist_ok=True)

# ---- data -------------------------------------------------------------
df = pd.read_csv(ROOT / "backend/data/egfr_chembl_labeled.csv")
df["canonical"] = [canonical_smiles(s) for s in df["smiles"]]
df = df.dropna(subset=["canonical", "pIC50"])
agg = (df.groupby("canonical")
         .agg(pIC50=("pIC50", "median"), n_measurements=("pIC50", "size"),
              pIC50_std=("pIC50", "std"), molecule_id=("molecule_id", "first"))
         .reset_index().rename(columns={"canonical": "smiles"}))
mols = [Chem.MolFromSmiles(s) for s in agg["smiles"]]
keep = [i for i, m in enumerate(mols) if m is not None and m.GetNumHeavyAtoms() > 0]
agg = agg.iloc[keep].reset_index(drop=True)
mols = [mols[i] for i in keep]
print(f"{len(agg)} unique compounds")

np.save(OUT / "y.npy", agg["pIC50"].to_numpy(np.float32))
np.save(OUT / "n_meas.npy", agg["n_measurements"].to_numpy(np.float32))
scaffolds = [murcko_scaffold(s) for s in agg["smiles"]]
(OUT / "scaffolds.json").write_text(json.dumps(scaffolds))
agg.to_csv(OUT / "compounds.csv", index=False)
print(f"{len(set(scaffolds))} scaffolds")

# ---- generators -------------------------------------------------------
def counts(gen, name):
    t = time.time()
    M = np.vstack([np.log1p(gen.GetCountFingerprintAsNumPy(m).astype(np.float32)) for m in mols])
    np.save(OUT / f"{name}.npy", M)
    print(f"  {name:<20} {M.shape}  nnz-cols={int((M.sum(0) > 0).sum()):<5} {time.time()-t:.0f}s")

counts(rfg.GetMorganGenerator(radius=2, fpSize=2048), "morgan2_2048")
counts(rfg.GetMorganGenerator(radius=3, fpSize=2048), "morgan3_2048")
counts(rfg.GetMorganGenerator(radius=2, fpSize=4096), "morgan2_4096")
# FCFP: feature-invariant Morgan -- pharmacophoric abstraction
fcfp_inv = rfg.GetMorganFeatureAtomInvGen()
counts(rfg.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=fcfp_inv), "fcfp2_2048")
counts(rfg.GetMorganGenerator(radius=3, fpSize=2048, atomInvariantsGenerator=fcfp_inv), "fcfp3_2048")
counts(rfg.GetAtomPairGenerator(fpSize=2048), "atompair_2048")
counts(rfg.GetTopologicalTorsionGenerator(fpSize=2048), "torsion_2048")
counts(rfg.GetRDKitFPGenerator(fpSize=2048), "rdkitfp_2048")

t = time.time()
M = np.vstack([np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), "u1") - ord("0")
               for m in mols]).astype(np.float32)
np.save(OUT / "maccs_167.npy", M); print(f"  maccs_167            {M.shape} {time.time()-t:.0f}s")

t = time.time()
M = np.vstack([np.frombuffer(pyAvalonTools.GetAvalonCountFP(m, nBits=1024).ToList().__str__().encode()[:0] or b"", "u1")
               if False else np.array(pyAvalonTools.GetAvalonCountFP(m, nBits=1024).ToList(), dtype=np.float32)
               for m in mols])
M = np.log1p(M)
np.save(OUT / "avalon_1024.npy", M); print(f"  avalon_1024          {M.shape} {time.time()-t:.0f}s")

# ---- descriptors ------------------------------------------------------
DROP = {"Ipc"}  # overflows to inf on large molecules
desc_names = [n for n, _ in Descriptors._descList if n not in DROP]
funcs = dict(Descriptors._descList)
t = time.time()
D = np.empty((len(mols), len(desc_names)), dtype=np.float32)
for i, m in enumerate(mols):
    for j, n in enumerate(desc_names):
        try:
            D[i, j] = funcs[n](m)
        except Exception:
            D[i, j] = np.nan
D[~np.isfinite(D)] = np.nan
np.save(OUT / "desc_full.npy", D)
(OUT / "desc_full_names.json").write_text(json.dumps(desc_names))
print(f"  desc_full            {D.shape} {time.time()-t:.0f}s")

idx = [desc_names.index(n) for n in DESCRIPTOR_NAMES]
np.save(OUT / "desc26.npy", D[:, idx])
(OUT / "desc26_names.json").write_text(json.dumps(list(DESCRIPTOR_NAMES)))
print(f"  desc26               {D[:, idx].shape}")
print("done")
