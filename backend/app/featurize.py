"""Molecular featurization.

One place where SMILES becomes numbers, so training and inference can never
disagree about what a feature vector means. Anything that changes the width or
the order of the vector must bump FEATURE_VERSION -- the model artifacts record
it and refuse to load against a mismatch.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

FEATURE_VERSION = 2
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Curated descriptor set. The full 217-descriptor list carries redundant and
# occasionally NaN-producing entries (Ipc overflows on large molecules); these
# 26 are cheap, well-conditioned, and physically interpretable.
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "HeavyAtomCount", "NHOHCount", "NOCount",
    "FractionCSP3", "BalabanJ", "BertzCT", "Chi0v", "Chi1v", "Chi2v",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "qed",
]
_DESC_FUNCS = {name: fn for name, fn in Descriptors._descList}
_MISSING = [n for n in DESCRIPTOR_NAMES if n not in _DESC_FUNCS]
if _MISSING:  # fail loudly at import rather than silently shipping zeros
    raise ImportError(f"RDKit is missing expected descriptors: {_MISSING}")

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)
N_FEATURES = MORGAN_BITS + N_DESCRIPTORS

_morgan = rfg.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

_sa_path = os.path.join(os.path.dirname(Chem.__file__), "..", "Contrib", "SA_Score")
_sa_path = os.path.normpath(_sa_path)
if _sa_path not in sys.path:
    sys.path.append(_sa_path)
try:
    import sascorer  # type: ignore

    _HAS_SA = True
except ImportError:  # pragma: no cover - SA score is bundled with RDKit
    _HAS_SA = False


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse and standardise. Returns None on anything RDKit rejects."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return mol


def canonical_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_counts(mol: Chem.Mol) -> np.ndarray:
    """Count-based Morgan fingerprint.

    Counts rather than bits: a molecule with four anilines is chemically
    different from one with a single aniline, and binary bits erase that.
    Counts are log1p-compressed so a hot bit cannot dominate the tree splits.
    """
    counts = _morgan.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    return np.log1p(counts)


def morgan_bits(mol: Chem.Mol) -> np.ndarray:
    """Binary fingerprint, used for Tanimoto similarity (not for the model)."""
    return _morgan.GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptors(mol: Chem.Mol) -> np.ndarray:
    out = np.empty(N_DESCRIPTORS, dtype=np.float32)
    for i, name in enumerate(DESCRIPTOR_NAMES):
        try:
            val = _DESC_FUNCS[name](mol)
        except Exception:
            val = np.nan
        out[i] = val
    # XGBoost handles NaN natively; guard only the non-finite blowups.
    out[~np.isfinite(out)] = np.nan
    return out


def featurize_mol(mol: Chem.Mol) -> np.ndarray:
    return np.concatenate([morgan_counts(mol), descriptors(mol)])


def featurize_many(smiles_list) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Featurize a list of SMILES.

    Returns (X, fingerprint_bits, kept_indices). Unparseable SMILES are dropped
    and their positions reported, so callers can realign their own metadata
    instead of guessing which row went missing.
    """
    feats, bits, kept = [], [], []
    for i, smi in enumerate(smiles_list):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        feats.append(featurize_mol(mol))
        bits.append(morgan_bits(mol))
        kept.append(i)
    if not feats:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros((0, MORGAN_BITS), dtype=np.uint8),
            [],
        )
    return np.vstack(feats), np.vstack(bits), kept


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold, used to build honest train/test splits.

    Acyclic molecules have an empty scaffold; they are bucketed under their own
    SMILES so they do not all collapse into one giant pseudo-scaffold.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""
    return scaffold if scaffold else Chem.MolToSmiles(mol)


def sa_score(mol: Chem.Mol) -> float:
    """Ertl synthetic accessibility, 1 (easy) to 10 (hard)."""
    if not _HAS_SA:
        return float("nan")
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class FeatureSpec:
    """Recorded alongside every model artifact so a stale model cannot load."""

    version: int = FEATURE_VERSION
    morgan_radius: int = MORGAN_RADIUS
    morgan_bits: int = MORGAN_BITS
    n_descriptors: int = N_DESCRIPTORS
    descriptor_names: tuple = tuple(DESCRIPTOR_NAMES)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "morgan_radius": self.morgan_radius,
            "morgan_bits": self.morgan_bits,
            "n_descriptors": self.n_descriptors,
            "descriptor_names": list(self.descriptor_names),
        }


def spec_dict() -> dict:
    return FeatureSpec().as_dict()
