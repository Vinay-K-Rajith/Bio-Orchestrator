"""Chemical property, safety, and applicability-domain engine (Stage 2).

Everything here is deterministic cheminformatics -- no learned model. It answers
three separate questions that the plan originally collapsed into one "safety"
number:

  1. Is this molecule drug-like enough to be an oral lead?   (Lipinski, QED, SA)
  2. Does it contain known-problematic substructures?        (PAINS/Brenk/NIH)
  3. Is the affinity model entitled to an opinion about it?  (applicability domain)

(3) is not a safety question at all -- it is a question about the *model*, and
conflating it with toxicity is how screening pipelines end up trusting confident
nonsense. It is kept as its own axis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

from .featurize import mol_from_smiles, morgan_bits, sa_score

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Lipinski Rule of 5 thresholds.
RO5 = {"mw": 500.0, "logp": 5.0, "hbd": 5, "hba": 10}
# Veber oral-bioavailability criteria -- complements Ro5, catches floppy molecules
# that pass Ro5 but are still poorly absorbed.
VEBER = {"rotb": 10, "tpsa": 140.0}

# Below this max-Tanimoto to any training compound, the surrogate is
# extrapolating. 0.35 on ECFP4 is the widely used applicability-domain cut.
AD_SIMILARITY_FLOOR = 0.35


def _build_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    for cat in (FilterCatalogParams.FilterCatalogs.PAINS,
                FilterCatalogParams.FilterCatalogs.BRENK,
                FilterCatalogParams.FilterCatalogs.NIH):
        params.AddCatalog(cat)
    return FilterCatalog.FilterCatalog(params)


_CATALOG = _build_catalog()


@dataclass
class MoleculeProfile:
    """Everything Stage 2 knows about one candidate."""

    smiles: str
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    rotatable_bonds: int = 0
    aromatic_rings: int = 0
    heavy_atoms: int = 0
    fraction_csp3: float = 0.0
    qed: float = 0.0
    sa_score: float = 0.0

    lipinski_violations: int = 0
    lipinski_failed: list = field(default_factory=list)
    veber_pass: bool = True
    structural_alerts: list = field(default_factory=list)
    n_alerts: int = 0

    max_similarity: float = 0.0
    nearest_neighbor_idx: int = -1
    in_domain: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def profile_molecule(smiles: str, mol: Chem.Mol | None = None) -> MoleculeProfile | None:
    """Compute physicochemical properties and structural alerts."""
    if mol is None:
        mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    p = MoleculeProfile(smiles=Chem.MolToSmiles(mol))
    p.mw = round(float(Descriptors.MolWt(mol)), 2)
    p.logp = round(float(Descriptors.MolLogP(mol)), 2)
    p.hbd = int(Descriptors.NumHDonors(mol))
    p.hba = int(Descriptors.NumHAcceptors(mol))
    p.tpsa = round(float(Descriptors.TPSA(mol)), 2)
    p.rotatable_bonds = int(Descriptors.NumRotatableBonds(mol))
    p.aromatic_rings = int(Descriptors.NumAromaticRings(mol))
    p.heavy_atoms = int(mol.GetNumHeavyAtoms())
    p.fraction_csp3 = round(float(Descriptors.FractionCSP3(mol)), 3)
    p.qed = round(float(Descriptors.qed(mol)), 3)
    p.sa_score = round(sa_score(mol), 2)

    failed = []
    if p.mw >= RO5["mw"]:
        failed.append(f"MW {p.mw} >= {RO5['mw']}")
    if p.logp >= RO5["logp"]:
        failed.append(f"LogP {p.logp} >= {RO5['logp']}")
    if p.hbd > RO5["hbd"]:
        failed.append(f"HBD {p.hbd} > {RO5['hbd']}")
    if p.hba > RO5["hba"]:
        failed.append(f"HBA {p.hba} > {RO5['hba']}")
    p.lipinski_failed = failed
    p.lipinski_violations = len(failed)
    p.veber_pass = (p.rotatable_bonds <= VEBER["rotb"] and p.tpsa <= VEBER["tpsa"])

    alerts = [m.GetDescription() for m in _CATALOG.GetMatches(mol)]
    # Deduplicate while preserving order; catalogs overlap heavily.
    p.structural_alerts = list(dict.fromkeys(alerts))
    p.n_alerts = len(p.structural_alerts)
    return p


class ApplicabilityDomain:
    """Tanimoto nearest-neighbour check against the training set.

    Bulk Tanimoto over packed bit arrays: with 2,765 references and a 2048-bit
    fingerprint, the whole 10k x 2,765 similarity matrix is a single matrix
    multiply, which is far faster than RDKit's per-pair loop.
    """

    def __init__(self, reference_fps: np.ndarray):
        self.ref = reference_fps.astype(np.float32)
        self.ref_popcount = self.ref.sum(axis=1)

    @classmethod
    def load(cls, path: Path | None = None) -> "ApplicabilityDomain":
        path = path or (MODEL_DIR / "reference_fps.npy")
        if not path.exists():
            raise FileNotFoundError(
                f"Reference fingerprints missing at {path}. "
                "Run backend/scripts/train_surrogate.py first."
            )
        return cls(np.load(path))

    def max_similarity(self, query_fps: np.ndarray):
        """Return (max_similarity, argmax_index) per query molecule.

        Tanimoto for binary vectors: |A and B| / (|A| + |B| - |A and B|).
        The intersection for all pairs at once is Q @ R.T.
        """
        q = query_fps.astype(np.float32)
        inter = q @ self.ref.T
        q_pop = q.sum(axis=1)[:, None]
        union = q_pop + self.ref_popcount[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        return sim.max(axis=1), sim.argmax(axis=1)


def annotate_domain(profiles: list[MoleculeProfile], fps: np.ndarray,
                    domain: ApplicabilityDomain,
                    floor: float = AD_SIMILARITY_FLOOR) -> None:
    """Attach applicability-domain results to profiles in place."""
    if not profiles:
        return
    sims, idxs = domain.max_similarity(fps)
    for p, s, i in zip(profiles, sims, idxs):
        p.max_similarity = round(float(s), 3)
        p.nearest_neighbor_idx = int(i)
        p.in_domain = bool(s >= floor)


def passes_stage2(p: MoleculeProfile, require_lipinski: bool = True,
                  max_alerts: int = 0, require_in_domain: bool = False) -> bool:
    """Stage-2 gate.

    Defaults are deliberately strict on structural alerts (zero tolerated) but
    do not require in-domain by default -- an out-of-domain molecule is a
    *flagged* result, not a discarded one, since novel chemistry is often the
    point of a screen.
    """
    if require_lipinski and p.lipinski_violations > 1:
        return False
    if p.n_alerts > max_alerts:
        return False
    if require_in_domain and not p.in_domain:
        return False
    return True


def profile_batch(smiles_list: list[str]):
    """Profile many SMILES, returning (profiles, fingerprints, kept_indices)."""
    profiles, bits, kept = [], [], []
    for i, smi in enumerate(smiles_list):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        prof = profile_molecule(smi, mol)
        if prof is None:
            continue
        profiles.append(prof)
        bits.append(morgan_bits(mol))
        kept.append(i)
    fps = np.vstack(bits) if bits else np.zeros((0, 2048), dtype=np.uint8)
    return profiles, fps, kept


if __name__ == "__main__":
    # Self-test against molecules with known, checkable properties.
    tests = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "cyclosporine-like(big)": "CC(C)C[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)[C@H](C)NC1=O",
        "azo-dye(PAINS)": "O=C(O)c1ccccc1N=Nc1ccc(O)cc1",
    }
    print(f"{'name':<24}{'MW':>8}{'LogP':>7}{'QED':>6}{'SA':>6}{'Ro5':>5}{'alerts':>7}")
    print("-" * 63)
    for name, smi in tests.items():
        p = profile_molecule(smi)
        print(f"{name:<24}{p.mw:>8.1f}{p.logp:>7.2f}{p.qed:>6.2f}{p.sa_score:>6.2f}"
              f"{p.lipinski_violations:>5}{p.n_alerts:>7}")
        if p.structural_alerts:
            print(f"{'':<24}alerts: {', '.join(p.structural_alerts[:4])}")

    try:
        dom = ApplicabilityDomain.load()
        profiles, fps, _ = profile_batch(list(tests.values()))
        annotate_domain(profiles, fps, dom)
        print("\napplicability domain vs EGFR training set:")
        for name, p in zip(tests, profiles):
            tag = "IN-DOMAIN" if p.in_domain else "OUT-OF-DOMAIN"
            print(f"  {name:<24} max Tanimoto={p.max_similarity:.3f}  {tag}")
    except FileNotFoundError as e:
        print(f"\n[skip domain test] {e}")
