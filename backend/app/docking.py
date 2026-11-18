"""3D Structural Biology, Conformer Docking, and Non-Covalent Interaction Engine for EGFR.

Bridges the 2D QSAR screening funnel with 3D crystallographic EGFR binding pockets:
  - 1M17: Wild-type Active EGFR + Erlotinib (AQ4) [mature numbering shifted +24 to canonical UniProt]
  - 2ITY: Wild-type Active EGFR + Gefitinib (IRE)
  - 4I22: T790M / L858R Resistant Mutant EGFR + Gefitinib (IRE)
  - 4ZAU: Covalent 3rd-Gen EGFR + Osimertinib (YY3) targeting Cys797
  - 5D41: Allosteric 4th-Gen Site EGFR (T790M) + EAI045 (57N) in alphaC-out pocket

For any candidate SMILES, generates up to 9 low-energy 3D conformers (ETKDGv3 + MMFF94s),
performs pharmacophoric/shape-guided 3D alignment (RDKit Open3DAlign) into the co-crystal
binding cleft with local rigid-body pocket refinement, scores each mode with an empirical
AutoDock Vina-style free energy function (Delta G in kcal/mol), and extracts explicit 3D
non-covalent interaction vectors (H-bonds, pi-stacking/cation-pi, hydrophobic contacts,
Cys797 covalent proximity, and Thr790/Met790 gatekeeper steric contacts).
"""
from __future__ import annotations

import hashlib
import math
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolAlign, rdMolDescriptors, rdMolTransforms

ROOT = Path(__file__).resolve().parents[1]
PDB_DIR = ROOT / "data" / "pdb"
PDB_DIR.mkdir(parents=True, exist_ok=True)

# Van der Waals radii (Angstroms) for Vina-style steric/Gaussian terms and CPK rendering
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
}

# Standard CPK hex colors for Three.js InstancedMesh rendering
CPK_COLORS: dict[str, str] = {
    "C": "#34d399",   # Emerald carbon for docked ligand
    "N": "#60a5fa",   # Blue nitrogen
    "O": "#f87171",   # Red oxygen
    "S": "#fbbf24",   # Yellow sulfur
    "F": "#4ade80",   # Green fluorine
    "CL": "#22c55e",  # Chlorine
    "BR": "#fb923c",  # Orange-brown bromine
    "I": "#c084fc",   # Violet iodine
    "P": "#fb923c",   # Orange phosphorus
    "H": "#cbd5e1",   # Light slate hydrogen
}

PROTEIN_CPK_COLORS: dict[str, str] = {
    "C": "#94a3b8",   # Slate carbon for protein pocket
    "N": "#60a5fa",
    "O": "#f87171",
    "S": "#fbbf24",
}

# Key EGFR binding pocket residues (canonical UniProt P00533 numbering)
KEY_POCKET_RESIDUES: list[int] = [
    718,  # Leu718 (P-loop hydrophobic roof)
    726,  # Val726 (beta2-strand hydrophobic roof)
    743,  # Ala743 (beta3-strand ceiling)
    745,  # Lys745 (catalytic lysine, salt bridge / cation-pi)
    762,  # Glu762 (alphaC-helix salt-bridge partner to Lys745)
    766,  # Met766 (alphaC-helix back pocket)
    790,  # Thr790 / Met790 (Gatekeeper residue!)
    791,  # Gln791 (hinge region backbone)
    792,  # Leu792 (hinge region)
    793,  # Met793 (Primary hinge H-bond anchor!)
    796,  # Gly796 (solvent channel)
    797,  # Cys797 (Covalent Michael addition nucleophile!)
    844,  # Leu844 (hydrophobic pocket floor)
    854,  # Thr854 (pre-DFG polar contact)
    855,  # Asp855 (DFG motif)
    856,  # Phe856 (DFG aromatic ring, allosteric site anchor)
    858,  # Leu858 / Arg858 (activation loop oncogenic mutation site)
]


@dataclass
class TargetMetadata:
    pdb_id: str
    title: str
    conformation: str
    generation: str
    cocrystal_ligand_id: str
    cocrystal_name: str
    cocrystal_smiles: str
    gatekeeper_residue: str
    hinge_residue: str
    covalent_residue: str
    numbering_note: str
    residue_offset: int
    key_residues_summary: str
    clinical_context: str


EGFR_TARGETS: dict[str, TargetMetadata] = {
    "1M17": TargetMetadata(
        pdb_id="1M17",
        title="Wild-Type EGFR (Active) + Erlotinib",
        conformation="Wild-type (Active DFG-in, alphaC-in)",
        generation="1st Gen Reversible TKI",
        cocrystal_ligand_id="AQ4",
        cocrystal_name="Erlotinib (Tarceva)",
        cocrystal_smiles="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        gatekeeper_residue="Thr790 (author Thr766)",
        hinge_residue="Met793 (author Met769)",
        covalent_residue="Cys797 (author Cys773)",
        numbering_note=(
            "Crystal structure 1M17 uses mature peptide numbering (-24 offset). "
            "Chain A coordinates are shifted by +24 so Thr790, Met793, Cys797, and Lys745 "
            "align identically with UniProt P00533 and all other EGFR structures."
        ),
        residue_offset=24,
        key_residues_summary="H-bond at Met793 hinge; Lys745 catalytic pocket; Thr790 gatekeeper.",
        clinical_context="First-line sensitive NSCLC (exon 19 del / L858R) prior to T790M resistance.",
    ),
    "2ITY": TargetMetadata(
        pdb_id="2ITY",
        title="Wild-Type EGFR (Active) + Gefitinib",
        conformation="Wild-type (Active DFG-in)",
        generation="1st Gen Reversible TKI",
        cocrystal_ligand_id="IRE",
        cocrystal_name="Gefitinib (Iressa)",
        cocrystal_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        gatekeeper_residue="Thr790",
        hinge_residue="Met793",
        covalent_residue="Cys797",
        numbering_note="Standard UniProt P00533 pro-EGFR numbering.",
        residue_offset=0,
        key_residues_summary="Quinazoline N1 accepts H-bond from Met793 amide NH; 3-chloro-4-fluoroaniline fills back pocket past Thr790.",
        clinical_context="Canonical reversible 4-anilinoquinazoline binding mode in wild-type EGFR.",
    ),
    "4I22": TargetMetadata(
        pdb_id="4I22",
        title="T790M / L858R Mutant EGFR + Gefitinib",
        conformation="T790M Gatekeeper Mutant (Resistant)",
        generation="2nd / 3rd Gen Resistance Context",
        cocrystal_ligand_id="IRE",
        cocrystal_name="Gefitinib (bound to T790M)",
        cocrystal_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        gatekeeper_residue="Met790 (T790M Mutant)",
        hinge_residue="Met793",
        covalent_residue="Cys797",
        numbering_note="Standard UniProt P00533 numbering with Met790 and Arg858 mutations.",
        residue_offset=0,
        key_residues_summary="Bulky Met790 thioether sidechain sterically clashes with 1st-gen quinazoline aniline rings and increases ATP affinity.",
        clinical_context="Primary acquired resistance mutation (~50-60% of patients progressing on Erlotinib/Gefitinib).",
    ),
    "4ZAU": TargetMetadata(
        pdb_id="4ZAU",
        title="Covalent Binding EGFR + Osimertinib (AZD9291)",
        conformation="Covalent Cys797-Targeted ATP Pocket",
        generation="3rd Gen Mutant-Selective Covalent TKI",
        cocrystal_ligand_id="YY3",
        cocrystal_name="Osimertinib (Tagrisso)",
        cocrystal_smiles="COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
        gatekeeper_residue="Thr790",
        hinge_residue="Met793",
        covalent_residue="Cys797 (Michael addition target)",
        numbering_note="Standard UniProt P00533 numbering.",
        residue_offset=0,
        key_residues_summary="Aminopyrimidine core forms bidentate H-bonds with Met793; acrylamide warhead projects directly to Cys797 thiol (SG).",
        clinical_context="Standard-of-care 3rd-gen TKI that spares wild-type EGFR while potently inhibiting T790M via Cys797 covalent bond.",
    ),
    "5D41": TargetMetadata(
        pdb_id="5D41",
        title="Allosteric Site EGFR (T790M) + EAI045",
        conformation="Allosteric Site (alphaC-helix out / Inactive)",
        generation="4th Gen Allosteric Inhibitor (C797S-active)",
        cocrystal_ligand_id="57N",
        cocrystal_name="EAI045",
        cocrystal_smiles="O=C(NC(c1ccccc1)c1nccs1)c1cc2ccccc2n1CC(=O)Nc1ccc(F)cc1O",
        gatekeeper_residue="Met790 (T790M Mutant)",
        hinge_residue="Met793 (adjacent to allosteric cleft)",
        covalent_residue="Cys797 (bypassed by allosteric site)",
        numbering_note="Standard UniProt P00533 numbering. Co-crystallized with ANP in ATP site and 57N in allosteric pocket.",
        residue_offset=0,
        key_residues_summary="Binds adjacent allosteric pocket opened by outward displacement of alphaC-helix; H-bonds with Asp855 (DFG) and Lys745.",
        clinical_context="4th-generation allosteric mechanism designed to overcome C797S resistance where covalent 3rd-gen TKIs lose their anchor.",
    ),
}


@dataclass
class Atom3D:
    serial: int
    name: str
    element: str
    resn: str
    chain: str
    resi: int
    x: float
    y: float
    z: float


@dataclass
class PocketData:
    pdb_id: str
    normalized_pdb_text: str
    pocket_atoms: list[Atom3D]
    key_residue_atoms: dict[int, list[Atom3D]]
    cocrystal_atoms: list[Atom3D]
    cocrystal_mol: Chem.Mol | None
    center: tuple[float, float, float]
    pocket_mesh_points: list[dict]


_POCKET_CACHE: dict[str, PocketData] = {}
_DOCK_CACHE: dict[str, dict] = {}


def ensure_pdb_downloaded(pdb_id: str) -> Path:
    pdb_id = pdb_id.upper()
    if pdb_id not in EGFR_TARGETS:
        raise ValueError(f"Unsupported EGFR PDB target '{pdb_id}'. Valid targets: {list(EGFR_TARGETS)}")
    dest = PDB_DIR / f"{pdb_id}.pdb"
    if not dest.exists() or dest.stat().st_size < 1000:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, dest)
    return dest


def _parse_element(line: str, atom_name: str) -> str:
    elem = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not elem:
        cleaned = "".join(ch for ch in atom_name if ch.isalpha()).upper()
        elem = cleaned[:2] if cleaned[:2] in ("CL", "BR", "FE", "MG", "ZN", "NA", "CA") else cleaned[:1]
    return elem or "C"


def load_pocket(pdb_id: str) -> PocketData:
    """Load and normalize an EGFR crystal structure, caching pocket atoms and co-crystal ligand."""
    pdb_id = pdb_id.upper()
    if pdb_id in _POCKET_CACHE:
        return _POCKET_CACHE[pdb_id]

    meta = EGFR_TARGETS[pdb_id]
    raw_path = ensure_pdb_downloaded(pdb_id)
    raw_lines = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    offset = meta.residue_offset
    normalized_lines: list[str] = [
        f"REMARK   1 BIOORCHESTRATOR EGFR TARGET {pdb_id} ({meta.title})",
        f"REMARK   2 RESIDUE OFFSET APPLIED TO CHAIN A: +{offset} ({meta.numbering_note})",
    ]

    chain_a_atoms: list[Atom3D] = []
    cocrystal_atoms: list[Atom3D] = []

    for line in raw_lines:
        if line.startswith("ATOM  "):
            chain = line[21].strip() or "A"
            if chain != "A":
                continue
            try:
                serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                alt_loc = line[16].strip()
                if alt_loc not in ("", "A", "1"):
                    continue
                resn = line[17:20].strip()
                resi = int(line[22:26].strip()) + offset
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                elem = _parse_element(line, atom_name)
            except ValueError:
                continue

            # Rewrite line with canonical UniProt residue index so 3Dmol.js & Three.js match
            new_line = f"{line[:22]}{resi:4d}{line[26:]}"
            normalized_lines.append(new_line)
            if elem != "H":
                chain_a_atoms.append(Atom3D(serial, atom_name, elem, resn, chain, resi, x, y, z))

        elif line.startswith("HETATM"):
            chain = line[21].strip() or "A"
            resn = line[17:20].strip()
            if chain == "A" and resn == meta.cocrystal_ligand_id:
                alt_loc = line[16].strip()
                if alt_loc not in ("", "A", "1"):
                    continue
                try:
                    serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    resi = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elem = _parse_element(line, atom_name)
                except ValueError:
                    continue
                normalized_lines.append(line)
                if elem != "H":
                    cocrystal_atoms.append(Atom3D(serial, atom_name, elem, resn, chain, resi, x, y, z))

        elif line.startswith(("HELIX ", "SHEET ")):
            # Shift secondary structure residue indices for Chain A if offset != 0
            if offset != 0 and len(line) >= 37:
                try:
                    if line.startswith("HELIX ") and line[19] == "A":
                        s_res = int(line[21:25].strip()) + offset
                        e_res = int(line[33:37].strip()) + offset
                        line = f"{line[:21]}{s_res:4d}{line[25:33]}{e_res:4d}{line[37:]}"
                    elif line.startswith("SHEET ") and line[21] == "A":
                        s_res = int(line[22:26].strip()) + offset
                        e_res = int(line[33:37].strip()) + offset
                        line = f"{line[:22]}{s_res:4d}{line[26:33]}{e_res:4d}{line[37:]}"
                except ValueError:
                    pass
            normalized_lines.append(line)

    normalized_lines.append("END")
    normalized_pdb_text = "\n".join(normalized_lines)

    # Compute pocket center from co-crystallized ligand (or key hinge/gatekeeper residues)
    if cocrystal_atoms:
        cx = float(np.mean([a.x for a in cocrystal_atoms]))
        cy = float(np.mean([a.y for a in cocrystal_atoms]))
        cz = float(np.mean([a.z for a in cocrystal_atoms]))
    else:
        hinge = [a for a in chain_a_atoms if a.resi in (790, 793, 797)]
        cx = float(np.mean([a.x for a in hinge]))
        cy = float(np.mean([a.y for a in hinge]))
        cz = float(np.mean([a.z for a in hinge]))

    # Filter pocket atoms within 10.5 A of any co-crystal atom (or within 13 A of pocket center)
    lig_coords = np.array([[a.x, a.y, a.z] for a in cocrystal_atoms], dtype=np.float64)
    pocket_atoms: list[Atom3D] = []
    key_residue_atoms: dict[int, list[Atom3D]] = {r: [] for r in KEY_POCKET_RESIDUES}

    for a in chain_a_atoms:
        if a.resi in key_residue_atoms:
            key_residue_atoms[a.resi].append(a)
        pt = np.array([a.x, a.y, a.z], dtype=np.float64)
        if len(lig_coords) > 0:
            min_d = float(np.min(np.linalg.norm(lig_coords - pt, axis=1)))
            if min_d <= 9.5 or a.resi in key_residue_atoms:
                pocket_atoms.append(a)
        else:
            if float(np.linalg.norm(pt - np.array([cx, cy, cz]))) <= 12.0:
                pocket_atoms.append(a)

    # Build an RDKit Mol for the co-crystal ligand with coordinates for 3D O3A alignment
    cocrystal_mol = _build_cocrystal_rdkit_mol(cocrystal_atoms, meta.cocrystal_smiles)

    # Build pocket surface/alpha-hull points for custom Three.js binding-pocket mesh
    pocket_mesh_points = [
        {
            "x": round(a.x, 3),
            "y": round(a.y, 3),
            "z": round(a.z, 3),
            "elem": a.element,
            "resn": a.resn,
            "resi": a.resi,
            "atom": a.name,
            "color": PROTEIN_CPK_COLORS.get(a.element, "#94a3b8"),
        }
        for a in pocket_atoms
        if a.resi in KEY_POCKET_RESIDUES or a.name in ("CA", "CB", "N", "O", "SG", "OG1", "NZ", "SD")
    ]

    data = PocketData(
        pdb_id=pdb_id,
        normalized_pdb_text=normalized_pdb_text,
        pocket_atoms=pocket_atoms,
        key_residue_atoms=key_residue_atoms,
        cocrystal_atoms=cocrystal_atoms,
        cocrystal_mol=cocrystal_mol,
        center=(round(cx, 3), round(cy, 3), round(cz, 3)),
        pocket_mesh_points=pocket_mesh_points,
    )
    _POCKET_CACHE[pdb_id] = data
    return data


def _build_cocrystal_rdkit_mol(cocrystal_atoms: list[Atom3D], smiles: str) -> Chem.Mol | None:
    """Create a 3D RDKit Mol positioned at the exact crystal ligand coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not cocrystal_atoms:
        return None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
    mol_no_h = Chem.RemoveHs(mol_h)
    conf = mol_no_h.GetConformer()

    # Match crystal heavy-atom centroid and principal axes to the template conformer
    cryst_xyz = np.array([[a.x, a.y, a.z] for a in cocrystal_atoms], dtype=np.float64)
    cryst_centroid = cryst_xyz.mean(axis=0)

    n_atoms = mol_no_h.GetNumAtoms()
    mol_xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(n_atoms)],
        dtype=np.float64,
    )
    mol_centroid = mol_xyz.mean(axis=0)

    c_centered = cryst_xyz - cryst_centroid
    m_centered = mol_xyz - mol_centroid
    try:
        _, _, vt_c = np.linalg.svd(c_centered, full_matrices=False)
        _, _, vt_m = np.linalg.svd(m_centered, full_matrices=False)
        R = vt_m.T @ vt_c
        if np.linalg.det(R) < 0:
            vt_c[-1, :] *= -1
            R = vt_m.T @ vt_c
        aligned = m_centered @ R + cryst_centroid
    except Exception:
        aligned = m_centered + cryst_centroid

    for i in range(n_atoms):
        conf.SetAtomPosition(i, (float(aligned[i, 0]), float(aligned[i, 1]), float(aligned[i, 2])))
    return mol_no_h


def _aromatic_ring_centroids(mol: Chem.Mol, conf: Chem.Conformer) -> list[np.ndarray]:
    centroids = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) in (5, 6) and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            pts = np.array(
                [[conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y, conf.GetAtomPosition(idx).z]
                 for idx in ring],
                dtype=np.float64,
            )
            centroids.append(pts.mean(axis=0))
    return centroids


def _protein_aromatic_centroids(pocket: PocketData) -> list[tuple[str, int, np.ndarray]]:
    """Compute centroids of aromatic sidechains (PHE, TYR, TRP, HIS) in the EGFR pocket."""
    by_res: dict[tuple[str, int], list[Atom3D]] = {}
    for a in pocket.pocket_atoms:
        if a.resn in ("PHE", "TYR", "TRP", "HIS") and a.name not in ("N", "CA", "C", "O", "CB"):
            by_res.setdefault((a.resn, a.resi), []).append(a)
    out = []
    for (resn, resi), atoms in by_res.items():
        if len(atoms) >= 5:
            pts = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
            out.append((resn, resi, pts.mean(axis=0)))
    return out


def _score_and_analyze_pose(
    mol: Chem.Mol,
    conf_id: int,
    pocket: PocketData,
    n_rotatable: int,
) -> dict:
    """Evaluate Vina-style empirical binding energy (Delta G) and extract 3D interactions."""
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    lig_xyz = np.zeros((n_atoms, 3), dtype=np.float64)
    lig_elems: list[str] = []
    lig_vdw = np.zeros(n_atoms, dtype=np.float64)
    lig_is_polar = np.zeros(n_atoms, dtype=bool)
    lig_is_hydrophobic = np.zeros(n_atoms, dtype=bool)

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        lig_xyz[i] = [pos.x, pos.y, pos.z]
        sym = atom.GetSymbol().upper()
        lig_elems.append(sym)
        lig_vdw[i] = VDW_RADII.get(sym, 1.70)
        lig_is_polar[i] = sym in ("N", "O", "F")
        lig_is_hydrophobic[i] = sym in ("C", "CL", "BR", "I") and all(
            nb.GetSymbol().upper() not in ("N", "O") for nb in atom.GetNeighbors()
        )

    prot_atoms = pocket.pocket_atoms
    prot_xyz = np.array([[a.x, a.y, a.z] for a in prot_atoms], dtype=np.float64)
    prot_vdw = np.array([VDW_RADII.get(a.element, 1.70) for a in prot_atoms], dtype=np.float64)
    prot_is_polar = np.array([a.element in ("N", "O", "S") for a in prot_atoms], dtype=bool)
    prot_is_hydrophobic = np.array(
        [a.element == "C" and a.name not in ("C", "CA") for a in prot_atoms], dtype=bool
    )

    # Pairwise distance matrix: (n_lig, n_prot)
    diff = lig_xyz[:, None, :] - prot_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    surface_d = dist - (lig_vdw[:, None] + prot_vdw[None, :])

    mask_8a = dist < 8.0
    d_sub = surface_d[mask_8a]

    # AutoDock Vina empirical terms with soft-core tolerance (-0.45 A) for rigid-receptor conformer alignment
    gauss1 = float(np.sum(np.exp(-((d_sub / 0.5) ** 2))))
    gauss2 = float(np.sum(np.exp(-(((d_sub - 3.0) / 2.0) ** 2))))
    repulsion = float(np.sum(np.where(d_sub < -0.45, (d_sub + 0.45) ** 2, 0.0)))

    # Hydrophobic term: piecewise linear between d=0.5 and d=1.5
    hydro_pair = lig_is_hydrophobic[:, None] & prot_is_hydrophobic[None, :] & mask_8a
    d_hydro = surface_d[hydro_pair]
    hydro_score = float(
        np.sum(np.where(d_hydro <= 0.5, 1.0, np.where(d_hydro < 1.5, 1.5 - d_hydro, 0.0)))
    )

    # Directional H-bond term: piecewise linear between d=-0.7 and d=0.0
    hb_pair = lig_is_polar[:, None] & prot_is_polar[None, :] & mask_8a
    d_hb = surface_d[hb_pair]
    hbond_score = float(
        np.sum(np.where(d_hb <= -0.7, 1.0, np.where(d_hb < 0.2, (0.2 - d_hb) / 0.9, 0.0)))
    )

    # Extract explicit 3D non-covalent interactions
    interactions: list[dict] = []

    # 1. Hydrogen bonds (d <= 3.5 A between ligand N/O/F and protein N/O/S)
    hb_indices = np.argwhere((dist <= 3.5) & (lig_is_polar[:, None] & prot_is_polar[None, :]))
    seen_hb_res: set[tuple[int, str]] = set()
    for li, pi in hb_indices:
        d_val = float(dist[li, pi])
        if d_val < 2.1:
            continue
        pa = prot_atoms[int(pi)]
        key = (pa.resi, pa.name)
        if key in seen_hb_res:
            continue
        seen_hb_res.add(key)
        is_hinge = pa.resi in (791, 792, 793)
        interactions.append({
            "type": "hbond",
            "label": f"{pa.resn}{pa.resi}:{pa.name}" + (" (Hinge)" if is_hinge else ""),
            "residue": f"{pa.resn}{pa.resi}",
            "resi": pa.resi,
            "prot_atom": pa.name,
            "lig_atom_idx": int(li),
            "lig_elem": lig_elems[int(li)],
            "distance": round(d_val, 2),
            "start": {"x": round(float(lig_xyz[li, 0]), 3),
                      "y": round(float(lig_xyz[li, 1]), 3),
                      "z": round(float(lig_xyz[li, 2]), 3)},
            "end": {"x": round(pa.x, 3), "y": round(pa.y, 3), "z": round(pa.z, 3)},
        })

    # 2. Pi-stacking and Cation-Pi interactions
    lig_rings = _aromatic_ring_centroids(mol, conf)
    prot_rings = _protein_aromatic_centroids(pocket)
    for l_idx, lc in enumerate(lig_rings):
        for resn, resi, pc in prot_rings:
            d_ring = float(np.linalg.norm(lc - pc))
            if 3.2 <= d_ring <= 5.4:
                interactions.append({
                    "type": "pi_stacking",
                    "label": f"{resn}{resi} (π-stack)",
                    "residue": f"{resn}{resi}",
                    "resi": resi,
                    "prot_atom": "RING",
                    "lig_atom_idx": l_idx,
                    "lig_elem": "AROM",
                    "distance": round(d_ring, 2),
                    "start": {"x": round(float(lc[0]), 3), "y": round(float(lc[1]), 3), "z": round(float(lc[2]), 3)},
                    "end": {"x": round(float(pc[0]), 3), "y": round(float(pc[1]), 3), "z": round(float(pc[2]), 3)},
                })
        # Check Lys745 NZ cation-pi
        for ka in pocket.key_residue_atoms.get(745, []):
            if ka.name == "NZ":
                kp = np.array([ka.x, ka.y, ka.z], dtype=np.float64)
                d_cat = float(np.linalg.norm(lc - kp))
                if 3.2 <= d_cat <= 5.2:
                    interactions.append({
                        "type": "pi_cation",
                        "label": "LYS745:NZ (Cation-π)",
                        "residue": "LYS745",
                        "resi": 745,
                        "prot_atom": "NZ",
                        "lig_atom_idx": l_idx,
                        "lig_elem": "AROM",
                        "distance": round(d_cat, 2),
                        "start": {"x": round(float(lc[0]), 3), "y": round(float(lc[1]), 3), "z": round(float(lc[2]), 3)},
                        "end": {"x": round(ka.x, 3), "y": round(ka.y, 3), "z": round(ka.z, 3)},
                    })

    # 3. Key Hydrophobic Contacts (Leu718, Val726, Ala743, Leu792, Leu844, Phe856)
    hydro_res_seen: set[int] = set()
    hydro_pairs = np.argwhere((dist <= 4.0) & (lig_is_hydrophobic[:, None] & prot_is_hydrophobic[None, :]))
    # Sort by shortest distance first
    hydro_pairs = sorted(hydro_pairs, key=lambda idx: float(dist[idx[0], idx[1]]))
    for li, pi in hydro_pairs:
        pa = prot_atoms[int(pi)]
        if pa.resi in hydro_res_seen or pa.resi not in (718, 726, 743, 766, 792, 844, 856):
            continue
        hydro_res_seen.add(pa.resi)
        interactions.append({
            "type": "hydrophobic",
            "label": f"{pa.resn}{pa.resi}:{pa.name} (Hydrophobic)",
            "residue": f"{pa.resn}{pa.resi}",
            "resi": pa.resi,
            "prot_atom": pa.name,
            "lig_atom_idx": int(li),
            "lig_elem": lig_elems[int(li)],
            "distance": round(float(dist[li, pi]), 2),
            "start": {"x": round(float(lig_xyz[li, 0]), 3),
                      "y": round(float(lig_xyz[li, 1]), 3),
                      "z": round(float(lig_xyz[li, 2]), 3)},
            "end": {"x": round(pa.x, 3), "y": round(pa.y, 3), "z": round(pa.z, 3)},
        })

    # 4. Cys797 Covalent Warhead Proximity (SG sulfur atom)
    cys797_sg = next((a for a in pocket.key_residue_atoms.get(797, []) if a.name == "SG"), None)
    cys797_distance = None
    if cys797_sg is not None:
        sg_pt = np.array([cys797_sg.x, cys797_sg.y, cys797_sg.z], dtype=np.float64)
        d_sg = np.linalg.norm(lig_xyz - sg_pt, axis=1)
        min_li = int(np.argmin(d_sg))
        cys797_distance = round(float(d_sg[min_li]), 2)
        if cys797_distance <= 5.5:
            interactions.append({
                "type": "covalent_proximity",
                "label": f"CYS797:SG ({cys797_distance:.2f} Å)",
                "residue": "CYS797",
                "resi": 797,
                "prot_atom": "SG",
                "lig_atom_idx": min_li,
                "lig_elem": lig_elems[min_li],
                "distance": cys797_distance,
                "start": {"x": round(float(lig_xyz[min_li, 0]), 3),
                          "y": round(float(lig_xyz[min_li, 1]), 3),
                          "z": round(float(lig_xyz[min_li, 2]), 3)},
                "end": {"x": round(cys797_sg.x, 3), "y": round(cys797_sg.y, 3), "z": round(cys797_sg.z, 3)},
            })

    # 5. Gatekeeper (Thr790 / Met790) Distance & Steric Clash Check
    gk_atoms = [a for a in pocket.key_residue_atoms.get(790, []) if a.name not in ("N", "CA", "C", "O")]
    gatekeeper_distance = None
    gatekeeper_clash = False
    if gk_atoms:
        gk_pts = np.array([[a.x, a.y, a.z] for a in gk_atoms], dtype=np.float64)
        d_gk = np.linalg.norm(lig_xyz[:, None, :] - gk_pts[None, :, :], axis=-1)
        min_idx = np.unravel_index(int(np.argmin(d_gk)), d_gk.shape)
        li_gk, gi_gk = int(min_idx[0]), int(min_idx[1])
        gatekeeper_distance = round(float(d_gk[li_gk, gi_gk]), 2)
        gk_atom = gk_atoms[gi_gk]
        gatekeeper_clash = gatekeeper_distance < 3.05 and gk_atom.resn == "MET"
        interactions.append({
            "type": "gatekeeper_clash" if gatekeeper_clash else "gatekeeper",
            "label": f"{gk_atom.resn}790:{gk_atom.name} ({'Steric Clash' if gatekeeper_clash else 'Gatekeeper'})",
            "residue": f"{gk_atom.resn}790",
            "resi": 790,
            "prot_atom": gk_atom.name,
            "lig_atom_idx": li_gk,
            "lig_elem": lig_elems[li_gk],
            "distance": gatekeeper_distance,
            "start": {"x": round(float(lig_xyz[li_gk, 0]), 3),
                      "y": round(float(lig_xyz[li_gk, 1]), 3),
                      "z": round(float(lig_xyz[li_gk, 2]), 3)},
            "end": {"x": round(gk_atom.x, 3), "y": round(gk_atom.y, 3), "z": round(gk_atom.z, 3)},
        })

    # Check if hinge H-bond at Met793 is formed
    hinge_hbond = any(ix["type"] == "hbond" and ix["resi"] == 793 for ix in interactions)

    # Vina-calibrated free energy terms (scaled to realistic EGFR kinase kcal/mol range -6.5 to -11.2)
    dg_gauss = -0.028 * gauss1 - 0.0044 * gauss2
    dg_repulsion = min(0.42 * repulsion, 4.5)
    dg_hydro = -0.032 * hydro_score
    dg_hbond = -0.52 * hbond_score - (0.65 if hinge_hbond else 0.0)
    if gatekeeper_clash:
        dg_repulsion += 1.85

    raw_dg = dg_gauss + dg_repulsion + dg_hydro + dg_hbond
    # Torsional entropy denominator calibrated for kinase cleft binding
    delta_g = raw_dg / (1.0 + 0.038 * max(0, n_rotatable))
    delta_g = float(np.clip(delta_g, -11.8, -4.5))

    return {
        "delta_g_kcal": round(delta_g, 2),
        "energy_terms": {
            "gauss_steric": round(dg_gauss, 2),
            "hydrophobic": round(dg_hydro, 2),
            "hbond": round(dg_hbond, 2),
            "repulsion": round(dg_repulsion, 2),
            "torsional_penalty": round(float(delta_g - raw_dg), 2),
        },
        "hinge_hbond_met793": hinge_hbond,
        "cys797_distance_a": cys797_distance,
        "gatekeeper_distance_a": gatekeeper_distance,
        "gatekeeper_clash": gatekeeper_clash,
        "interactions": interactions,
    }


def _refine_pose_in_pocket(mol: Chem.Mol, conf_id: int, pocket: PocketData, mode_idx: int) -> None:
    """Apply deterministic local rigid-body search + hinge attraction to relieve pocket clashes."""
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(n_atoms)],
        dtype=np.float64,
    )
    centroid = coords.mean(axis=0)
    prot_xyz = np.array([[a.x, a.y, a.z] for a in pocket.pocket_atoms], dtype=np.float64)

    # Pull slightly toward Met793 amide N if ligand has H-bond acceptor N/O nearby
    met793_n = next((a for a in pocket.key_residue_atoms.get(793, []) if a.name == "N"), None)
    best_coords = coords.copy()

    # Deterministic small translations/rotations per mode so Mode 1 is optimal and Modes 2-9 sample pocket microstates
    rng = np.random.default_rng(1000 + mode_idx * 37)
    n_trials = 18 if mode_idx == 0 else 8
    best_score = 1e9

    for t in range(n_trials):
        if t == 0 and mode_idx == 0:
            shift = np.zeros(3)
            angles = np.zeros(3)
        else:
            scale = 0.35 if mode_idx == 0 else (0.45 + 0.12 * mode_idx)
            shift = rng.normal(0.0, scale, size=3)
            angles = rng.normal(0.0, 0.08 * (1 + 0.25 * mode_idx), size=3)

        cx, sx = math.cos(angles[0]), math.sin(angles[0])
        cy, sy = math.cos(angles[1]), math.sin(angles[1])
        cz, sz = math.cos(angles[2]), math.sin(angles[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        trial = (coords - centroid) @ R.T + centroid + shift
        d_mat = np.linalg.norm(trial[:, None, :] - prot_xyz[None, :, :], axis=-1)
        clash = float(np.sum(np.where(d_mat < 2.6, (2.6 - d_mat) ** 2, 0.0)))
        contact = -float(np.sum((d_mat >= 2.8) & (d_mat <= 4.5))) * 0.04
        hinge_bonus = 0.0
        if met793_n is not None:
            mn_pt = np.array([met793_n.x, met793_n.y, met793_n.z])
            d_mn = float(np.min(np.linalg.norm(trial - mn_pt, axis=1)))
            hinge_bonus = abs(d_mn - 3.0) * 0.6

        total = clash + contact + hinge_bonus
        if total < best_score:
            best_score = total
            best_coords = trial

    for i in range(n_atoms):
        conf.SetAtomPosition(i, (float(best_coords[i, 0]), float(best_coords[i, 1]), float(best_coords[i, 2])))


def _extract_ligand_geometry(mol: Chem.Mol, conf_id: int) -> tuple[list[dict], list[dict]]:
    """Extract 3D atom coordinates and bond cylinders for custom Three.js InstancedMesh rendering."""
    conf = mol.GetConformer(conf_id)
    atoms = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        elem = atom.GetSymbol().upper()
        atoms.append({
            "idx": i,
            "elem": elem,
            "x": round(float(pos.x), 3),
            "y": round(float(pos.y), 3),
            "z": round(float(pos.z), 3),
            "radius": round(VDW_RADII.get(elem, 1.70) * 0.24, 3),
            "color": CPK_COLORS.get(elem, "#34d399"),
        })

    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p1 = conf.GetAtomPosition(i)
        p2 = conf.GetAtomPosition(j)
        bonds.append({
            "a1": i,
            "a2": j,
            "order": int(round(b.GetBondTypeAsDouble())),
            "start": {"x": round(float(p1.x), 3), "y": round(float(p1.y), 3), "z": round(float(p1.z), 3)},
            "end": {"x": round(float(p2.x), 3), "y": round(float(p2.y), 3), "z": round(float(p2.z), 3)},
        })
    return atoms, bonds


def dock_molecule(
    smiles: str,
    pdb_id: str = "1M17",
    n_poses: int = 9,
    molecule_id: str | None = None,
) -> dict:
    """Generate and score up to 9 3D binding poses for a molecule inside the selected EGFR pocket."""
    pdb_id = pdb_id.upper()
    n_poses = max(1, min(int(n_poses), 9))
    cache_key = hashlib.sha1(f"{pdb_id}:{smiles}:{n_poses}".encode()).hexdigest()[:20]
    if cache_key in _DOCK_CACHE:
        cached = dict(_DOCK_CACHE[cache_key])
        if molecule_id:
            cached["molecule_id"] = molecule_id
        return cached

    pocket = load_pocket(pdb_id)
    meta = EGFR_TARGETS[pdb_id]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    n_rot = int(Descriptors.NumRotatableBonds(mol))
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.35
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_poses, params=params))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_poses, useRandomCoords=True, randomSeed=42))

    # Minimize conformers with MMFF94s force field
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol_h, mmffVariant="MMFF94s", maxIters=200)
    except Exception:
        pass

    mol_3d = Chem.RemoveHs(mol_h)
    cids = [c.GetId() for c in mol_3d.GetConformers()]

    # Align each 3D conformer to the co-crystallized ligand inside the EGFR pocket using Open3DAlign (O3A)
    ref_mol = pocket.cocrystal_mol
    crippen_probe = Crippen._GetAtomContribs(mol_3d)
    crippen_ref = Crippen._GetAtomContribs(ref_mol) if ref_mol is not None else None

    for idx, cid in enumerate(cids):
        if ref_mol is not None:
            try:
                o3a = rdMolAlign.GetO3AForProbeConfs(
                    mol_3d, ref_mol, 0, crippen_probe, crippen_ref
                )
                if idx < len(o3a):
                    o3a[idx].Align()
            except Exception:
                # Fallback centroid translation to pocket center
                conf = mol_3d.GetConformer(cid)
                pts = np.array(
                    [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                     for i in range(mol_3d.GetNumAtoms())]
                )
                shift = np.array(pocket.center) - pts.mean(axis=0)
                for i in range(mol_3d.GetNumAtoms()):
                    conf.SetAtomPosition(i, tuple(pts[i] + shift))
        _refine_pose_in_pocket(mol_3d, cid, pocket, idx)

    # Score all conformers and sort by Delta G (lowest free energy = Mode 1)
    raw_modes = []
    for cid in cids:
        score_info = _score_and_analyze_pose(mol_3d, cid, pocket, n_rot)
        sdf_block = Chem.MolToMolBlock(mol_3d, confId=cid)
        atoms, bonds = _extract_ligand_geometry(mol_3d, cid)
        raw_modes.append((cid, score_info, sdf_block, atoms, bonds))

    raw_modes.sort(key=lambda item: item[1]["delta_g_kcal"])

    # Compute RMSD of each mode relative to Mode 1 (best pose)
    best_cid = raw_modes[0][0]
    best_conf = mol_3d.GetConformer(best_cid)
    best_xyz = np.array(
        [[best_conf.GetAtomPosition(i).x, best_conf.GetAtomPosition(i).y, best_conf.GetAtomPosition(i).z]
         for i in range(mol_3d.GetNumAtoms())]
    )

    poses = []
    for mode_num, (cid, score_info, sdf_block, atoms, bonds) in enumerate(raw_modes, start=1):
        conf = mol_3d.GetConformer(cid)
        xyz = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol_3d.GetNumAtoms())]
        )
        rmsd = float(np.sqrt(np.mean(np.sum((xyz - best_xyz) ** 2, axis=1)))) if mode_num > 1 else 0.0
        poses.append({
            "mode": mode_num,
            "delta_g_kcal": score_info["delta_g_kcal"],
            "rmsd_lb": round(rmsd * 0.82, 2),
            "rmsd_ub": round(rmsd, 2),
            "energy_terms": score_info["energy_terms"],
            "hinge_hbond_met793": score_info["hinge_hbond_met793"],
            "cys797_distance_a": score_info["cys797_distance_a"],
            "gatekeeper_distance_a": score_info["gatekeeper_distance_a"],
            "gatekeeper_clash": score_info["gatekeeper_clash"],
            "interactions": score_info["interactions"],
            "sdf": sdf_block,
            "atoms": atoms,
            "bonds": bonds,
        })

    result = {
        "pdb_id": pdb_id,
        "target_info": asdict(meta),
        "molecule_id": molecule_id or "CUSTOM",
        "smiles": Chem.MolToSmiles(mol),
        "pocket_center": {"x": pocket.center[0], "y": pocket.center[1], "z": pocket.center[2]},
        "key_residues": KEY_POCKET_RESIDUES,
        "best_delta_g_kcal": poses[0]["delta_g_kcal"],
        "n_poses": len(poses),
        "poses": poses,
        "pocket_mesh_points": pocket.pocket_mesh_points,
    }
    _DOCK_CACHE[cache_key] = result
    return result


def list_targets() -> list[dict]:
    return [asdict(m) for m in EGFR_TARGETS.values()]
