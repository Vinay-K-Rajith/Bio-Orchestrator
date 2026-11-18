/**
 * The wire contract with the FastAPI backend.
 *
 * Mirrors `backend/app/pipeline.py`, `backend/app/orchestrator.py`, and
 * `backend/app/docking.py` exactly.
 */

export type FunnelStage = 'pool' | 'potent' | 'safe' | 'leads'

export type EgfrPdbId = '1M17' | '2ITY' | '4I22' | '4ZAU' | '5D41'

export interface ScreeningParams {
  target: string
  pdb_id?: EgfrPdbId | string
  max_ic50_nm: number
  require_lipinski: boolean
  max_structural_alerts: number
  require_in_domain: boolean
  min_similarity: number
  top_n: number
  weight_affinity: number
  weight_confidence: number
  weight_safety: number
  weight_druglikeness: number
  source: string
  rationale: string
}

export interface Desirability {
  affinity: number
  confidence: number
  safety: number
  drug_likeness: number
}

export interface Lead {
  rank: number
  molecule_id: string
  name: string | null
  smiles: string
  scaffold?: string

  pic50: number
  pic50_lo: number
  pic50_hi: number
  ic50_nm: number
  ic50_nm_lo: number
  ic50_nm_hi: number
  interval_width: number

  krr_pic50?: number
  chemberta_pic50?: number
  ensemble_pic50?: number

  pdb_id?: string
  docking_score_kcal?: number | null
  hinge_hbond?: boolean
  cys797_distance_a?: number | null
  gatekeeper_distance_a?: number | null
  gatekeeper_clash?: boolean
  interaction_count?: number

  mw: number
  logp: number
  hbd: number
  hba: number
  tpsa: number
  qed: number
  sa_score: number
  rotatable_bonds: number

  lipinski_violations: number
  lipinski_failed: string[]
  veber_pass: boolean
  structural_alerts: string[]
  n_alerts: number

  max_similarity: number
  in_domain: boolean
  /** null when the hERG classifier is not loaded — not zero, not "safe". */
  herg_risk: number | null

  utility: number
  pareto_front: number
  desirability: Desirability
  flags: string[]
}

export interface DroppedCounts {
  lipinski?: number
  alerts?: number
  out_of_domain?: number
}

export interface FunnelCounts {
  pool: number
  potent: number
  safe: number
  leads: number
  dropped: DroppedCounts
}

export interface ParetoSummary {
  n: number
  n_fronts: number
  front_sizes: number[]
}

export interface ModelSummary {
  scaffold_r2: number
  scaffold_rmse: number
  random_r2?: number
  conformal_coverage: number
  conformal_alpha?: number
  herg_available: boolean
  herg_auc?: number | null
  ensemble_r2?: number
  ensemble_rmse?: number
  pretrained_model?: string
}

export interface Timings {
  stage1?: number
  stage2?: number
  stage3?: number
}

export interface ScreenResult {
  target: string
  pdb_id?: string
  parameters: ScreeningParams
  funnel: FunnelCounts
  pareto: ParetoSummary
  leads: Lead[]
  model: ModelSummary
  timings: Timings
  total_seconds: number
  query?: string
}

export interface SurrogateStats {
  scaffold_r2: number
  scaffold_rmse: number
  random_r2: number
  conformal_coverage: number
  n_training_compounds: number
  n_training_scaffolds: number
  ensemble_r2?: number
  ensemble_rmse?: number
  ensemble_spearman?: number
  pretrained_model?: string
}

export interface HealthModels {
  pool_size: number
  herg_model: boolean
  load_seconds: number
  surrogate: SurrogateStats
  herg: Record<string, number> | null
  pool: Record<string, unknown>
  targets?: string[]
}

export interface Health {
  status: 'ok' | 'loading'
  llm_backends: {
    llama_cpp: boolean
    rules: boolean
    nebius?: boolean
    nebius_configured?: boolean
    nebius_model?: string
    tavily?: boolean
  }
  models: HealthModels | null
}

/* ---- 3D EGFR Docking & Structural Biology Types ------------------------ */

export interface TargetMetadata {
  pdb_id: EgfrPdbId
  title: string
  conformation: string
  generation: string
  cocrystal_ligand_id: string
  cocrystal_name: string
  cocrystal_smiles: string
  gatekeeper_residue: string
  hinge_residue: string
  covalent_residue: string
  numbering_note: string
  residue_offset: number
  key_residues_summary: string
  clinical_context: string
}

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface DockInteraction {
  type: 'hbond' | 'pi_stacking' | 'pi_cation' | 'hydrophobic' | 'covalent_proximity' | 'gatekeeper' | 'gatekeeper_clash'
  label: string
  residue: string
  resi: number
  prot_atom: string
  lig_atom_idx: number
  lig_elem: string
  distance: number
  start: Vec3
  end: Vec3
}

export interface LigandAtom3D {
  idx: number
  elem: string
  x: number
  y: number
  z: number
  radius: number
  color: string
}

export interface LigandBond3D {
  a1: number
  a2: number
  order: number
  start: Vec3
  end: Vec3
}

export interface DockPose {
  mode: number
  delta_g_kcal: number
  rmsd_lb: number
  rmsd_ub: number
  energy_terms: {
    gauss_steric: number
    hydrophobic: number
    hbond: number
    repulsion: number
    torsional_penalty: number
  }
  hinge_hbond_met793: boolean
  cys797_distance_a: number | null
  gatekeeper_distance_a: number | null
  gatekeeper_clash: boolean
  interactions: DockInteraction[]
  sdf: string
  atoms: LigandAtom3D[]
  bonds: LigandBond3D[]
}

export interface PocketMeshPoint {
  x: number
  y: number
  z: number
  elem: string
  resn: string
  resi: number
  atom: string
  color: string
}

export interface DockResponse {
  pdb_id: EgfrPdbId
  target_info: TargetMetadata
  molecule_id: string
  smiles: string
  pocket_center: Vec3
  key_residues: number[]
  best_delta_g_kcal: number
  n_poses: number
  poses: DockPose[]
  pocket_mesh_points: PocketMeshPoint[]
}

/* ---- Server-sent events ------------------------------------------------ */

export interface StageEvent {
  stage: FunnelStage
  count: number
  message: string
  seconds?: number
  cutoff_pic50?: number
  cutoff_ic50_nm?: number
  dropped?: DroppedCounts
  pareto_fronts?: ParetoSummary
}

/* ---- Literature grounding ---------------------------------------------- */

export interface Citation {
  id: string
  title: string
  url: string
  snippet: string
  score: number
  read_in_full: boolean
  corpus: string
}

export interface EvidenceTraceEntry {
  tool: string
  query?: string
  urls?: string[]
  results?: number
  corpus?: string
  url?: string
  source_ids?: string[]
  backend?: string
  error?: string
}

export interface EvidenceBundle {
  citations: Citation[]
  trace: EvidenceTraceEntry[]
  api_calls: number
  cache_hits: number
  credits: number
  corpora: string[]
  available: boolean
  note: string
}

export interface ReportEvent {
  report: string
  source: string
}

export type ScreenPhase = 'idle' | 'parsing' | 'screening' | 'reporting' | 'done' | 'error'
