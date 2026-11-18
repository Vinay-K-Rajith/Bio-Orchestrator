# BioOrchestrator — Technical Documentation

Single reference for every technical element of the project, as of 2026-10-15.
Source of truth is the code; where the older planning documents disagree
(`TECHNICAL_PLAN.md`, `HANDOFF.md`), this document follows the code.

---

## 1. Overview

BioOrchestrator turns a natural-language screening request into a ranked,
uncertainty-quantified shortlist of EGFR inhibitor candidates, with an optional
literature / patent / clinical-news check on each lead. A screen over the cached
pool runs in roughly 30 ms.

### The four questions it answers

| # | Question | Answered by |
|---|---|---|
| 1 | Will it bind? | XGBoost affinity surrogate + conformal 90% interval |
| 2 | Will it be toxic? | hERG classifier, PAINS/Brenk/NIH structural alerts |
| 3 | Is it already known or claimed? | Literature + patent search (Tavily agent) |
| 4 | Has the chemotype already failed? | Clinical news + regulator search (Tavily agent) |

Questions 1–2 are modelled. Questions 3–4 are search problems and run only when
a request sets `ground=true`.

### Core design principle

**The LLM never touches a number.** A language model sits at the two ends of the
pipeline only: going in, it converts text to validated parameters; coming out, it
writes a report and cites sources. Predictions, filters and ranking are
deterministic code, so the science is identical whether an LLM ran or not.
This is enforced structurally:

- LLM-produced parameters are clamped to validated ranges (`ScreeningParams.clamp`).
- `evidence.py` does not import the surrogate, filters or ranker, and a test fails if that changes.
- Citation markers that do not resolve to a retrieved source are stripped before the report ships.

### Technology stack

| Layer | Technology |
|---|---|
| Cheminformatics | RDKit (Morgan fingerprints, descriptors, PAINS/Brenk/NIH catalogs, QED, SA score) |
| Models | XGBoost (point regressor, two quantile regressors, hERG classifier) |
| Applicability domain | Tanimoto nearest neighbour via NumPy matrix multiply |
| Ranking | NSGA-II non-dominated sort + crowding distance + weighted desirability |
| API | FastAPI, Pydantic, server-sent events (SSE) |
| LLM | Nemotron 3 Super 120B on Nebius Token Factory, local llama.cpp, or rule-based parser |
| Grounding | Tavily search / extract / crawl with on-disk cache |
| Frontend | React 18, TypeScript, Vite 6, Tailwind CSS 3, smiles-drawer |
| Data | ChEMBL 37 (EGFR = CHEMBL203, hERG = CHEMBL240) |

### Naming and target history

- The project was renamed from "Drug Detection" to **BioOrchestrator** on 2026-09-24. Cheminformatics terms such as *drug-likeness* are kept; `drug_likeness` / `weight_druglikeness` are live API fields.
- The training set was originally believed to be CDK2. It is **EGFR (CHEMBL203)**: 100% of the 2,765 compounds have EGFR IC50 records, 90.4% of labels equal the EGFR median exactly, and only 2.4% have any CDK2 record. Code, models and API are relabelled to EGFR. `TECHNICAL_PLAN.md` and `HANDOFF.md` keep their CDK2 wording on purpose, under a "superseded" banner. Evidence: `backend/experiments/audit/AUDIT.md`.

---

## 2. Architecture

### Repository layout

```
BioOrchestrator/
├── README.md, TECHNICAL_PLAN.md, HANDOFF.md, TECHNICAL_DOCUMENTATION.md
├── setup.ps1                      one-shot Windows build
├── .env.example                   all config, all optional
├── egfr_chembl_labeled_dataset.csv   training data (staged to backend/data/)
├── backend/
│   ├── app/
│   │   ├── __init__.py            loads .env before any module reads os.environ
│   │   ├── featurize.py           SMILES -> feature vector, FEATURE_VERSION
│   │   ├── splits.py              scaffold / random splitting
│   │   ├── predictor.py           affinity inference + conformal interval
│   │   ├── bio_engine.py          properties, alerts, applicability domain
│   │   ├── ranker.py              desirability functions, NSGA-II
│   │   ├── pipeline.py            ScreeningPipeline: the 3-stage funnel, HergModel
│   │   ├── orchestrator.py        Bookend A (parse) and Bookend B (report)
│   │   ├── llm_backends.py        Nebius / llama.cpp transport
│   │   ├── evidence.py            Tavily agent, citations, marker validation
│   │   └── server.py              FastAPI app
│   ├── scripts/                   train_surrogate, train_herg, fetch_chembl,
│   │                              build_pool_cache, verify, retry_kinase_pool.sh
│   ├── experiments/               benchmarks and audits (not production)
│   ├── tests/test_evidence.py     32 offline tests
│   ├── data/                      ChEMBL CSVs (gitignored)
│   └── models/                    trained artifacts + pool_cache (gitignored)
└── frontend/                      Vite + React + TypeScript SPA
```

### Component diagram

```
 Browser (React SPA, :5173)
   │  GET /api/screen/stream?query=…&ground=…      (SSE; Vite proxies /api → :8000)
   ▼
 FastAPI (server.py, :8000)
   │ 1. parse_query ──► llm_backends ──► Nebius | llama.cpp | (rules)      Bookend A
   │         └─ ScreeningParams.clamp()
   │ 2. ScreeningPipeline.screen(params)           deterministic, no LLM
   │         Stage 1 threshold on cached pool pIC50
   │         Stage 2 Lipinski / alerts / domain filters
   │         Stage 3 Pareto + desirability ranking
   │ 3. gather_evidence (only if ground=true) ──► Tavily + LLM tool loop
   │ 4. generate_report ──► LLM | deterministic template                    Bookend B
   ▼
 SSE events: params → stage×3 → result → [evidence_start → evidence] → report → done
```

### Request lifecycle (streaming route)

1. `params` — parsed, clamped `ScreeningParams`.
2. `stage` ×4 — `pool`, `potent`, `safe`, `leads`; each carries `count`, `message`, timing. The server inserts a 0.25 s pause between events so the funnel animation is legible.
3. `result` — the full numeric result.
4. `evidence_start` / `evidence` — only when `ground=true`.
5. `report` — text + `source` (which backend wrote it).
6. `done` — `total_seconds`. Any exception becomes an `error` event.

### Startup

`lifespan` builds the pipeline once: loads the three XGBoost models, the
applicability-domain reference fingerprints, the hERG model (optional), and the
pool cache, then **scores the entire pool once** (Stage 1 depends only on the
pool, not the query). Every request after that is a threshold on cached arrays.
Startup fails loudly if models or cache are missing or misaligned.

---

## 3. Data and machine learning

### 3.1 Datasets

| Dataset | File | Rows | Source |
|---|---|---|---|
| EGFR affinity | `backend/data/egfr_chembl_labeled.csv` | 2,765 unique compounds, 923 Bemis-Murcko scaffolds | ChEMBL 37, CHEMBL203, IC50 → pIC50 |
| hERG | `backend/data/herg_chembl.csv` | 10,929 compounds, 5,451 scaffolds, 66.6% blockers | ChEMBL CHEMBL240, IC50/Ki/Kd, `=` relation, pChEMBL > 5 = blocker |
| Screening pool (live) | `backend/models/pool_cache/` | 10,000 diverse drug-like ChEMBL molecules | built 2026-09-08 from `candidate_pool_diverse.csv` |
| Kinase pool (partial) | `backend/data/candidate_pool.csv` | ~7.7k rows | incomplete download, not yet cached |

Training-label statistics: mean pIC50 6.03, SD 1.50, range 2.19–11.52.
ChEMBL duplicates are median-aggregated per canonical SMILES so one molecule
cannot sit on both sides of a split with two labels.

### 3.2 Featurization (`featurize.py`)

One place where SMILES becomes numbers. Vector width **2,074** =

- 2,048 Morgan **count** fingerprint (radius 2), `log1p`-compressed. Counts, not bits, because four anilines differ chemically from one.
- 26 curated RDKit descriptors (MolWt, MolLogP, MolMR, TPSA, HBD/HBA, rotatable bonds, ring counts, FractionCSP3, BalabanJ, BertzCT, Chi/Kappa indices, LabuteASA, QED …). Non-finite values become NaN, which XGBoost handles natively.
- A separate **binary** Morgan fingerprint (`morgan_bits`) is used only for Tanimoto similarity, never as model input.

**`FEATURE_VERSION = 2`** is stamped into every model artifact and the pool
manifest. `AffinityPredictor` raises on mismatch rather than silently scoring a
misaligned vector. Changing feature width or order requires bumping it, which
forces retraining both models and rebuilding the pool cache.

Other helpers: `mol_from_smiles` (rejects empty / zero-heavy-atom), `canonical_smiles`, `murcko_scaffold` (acyclic molecules bucketed under their own SMILES), `sa_score` (Ertl, from RDKit Contrib).

### 3.3 Splitting (`splits.py`)

- `scaffold_split` — capacity-based greedy: groups walked largest-first into train until capacity, remainder to test. Cannot invert on degenerate scaffold distributions (pinned by four synthetic tests in `verify.py`).
- `scaffold_kfold` — shuffled keys, greedy bin-packing into the smallest fold; no scaffold crosses a fold boundary.
- `random_kfold` — the optimistic baseline, kept only to report the gap.

### 3.4 Affinity surrogate (`scripts/train_surrogate.py`, `predictor.py`)

Three XGBoost regressors on the 2,074-feature vector:

| Model | Objective | Trees | LR | Notes |
|---|---|---|---|---|
| Point | `reg:squarederror` | 600 | 0.03 | depth 6, subsample 0.8, colsample 0.5, min_child_weight 3, λ=2, α=0.1 |
| Quantile lo | `reg:quantileerror` α=0.05 | 400 | 0.05 | |
| Quantile hi | `reg:quantileerror` α=0.95 | 400 | 0.05 | |

**Conformalized quantile regression** (Romano et al., 2019): out-of-fold lo/hi
quantiles under scaffold 5-fold, conformity score = signed distance outside the
band, global correction = the finite-sample-adjusted (1−α) quantile. At inference
`lo − c`, `hi + c`; ordering is enforced (`lo ≤ point ≤ hi`) because independent
quantile models can cross.

Units: output is **pIC50**; `IC50_nM = 10^(9 − pIC50)`. It is deliberately IC50,
not Kd — converting would need Cheng-Prusoff and assay ATP concentration, which
the dataset lacks.

Artifacts in `backend/models/`: `egfr_xgboost.json`, `egfr_quantile_lo.json`,
`egfr_quantile_hi.json`, `reference_fps.npy`, `reference_pic50.npy`,
`reference_compounds.csv`, `surrogate_metadata.json`, `train_log.txt`.

### 3.5 Measured performance

| Validation | R² | RMSE | Spearman |
|---|---|---|---|
| Random 5-fold | 0.785 | 0.695 | 0.889 |
| **Scaffold 5-fold** | **0.707** | **0.812** | **0.844** |

Optimism gap 0.078 R². Multi-seed scaffold CV in `HANDOFF.md`: 0.7025 ± 0.0052
(noise floor ≈ ±0.005; deltas below ~0.01 are noise).

| Conformal interval | Value |
|---|---|
| Raw quantile coverage | 0.760 |
| After correction | **0.901** (target 0.90) |
| Correction | ±0.333 log units |
| Mean width | 2.75 log units |

External sanity panel (AUDIT.md, scaffold-held-out scores for in-training drugs):
Spearman 0.726 vs measured IC50, MAE 0.94 log. gefitinib 7.62, lapatinib 7.61,
dacomitinib 7.40 and icotinib 7.13 (both fully external), imatinib 4.68,
aspirin 3.53. Osimertinib is the one miss (5.33 vs 7.91) and is correctly
flagged out-of-domain.

### 3.6 hERG classifier (`scripts/train_herg.py`)

XGBoost classifier, 500 trees, depth 6, LR 0.04, `eval_metric=aucpr`,
`scale_pos_weight` set per fold from the class ratio. Scaffold 5-fold CV.
Decision threshold chosen to maximise out-of-fold F1.

| Metric | Value |
|---|---|
| ROC-AUC | 0.799 |
| AUPRC | 0.882 (base rate 0.666 → 1.32× lift) |
| Brier | 0.178 |
| Best F1 | 0.832 at threshold 0.28 |

ChEMBL hERG data is publication-biased toward positives, so AUPRC lift (not
ROC-AUC) is the honest signal. `HergModel` is optional: without it `herg_risk`
is `null` (never 0) and the safety desirability uses alerts alone.

### 3.7 Applicability domain (`bio_engine.ApplicabilityDomain`)

Max Tanimoto similarity of each molecule to the 2,765 training fingerprints,
computed as one matrix multiply: `inter = Q @ Rᵀ`, `union = |Q| + |R| − inter`.
`in_domain = max_similarity ≥ 0.35` (`AD_SIMILARITY_FLOOR`). This is a question
about the *model* (is it entitled to an opinion?), kept separate from toxicity.

### 3.8 Model research findings (not shipped)

From `HANDOFF.md` and `backend/experiments/` (3 seeds × 5-fold scaffold CV):

| Model | R² | RMSE | ρ (actives) |
|---|---|---|---|
| XGBoost baseline (shipped) | 0.7025 ± 0.0052 | 0.818 | 0.614 |
| Tanimoto-kernel ridge alone | 0.7078 ± 0.0056 | 0.811 | 0.616 |
| XGBoost fused features (d=6360) | 0.7124 ± 0.0043 | 0.804 | 0.618 |
| **XGBoost + Tanimoto-KRR, 50/50** | **0.7263 ± 0.0040** | **0.785** | **0.631** |

- The blend is the largest measured gain (+0.024 R²) and comes from decorrelation (residual correlation 0.88 vs 0.96 between XGBoost variants). **It is not in production**: `predictor.py` loads XGBoost only.
- Hyperparameter search (depth 4–8, 600–3000 trees) spans 0.7104–0.7156, inside noise. Feature-family search is saturated; fused set gives +0.011 at ~3× cost.
- Spearman among actives (pIC50 ≥ 6) is only 0.609 — the funnel ranks inside that regime, so 0.84 overstates utility.
- Error scales with distance from training chemistry (RMSE 0.56 at max-Tanimoto ≥ 0.85 vs 1.14 below 0.40) but the conformal band is a single global scalar.
- Not yet run: censoring-aware training (`survival:aft`), similarity-adaptive/Mondrian conformal intervals (KRR variance would give this natively), CheMeleon, TabPFN v2.

---

## 4. Orchestrator pipeline

### 4.1 Funnel (`pipeline.py`)

| Stage | Operation | Cost |
|---|---|---|
| 0 | Load cached pool (10,000) | startup only |
| 1 — Potent | `pic50 ≥ 9 − log10(max_ic50_nm)` on pre-scored pool | vectorised threshold |
| 2 — Safe | per survivor: drop if `lipinski_violations > 1` (when required), `n_alerts > max_structural_alerts`, or (when required) `max_similarity < min_similarity` | Python loop over survivors only |
| 3 — Rank | desirabilities → NSGA-II fronts → utility → top-N | dominates the ~30 ms |

Drop counters (`lipinski`, `alerts`, `out_of_domain`) are reported per run. An
empty stage short-circuits to `_empty_result`.

Note the Lipinski gate tolerates **one** violation (`> 1` drops); zero violations
is rewarded in the desirability score instead.

### 4.2 Parameters (`ScreeningParams`)

| Field | Default | Meaning |
|---|---|---|
| `target` | EGFR | |
| `max_ic50_nm` | 100 | potency cutoff, clamped 0.01–1,000,000 |
| `require_lipinski` | true | |
| `max_structural_alerts` | 0 | clamped 0–10 |
| `require_in_domain` | false | |
| `min_similarity` | 0.35 | clamped 0–1 |
| `top_n` | 5 | clamped 1–50 |
| `weight_affinity / confidence / safety / druglikeness` | 0.40 / 0.20 / 0.25 / 0.15 | renormalised to sum 1 |
| `source`, `rationale` | | which parser ran, and why |

### 4.3 Desirability functions (`ranker.py`)

| Objective | Function |
|---|---|
| Affinity | `clip((pIC50 − 5) / 4, 0, 1)` — linear in log space |
| Confidence | `0.5·clip((4 − width)/3) + 0.5·clip(max_sim/0.6)`, halved if out-of-domain |
| Safety | `0.65^n_alerts × (1 − herg_risk)` (hERG term omitted if model absent) |
| Drug-likeness | `(0.45·ro5 + 0.30·QED + 0.25·SA) × veber`; ro5 = 1.0 / 0.7 / 0.2 for 0 / 1 / ≥2 violations; SA term `clip((6 − SA)/3)`; Veber 1.0 pass / 0.75 fail |

### 4.4 Ranking

Two algorithms used for what each is good at:

- **Pareto (NSGA-II fast non-dominated sort)** answers "which molecules are not beaten on every axis?" — weight-free.
- **Weighted desirability** orders molecules *within* a front.
- Sort key: `(front, −utility, −crowding_distance)`. Front-major ordering means a dominated molecule can never win because the weights favour its one strong axis; crowding distance keeps the top-N from being near-identical analogs (audit: 8 leads spanned 6 scaffolds, median pairwise Tanimoto 0.32).

### 4.5 Lead object

Each lead carries identifiers, `pic50` with `pic50_lo/hi`, `ic50_nm` with bounds,
`interval_width`, MW/LogP/HBD/HBA/TPSA/QED/SA/rotatable bonds, Lipinski and Veber
results, structural alerts, `max_similarity`, `in_domain`, `herg_risk`,
`utility`, `pareto_front`, per-axis `desirability`, and a `flags` list. Flags are
emitted for: out of domain, interval > 2.5 log units, structural alerts, hERG
above threshold, Lipinski violations, SA > 6.

### 4.6 Pool cache (`scripts/build_pool_cache.py`)

RDKit parsing is the bottleneck (~29 s per 10,000 SMILES vs 0.14 s XGBoost
inference). The cache is built once and is row-aligned:
`pool_features.npy` (N×2074 float32), `pool_fps.npy` (N×2048 uint8),
`pool_profiles.json`, `pool_index.csv`, `manifest.json`. Misalignment raises at
startup.

Current manifest: 10,000 candidates, 0 rejected, **10.45% in-domain**, median
max-Tanimoto 0.262, 50.8% alert-free, 83.0% zero Ro5 violations, built
2026-09-08 in 92.5 s.

### 4.7 Custom molecules

`POST /api/predict` → `screen_custom`: profiles up to 500 SMILES, applies the
domain check, featurizes on demand and returns predictions. Pays the RDKit cost
per request.

---

## 5. LLM backends, orchestration and evidence

### 5.1 Backends (`llm_backends.py`)

All transport is OpenAI chat-completions over `httpx`.

| `llm` value | Resolves to |
|---|---|
| `rules` | no backend; deterministic parser, no network |
| `nebius` | Nebius only, and only if a key is set (never falls back to local) |
| `llama` | local llama.cpp only |
| `auto` (default) | Nebius if keyed, then llama.cpp, then rules |

Defaults: `NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1`,
`NEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b`, `LLAMA_SERVER_URL=http://localhost:8080`,
`LLM_TIMEOUT_SECONDS=90`, `LLM_MAX_TOKENS=2000`.

- Nemotron 3 is a reasoning model; `reasoning_content` counts against `max_tokens`. Report generation uses 2,400, the grounding loop 2,400/1,200.
- Every failure falls through to the next backend, then to the caller's deterministic floor. The backend that answered is returned in every response.
- `status()` probes llama.cpp `/health` and sends a real 1-token Nebius completion (cached 60 s) — reachability is tested, not assumed.

### 5.2 Bookend A — parsing (`orchestrator.parse_query`)

LLM path: system prompt with the tool schema → JSON → `_extract_json`
(brace-matching that tolerates fences and prose) → `_params_from_dict` → `clamp`.
If no usable JSON, falls to `_rule_based_parse`.

`_rule_based_parse` is the correctness floor:

- Explicit numbers beat keywords (`25 nm`, `1 µM`, unit-converted).
- Keyword tiers: very potent → 10 nM; potent → 100 nM; weak/any → 10,000 nM.
- Safety words raise `weight_safety` to 0.45; **dismissal phrases** ("don't care about", "ignore", "regardless of" …) lower it to 0.10 but still allow up to 2 alerts and still report them.
- Oral/Lipinski → drug-likeness weight ≥ 0.30. Novel → no domain requirement, confidence 0.10. Confident → in-domain required, confidence 0.35.
- `top N` / `N leads|hits|…` with a leading `\b` so "HER2 hits" is not "2 hits".

### 5.3 Bookend B — report (`orchestrator.generate_report`)

The LLM receives a compact JSON subset (target, funnel, model quality, parameters,
selected lead fields) and an explicit statement that the interval is 90% (a model
once reported it as 95%). Rules: never invent numbers, say these are QSAR
predictions, flag wide intervals and out-of-domain leads, recommend a next step,
3–5 sentences. If the output is ≤ 80 chars or the call fails, the deterministic
template is used, which carries the same caveats. `report_source` is one of
`nebius`, `llama.cpp`, `<backend>+tavily`, `deterministic`.

### 5.4 Grounding agent (`evidence.py`)

Runs strictly after the science is frozen, and only when `ground=true`.

| Tool | Corpus | Domains (mode `restrict`) |
|---|---|---|
| `search_literature` | literature | PubMed, PMC, EBI, Nature, Science, Cell, ACS, RSC, ScienceDirect, Wiley, Springer, bioRxiv, ChemRxiv, DrugBank, ClinicalTrials.gov, FDA, EMA, WHO |
| `search_patents` | patents | Google Patents, patent images, Espacenet, USPTO |
| `search_clinical_news` | clinical | FierceBiotech, Endpoints, STAT, BioPharma Dive, Reuters, Nature, FDA, EMA, ClinicalTrials.gov (`topic=news`, `time_range` month/year) |
| `read_sources` | — | `extract` up to 3 URLs, basic depth; snippet replaced by body, `read_in_full=True` |
| `crawl_source` | crawl | one URL + plain-language instructions, depth 1, ≤ 12 pages |

Search uses `search_depth=advanced`, `include_answer=false`,
`chunks_per_source=3`, `include_usage=true`. Domain restriction (not "prefer") is
deliberate: a result from the wrong corpus answers a different question. A test
asserts patent and literature domain lists do not overlap.

**Agent loop:** up to `EVIDENCE_MAX_TOOL_CALLS + 2` chat turns. Input is a brief
of the top 3 leads (id, SMILES, predicted pIC50, in-domain). The system prompt
tells the agent it cannot change a number and must say what it could not
establish ("no patent found" ≠ "unclaimed"). When the call budget is exhausted it
is asked to summarise with no further tools.

**Cost and cache controls**

- `EVIDENCE_MAX_TOOL_CALLS` default 8 in code (`.env.example` sets 6).
- Billed credits are read from Tavily's `usage` field and returned in the bundle.
- Responses cached on disk by payload hash (SHA-256 prefix), TTL 30 days, so a demo still cites real sources after a key expires. Directory `backend/data/evidence_cache` (override `EVIDENCE_CACHE_DIR`).

**Bundle** (`EvidenceBundle`): `citations` (id `S1…`, title, url, snippet, score,
`read_in_full`, `corpus`), `trace` (every tool call with backend), `api_calls`,
`cache_hits`, `credits`, `corpora`, `available`, `note`.

### 5.5 Citation integrity

`strip_unresolved_markers` removes any `[S#]` the agent was never given.

- Matches bracket variants (`[ ] 【 】 〔 〕 ［ ］ 〚 〛`) and range separators including U+2010–U+2015 and U+2212, because a real run produced `[S5‑S8]` with a non-breaking hyphen that an ASCII pattern silently skipped.
- Ranges (`S5–S8`) are expanded to explicit ids and each is validated individually.
- Output is normalised to `[S1, S2]` ASCII; gaps and orphan punctuation are tidied.

### 5.6 Failure behaviour

No key, no network, no credits, no chat backend, or a Tavily outage → empty
bundle with an explanatory `note` and an ungrounded report. `gather_evidence`
never raises.

---

## 6. Backend API

Base: FastAPI app `BioOrchestrator` v1.0.0. CORS allows `localhost:5173`,
`127.0.0.1:5173`, `localhost:4173`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | service descriptor |
| GET | `/api/health` | `status` (`ok`/`loading`), `llm_backends`, pipeline stats |
| GET | `/api/model-card` | accuracy disclosure, limitations, intended use |
| GET | `/api/pool-stats` | predicted pIC50 mean, percentiles (5–99), max, counts above 5.0–7.5, manifest |
| POST | `/api/parse` | Bookend A only; returns parameters |
| POST | `/api/screen` | full run, single JSON response, includes `report`, `report_source`, optional `evidence` |
| GET | `/api/screen/stream` | SSE run; query params `query`, `llm`, `top_n`, `ground` |
| POST | `/api/predict` | score 1–500 arbitrary SMILES |

**`ScreenRequest`:** `query` (1–2,000 chars), `llm` (`auto|nebius|llama|rules`),
`top_n` (1–50, optional), `ground` (bool, default false).
**`CustomScreenRequest`:** `smiles` list, 1–500.

Errors: `503` while models load; `400` on `/api/predict` with no valid molecule.
Streaming errors arrive as an `error` SSE event.

**Result schema** (`/api/screen`): `target`, `parameters`, `funnel`
(`pool, potent, safe, leads, dropped{}`), `pareto` (`n, n_fronts, front_sizes`),
`leads[]`, `model` (scaffold/random R², RMSE, coverage, alpha, hERG availability
and AUC), `timings` (`stage1..3`), `total_seconds`, `query`.

---

## 7. Frontend

Vite + React 18 + TypeScript SPA in `frontend/`. Dev server on `:5173` proxies
`/api` to `127.0.0.1:8000`, so SSE needs no CORS preflight. Path alias `@` →
`src`. Build: `tsc --noEmit && vite build` (ES2022, sourcemaps).

Fonts via `@fontsource`: Fraunces (display), Instrument Sans, IBM Plex Mono.
Structures drawn client-side with `smiles-drawer`.

### Files

| File | Role |
|---|---|
| `App.tsx` | page composition, primer (pre-run explainer), notices, layout |
| `hooks/useScreening.ts` | `useScreening` (SSE state machine) and `useHealth` |
| `types/api.ts` | wire contract mirroring `pipeline.py` / `orchestrator.py` |
| `components/QueryConsole.tsx` | prompt input, preset chips, grounding toggle |
| `components/PhaseRail.tsx` | parsing → screening → reporting → done progress |
| `components/FunnelChart.tsx` | real attrition per stage, dropped-reason counts |
| `components/FunnelIllustration.tsx` | primer illustration |
| `components/LeadCard.tsx` | one lead: metrics, interval, desirability, flags |
| `components/StructureView.tsx` | 2D structure rendering |
| `components/ExecutiveReport.tsx` | report text, citation markers linking to sources |
| `components/EvidencePanel.tsx` | agent tool-call trace by corpus, citations, credits |
| `components/AppHeader.tsx` | brand, backend availability, theme |
| `components/primitives.tsx`, `icons.tsx` | shared UI atoms |
| `lib/format.ts`, `lib/motion.ts`, `lib/theme.tsx` | formatting, animation, light/dark theme |

### State machine

`ScreenPhase`: `idle → parsing → screening → reporting → done | error`.
`useScreening.run(query, ground)` opens an `EventSource` and maps events to
state. Nothing invents progress: a stage appears when the server emitted it.
One `error` listener distinguishes a server-sent error (has a payload) from a
dropped transport (no payload; message points at uvicorn on :8000).

### Typing discipline

Anything the backend can send as null (`herg_risk`, `name`) is typed `null`, not
optional, so the compiler forces the degraded case to render rather than print
"undefined". `random_r2` and others are optional because the empty-result path
omits them.

---

## 8. Testing, configuration and deployment

### 8.1 Tests

| Suite | Command | Scope |
|---|---|---|
| `backend/tests/test_evidence.py` | `cd backend && python -m pytest tests/ -q` | 32 offline tests, no network, no keys: empty-bundle behaviour, marker stripping (range, bracket and dash variants), import isolation, budget and cache logic, backend resolution, domain separation, credit counting |
| `backend/scripts/verify.py` | `python backend/scripts/verify.py` | end-to-end assertions: featurization, split leakage (incl. degenerate cases), calibration within 0.03 of 0.90, unit conversion, chemistry engine, ranker behaviour, rule parser, funnel monotonicity, speed < 2 s, flags, report caveats |

Known gap in `verify.py`: it exercises the funnel at 5,000 nM, so it passes even
though the shipped default (100 nM) returns zero leads on the current pool.

### 8.2 Configuration (`.env`, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `NEBIUS_API_KEY` | — | enables hosted Nemotron |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` | |
| `NEBIUS_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | |
| `LLAMA_SERVER_URL` | `http://localhost:8080` | local fallback |
| `LLM_TIMEOUT_SECONDS` | 90 | per-call bound |
| `LLM_MAX_TOKENS` | 2000 | default budget |
| `TAVILY_API_KEY` | — | enables grounding |
| `TAVILY_BASE_URL` | `https://api.tavily.com` | |
| `EVIDENCE_MAX_TOOL_CALLS` | 8 (code) / 6 (.env.example) | hard stop per run |
| `EVIDENCE_CACHE_DIR` | `backend/data/evidence_cache` | |
| `EVIDENCE_CACHE_TTL` | 2,592,000 s | 30 days |
| `BIOORCH_SKIP_DOTENV` | — | skip `.env` loading |

`app/__init__.py` loads repo-root `.env` (then `backend/.env`) with
`override=False`, before any module-level `os.environ.get`, so real environment
secrets always win. `.env` is gitignored.

### 8.3 Dependencies (pinned, `backend/requirements.txt`)

rdkit 2026.3.6, xgboost 3.4.1, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5,
scipy 1.18.1, fastapi 0.141.1, uvicorn[standard] 0.52.4, pydantic 2.13.5,
httpx 0.28.1, python-dotenv 1.2.3. No vendor SDK is used for Nebius or Tavily.

### 8.4 Build and run

```bash
pip install -r backend/requirements.txt
python backend/scripts/train_surrogate.py        # ~70 s
python backend/scripts/fetch_chembl.py all       # hERG + pool (keyless)
python backend/scripts/train_herg.py             # ~60 s
python backend/scripts/build_pool_cache.py       # ~90 s
python backend/scripts/verify.py

cd backend  && python -m uvicorn app.server:app --port 8000
cd frontend && npm install && npm run dev        # http://localhost:5173
```

`setup.ps1` (Windows) runs all of it idempotently, skipping steps whose
artifact exists; `-Force` rebuilds. ChEMBL downloads retry 5× with 60 s waits.
`fetch_chembl.py` retries 4× with exponential backoff and checkpoints per target.

Timings: featurize 10,000 SMILES 29.4 s; XGBoost inference on 10,000 0.14 s;
cached full screen ~30 ms; model load 3.4 s.

### 8.5 Version control

`backend/models/`, `backend/data/`, `*.npy`, logs and `node_modules` are
gitignored; everything is regenerable from the quickstart. Branch `master`;
main branch for PRs is `main`.

---

## 9. Known limitations, risks and next steps

### Scientific limitations (also served at `/api/model-card`)

- 2D QSAR only: no 3D structure, pose, or active-site knowledge.
- IC50 is not Kd; no Cheng-Prusoff correction.
- Structural alerts flag assay-interference substructures, not proven toxicity.
- Out-of-domain predictions are extrapolation; the flag is not optional context.
- No ADME, no kinase selectivity panel, no in-vivo endpoint; CYP is not modelled.
- hERG training data is publication-biased toward positives.
- ~25% of EGFR labels are censored (`>`) but were stored as exact values (the `standard_relation` field was dropped when the dataset was built).

### Open defects and risks

| Issue | Detail |
|---|---|
| Default query returns zero leads | `max_ic50_nm=100` vs a pool whose maximum predicted pIC50 is 6.48 (≈332 nM). Realistic queries need a looser cutoff. |
| Pool is mostly out of domain | 89.6% of the pool is below the 0.35 similarity floor; predictions compress into 4.10–6.48 against training labels of 2.19–11.52, so ranking leans on regression to the mean. |
| Pool is not target-focused | The kinase-focused pool is incomplete (ChEMBL outage); the live cache is still the diverse pool. Its 20-target list also included EGFR, so a rebuild should drop it. A rebuild changes every result. |
| Weak prospective evidence | 13 uncensored pool compounds with measured EGFR IC50: Spearman −0.14, RMSE 1.61, coverage 69% vs 90% claimed. n is too small to conclude, but it is the only prospective check. |
| Global conformal scalar | Intervals too wide for familiar chemistry, too narrow for novel chemistry. |
| No salt stripping | `[Na+].[Cl-]` scores 4.07 instead of being rejected; counter-ions are fingerprinted. |
| Empty-input key mismatch | `predict_features` returns 5 keys on empty input vs 7 otherwise (`ic50_nm_lo/hi` missing). |
| Hosted path unproven | Nebius and Tavily code paths are implemented and unit-tested offline; no end-to-end hosted run is recorded in the repo. |
| Stale docs | `TECHNICAL_PLAN.md` and `HANDOFF.md` use CDK2 and old file names by design; `README.md` still says CDK2-era pool wording in places and lists 32 tests / 44 assertions that should be re-checked against current counts. |

### Suggested next steps, by expected value

1. Ship the XGBoost + Tanimoto-KRR blend (+0.024 R², measured; needs a predictor change and artifact, and the KRR variance can replace the global conformal scalar).
2. Complete the kinase pool (`fetch_chembl.py kinase_pool`, then `build_pool_cache.py`), excluding EGFR from the target list.
3. Add the known-drug panel as a blocking check in `verify.py` and test at the shipped default cutoff.
4. Re-fetch EGFR with `standard_relation` preserved, then train censoring-aware (`survival:aft`) and score on clean compounds only.
5. Strip salts / take the largest fragment before featurization.
6. Run and record an end-to-end Nebius + Tavily grounded screen.
