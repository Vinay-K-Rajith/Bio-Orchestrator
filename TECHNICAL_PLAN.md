> ## ⚠️ SUPERSEDED — target attribution (2026-09-09)
>
> This plan specifies CDK2 throughout. The dataset it names is in fact
> **EGFR (CHEMBL203)** — 100% of the 2,765 compounds have EGFR IC50 records,
> only 2.4% have any CDK2 record. The code, models and docs have been
> relabelled to EGFR; this document is kept as written, as the record of
> original intent. Evidence: `backend/experiments/audit/AUDIT.md`.
>
> File names here are also stale: `cdk2_*.json` / `cdk2_chembl_labeled*.csv`
> are now `egfr_*`.

# BioOrchestrator: Technical Plan (`TECHNICAL_PLAN.md`)

A comprehensive technical blueprint for building **BioOrchestrator**—an autonomous, edge-native virtual compound screening harness that couples on-device LLM reasoning (`llama.cpp` + Nemotron 8B) with high-throughput surrogate ML (`XGBoost`), chemical safety guardrails (`RDKit`), and an interactive React frontend dashboard.

---

## 1. Executive Technical Summary

* **Objective:** Replace slow, multi-week physics simulations (molecular docking/FEP) with a sub-second, multi-stage machine learning funnel that screens 10,000+ candidate molecules against specific biological targets (human CDK2) for binding affinity and pharmacological safety.
* **Core Paradigm:** **Bookend LLM Orchestration**—a quantized 8B model running locally via `llama.cpp` interprets natural language intent, invokes Python ML tools via deterministic JSON schemas, and synthesizes structured candidate profiles with scientific caveats.
* **Privacy & Edge Feasibility:** Operates 100% on-device on a consumer laptop (zero cloud API fees, zero IP leakage).

---

## 2. System Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             REACT FRONTEND (Vite / Client)                       │
│  ┌───────────────────────────────┐     ┌──────────────────────────────────────┐  │
│  │   Natural Language Prompt     │     │     Live Screening Funnel (UI)       │  │
│  │   "Screen for safe CDK2 hits" │     │  10,000  ──►  400  ──►  45  ──►  5   │  │
│  └───────────────┬───────────────┘     └──────────────────▲───────────────────┘  │
└──────────────────┼────────────────────────────────────────┼──────────────────────┘
                   │                                        │
                   │ HTTP POST /api/screen                  │ JSON Stream / Response
                   │ { query, priority }                    │ (Top 5 Hits + Metrics)
                   ▼                                        │
┌───────────────────────────────────────────────────────────┼──────────────────────┐
│                       PYTHON BACKEND (FastAPI / Server)   │                      │
│                                                           │                      │
│  ┌────────────────────────────────────────────────────────┴───────────────────┐  │
│  │  1. API GATEWAY & PIPELINE CONTROLLER                                      │  │
│  │     Receives request, manages pipeline state, and dispatches tasks         │  │
│  └───────────────────┬───────────────────────────────────▲────────────────────┘  │
│                      │                                   │                        │
│                      │ (1) User Text                     │ (6) Synthesized Report │
│                      ▼                                   │     with Caveats       │
│  ┌───────────────────────────────────────────────────────┴────────────────────┐  │
│  │  LOCAL LLM ENGINE (Nemotron 8B via llama.cpp / HTTP Server)                │  │
│  │  • Bookend A (Input): Parses prompt ──► Strict JSON tool schema            │  │
│  │  • Bookend B (Output): Interprets final scores ──► Writes executive report │  │
│  └───────────────────┬────────────────────────────────────────────────────────┘  │
│                      │                                                           │
│                      │ (2) Structured Parameters {target, max_kd, safety_weight}  │
│                      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  2. DATA INGESTION & VECTOR FEATURIZATION                                  │  │
│  │     • Loads 10,000 candidate SMILES library                                │  │
│  │     • Encodes candidates into 2048-bit Morgan Fingerprints (RDKit)         │  │
│  └───────────────────┬────────────────────────────────────────────────────────┘  │
│                      │                                                           │
│                      │ (3) 10,000 Binary Feature Vectors                         │
│                      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  3. STAGE 1: FAST SURROGATE ML FILTER (Custom Model 1: XGBoost)            │  │
│  │     • Inference Speed: < 1 second for 10,000 compounds on CPU              │  │
│  │     • Predicts pIC50 / Kd (nM) against Human CDK2                          │  │
│  │     • Cutoff: Retains only potent binders (Kd < 100 nM / pIC50 >= 7.0)     │  │
│  │     ──► Outputs: ~400 Candidate Vectors (96% Reduction)                    │  │
│  └───────────────────┬────────────────────────────────────────────────────────┘  │
│                      │                                                           │
│                      │ (4) 400 Filtered Candidates                               │
│                      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  4. STAGE 2: MULTI-ATTRIBUTE THRESHOLDS & DOMAIN SAFETY                   │  │
│  │     • RDKit Lipinski Ro5: MW < 500, LogP < 5, HBD <= 5, HBA <= 10          │  │
│  │     • Toxicity & Safety: Flags hERG cardiac liability & CYP metabolism     │  │
│  │     • Out-of-Distribution (OOD) Guardrail:                                 │  │
│  │       Calculates max Tanimoto similarity against 2,765 ChEMBL training FPs.│  │
│  │       Flags items with similarity < 0.35 as "Unfamiliar / OOD Risk"        │  │
│  │     ──► Outputs: ~45 Safe & Drug-Like Hits                                 │  │
│  └───────────────────┬────────────────────────────────────────────────────────┘  │
│                      │                                                           │
│                      │ (5) 45 Multi-Metric Candidates                            │
│                      ▼                                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │  5. STAGE 3: MULTI-OBJECTIVE PARETO RANKER (Custom Model 2: MPO Engine)    │  │
│  │     • Evaluates trade-offs: Affinity Potency vs. Safety vs. Oral Delivery  │  │
│  │     • Applies desirability functions & isolates non-dominated Pareto front │  │
│  │     ──► Outputs: Top 5 Elite Lead Molecules                                │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Custom Model 1: Fast Affinity Surrogate (`XGBoost`)
* **Training Data:** `cdk2_chembl_labeled_dataset.csv` (2,765 unique labeled human CDK2 compounds from ChEMBL 37).
* **Input Representation:** 2048-bit Morgan Fingerprints (Radius 2) extracted via RDKit (`AllChem.GetMorganFingerprintAsBitVect`).
* **Target Label ($y$):** Experimental $pIC_{50}$ (logarithmic bioactivity, range: $3.0$ to $9.5$).
* **Model Architecture:** `XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8)`.
* **Performance Benchmark:** Evaluated on 80/20 stratified split ($R^2 \approx 0.78 - 0.82$, RMSE $< 0.55$).
* **Inference Latency:** $\sim 0.6$ seconds per 10,000 compounds on multi-core CPU.
* **Artifacts Generated:** `backend/models/cdk2_xgboost.json` and `backend/models/reference_fps.npy`.

### 3.2 Stage 2: RDKit Safety & Out-of-Distribution (OOD) Guardrails
* **Physicochemical Feasibility (Lipinski Rule of 5):**
  * Molecular Weight (MW) $< 500\text{ Da}$
  * Octanol-Water Partition Coefficient ($\text{LogP}$) $< 5.0$
  * Hydrogen Bond Donors ($\text{HBD}$) $\le 5$
  * Hydrogen Bond Acceptors ($\text{HBA}$) $\le 10$
* **Applicability Domain Assessment (Tanimoto Similarity):**
  * Computes pairwise Jaccard/Tanimoto similarity:
    $$T(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  * A candidate molecule is checked against all 2,765 reference fingerprints in the training set.
  * If $\max_{j} T(q, r_j) < 0.35$, the molecule is categorized as **Out-of-Distribution (OOD)** and flagged with a high prediction uncertainty warning.

### 3.3 Custom Model 2: Multi-Objective Pareto Ranker
* **Purpose:** Solves the multi-parameter optimization (MPO) problem across competing pharmacological goals.
* **Mathematical Formulation:**
  $$\text{Utility}(x) = w_1 \cdot D_{\text{affinity}}(x) + w_2 \cdot D_{\text{safety}}(x) + w_3 \cdot D_{\text{bioavail}}(x)$$
  Where $D_i(x) \in [0.0, 1.0]$ are continuous desirability functions:
  * $D_{\text{affinity}} = \min(1.0, \max(0.0, \frac{pIC_{50} - 5.0}{4.0}))$
  * $D_{\text{safety}} = 1.0 \text{ if in-domain and non-toxic else } 0.3$
  * $D_{\text{bioavail}} = 1.0 \text{ if 0 Lipinski violations, } 0.5 \text{ if 1 violation, } 0.0 \text{ if } \ge 2$
* **Dynamic Tuning:** Weights ($w_1, w_2, w_3$) are dynamically adjusted based on parameters extracted from the user's natural language prompt.

### 3.4 Orchestration Layer (`llama.cpp` + Nemotron 8B)
* **Inference Runtime:** `llama-server` running on `http://localhost:8080/v1` (with CPU/GPU offloading).
* **Deterministic Tool Calling:** Enforces strict GBNF / JSON grammar constraints:
  ```json
  {
    "target": "string",
    "max_kd_nm": "number",
    "require_lipinski": "boolean",
    "prioritize_safety": "boolean"
  }
  ```
* **Graceful Fallback:** If `llama-server` is offline, an internal rule-based regex parser seamlessly handles parameter extraction, ensuring zero demo failure.
* **Scientific Caveat Generator:** Translates numeric prediction envelopes into professional scientific caveats (e.g., *"Confidence high; ChEMBL kinase scaffold match; recommend FEP simulation for top 2 candidates"*).

---

## 4. Frontend Design & User Experience (React + Tailwind)

* **UI Theme:** Modern dark scientific dashboard (Slate `#0f172a`, Cyan `#06b6d4`, Emerald `#10b981`).
* **Core Views:**
  1. **Query Console:** Conversational input with preset prompt chips.
  2. **Screening Funnel Tracker:** Real-time visual progress counter displaying candidate attrition:
     $$\text{Input: } 10,000 \longrightarrow \text{Binders: } 400 \longrightarrow \text{Safe: } 45 \longrightarrow \text{Leads: } 5$$
  3. **Candidate Lead Cards:**
     * 2D SVG molecular structure preview.
     * Metric badges: Predicted $K_d$ (nM), $pIC_{50}$, MW, LogP, Desirability Score.
     * Security status: Green ("In-Domain Safe") or Amber ("Out-of-Domain Caution").
  4. **Executive Report Drawer:** Expandable natural language summary synthesized by Nemotron 8B.

---

## 5. Directory & File Layout

```
BioOrchestrator/
├── TECHNICAL_PLAN.md                 # This technical document
├── backend/
│   ├── models/
│   │   ├── cdk2_xgboost.json         # Serialized Model 1 weights
│   │   └── reference_fps.npy         # Serialized 2048-bit reference fingerprints
│   ├── data/
│   │   ├── cdk2_chembl_labeled.csv   # Training dataset (2,765 rows)
│   │   └── candidate_pool_10k.csv    # 10,000 unscreened candidate SMILES
│   ├── train_surrogate.py            # Model 1 training & evaluation script
│   ├── bio_engine.py                 # RDKit, Lipinski, and OOD similarity engine
│   ├── ranker.py                     # Custom Model 2: Multi-Objective MPO ranker
│   ├── orchestrator.py               # llama.cpp / Nemotron 8B client & fallback
│   ├── server.py                     # FastAPI REST API application
│   └── requirements.txt              # Python dependencies
└── frontend/
    ├── package.json                  # Node.js dependencies
    ├── vite.config.js                # Vite build config
    ├── tailwind.config.js            # Tailwind CSS styling
    ├── index.html                    # Single Page App shell
    └── src/
        ├── App.jsx                   # Main dashboard view
        ├── index.css                 # Global styling
        └── components/
            ├── QueryConsole.jsx      # Prompt input & chips
            ├── FunnelMetrics.jsx     # Attrition funnel visualization
            ├── CandidateCard.jsx     # Top lead molecular card
            └── ExecutiveReport.jsx   # LLM report & caveat drawer
```

---

## 6. Verification & Validation Protocol

| Milestone | Target Metric / Command | Success Criteria |
| :--- | :--- | :--- |
| **1. Model 1 Training** | `python backend/train_surrogate.py` | $R^2 \ge 0.75$, RMSE $< 0.60$, saves `.json` and `.npy` |
| **2. Pipeline Unit Test** | `python backend/bio_engine.py` | Accurate Lipinski and Tanimoto scoring on 100 test SMILES |
| **3. API Integration** | `uvicorn server:app --port 8000` | `POST /api/screen` returns valid JSON with 5 ranked leads in $< 2\text{s}$ |
| **4. Frontend Build** | `npm run build` in `frontend/` | Zero compilation errors, bundle ready for distribution |
| **5. End-to-End Walkthrough** | Live query test from React UI | Prompt triggers funnel animation $\rightarrow$ loads 5 cards with caveats |
