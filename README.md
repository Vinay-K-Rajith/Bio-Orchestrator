# BioOrchestrator

**The check that belongs between a virtual screen and a decision to
synthesise.**

Synthesising one novel compound costs months of bench time and tens of
thousands of dollars, and it is the rate-limiting step in small-molecule
discovery. A screening model can tell you a molecule should bind and should not
be cardiotoxic. It cannot tell you the scaffold is already claimed in a patent,
or that the chemotype died in a trial in 2019 — because those are not
properties of a structure, they are facts about the world.

BioOrchestrator answers all four questions in one pass: a natural-language
request goes in, and a ranked, uncertainty-quantified shortlist comes out in
about 30 ms — each lead carrying a prior-art and liability check with
citations.

The science runs on your machine. The language layer runs on Nemotron via
Nebius Token Factory, locally via llama.cpp, or not at all — your choice, per
request. No key is required to run the pipeline end to end.

---

## The decision this serves

A medicinal chemist with 50 predicted hits and budget for 5 has to choose. That
choice is gated on four questions:

| | Question | Answered by |
|---|---|---|
| 1 | Will it bind? | affinity surrogate + conformal interval |
| 2 | Will it be toxic? | hERG model, PAINS/Brenk/NIH alerts |
| 3 | **Is it already known or claimed?** | **literature + patent search** |
| 4 | **Has this chemotype already failed?** | **clinical news + regulators** |

Most screening tools answer 1 and 2 and stop. Rows 3 and 4 are search problems,
not modelling problems — no amount of additional training data answers them,
because the answer changes when someone files a patent. That is the gap this
fills, and it is why the agent is load-bearing rather than decorative.

It also generalises in a way the model does not: the surrogate is trained on one
target, but the prior-art check works on any molecule you can name.

Two real runs, both unedited:

**It catches the model being wrong.** The surrogate ranked `CHEMBL274064` as a
sub-500 nM EGFR binder. The agent searched the literature and came back with:
SB-220025 is a **p38 MAPK inhibitor, not an EGFR binder** — a false positive
the model was confident about. No amount of additional training data produces
that correction, because it is not in the training distribution; it is in a
paper.

**It catches work already done.** Screen gefitinib and the agent returns
`WO2006090413A1` and reports the scaffold as already claimed. The numbers say
*make this*. The evidence says *someone already did, and patented it*.

In both cases the prediction is unchanged — the firewall holds. What changes is
whether you would act on it.

---

## What it actually does

```
natural language  ──►  screening parameters      (LLM or rule-based)
                            │
   10,000 compounds  ──►  Stage 1: affinity surrogate      XGBoost + conformal intervals
                            │
        binders     ──►  Stage 2: safety & domain         RDKit, PAINS/Brenk/NIH, hERG, Tanimoto AD
                            │
     safe & drug-like ─►  Stage 3: Pareto ranking         NSGA-II fronts + desirability weights
                            │
                        top N leads  ──►  written report with caveats
```

The language model sits at the two ends and never in the middle. It selects
parameters going in, and on the way out it runs an agentic loop over the
literature — deciding what to search, reading sources, and citing them in the
report. What it cannot do is compute, adjust, or filter a prediction.

That separation is deliberate and structural: the numbers are identical whether
Nemotron ran or not. It is also the entire safety argument for putting a
language model in a drug-screening loop, so it is enforced by dependency
direction and covered by tests, not by a prompt asking the model to behave.

---

## Quickstart

```bash
# 1. Install
pip install -r backend/requirements.txt

# 2. Train the affinity surrogate  (~70 s)
python backend/scripts/train_surrogate.py

# 3. Download hERG data and the screening pool from ChEMBL  (keyless, public)
python backend/scripts/fetch_chembl.py all

# 4. Train the hERG cardiotoxicity classifier  (~60 s)
python backend/scripts/train_herg.py

# 5. Precompute the pool  (~90 s, one time — this is what makes screening fast)
python backend/scripts/build_pool_cache.py

# 6. Verify everything
python backend/scripts/verify.py

# 7. Run
cd backend && python -m uvicorn app.server:app --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173
```

---

## Measured results

### Affinity surrogate (Model 1)

2,765 unique ChEMBL 37 compounds across 923 Bemis-Murcko scaffolds.

| Validation | R² | RMSE | Spearman |
|---|---|---|---|
| Random 5-fold | 0.785 | 0.695 | 0.889 |
| **Scaffold 5-fold** | **0.707** | **0.812** | **0.844** |

**Read the scaffold row.** ChEMBL kinase data is full of congeneric series —
about 3 analogs per scaffold here. A random split puts near-identical molecules
on both sides of the fold, so the model is graded partly on compounds it has
effectively already seen. The scaffold split holds out whole series, which is
the regime a screening tool actually operates in.

The 0.078 gap is the cost of that honesty. It is smaller than typical for
kinase QSAR, which is a genuinely good result for this dataset.

### Uncertainty

Conformalized quantile regression (Romano et al., 2019).

| | Coverage |
|---|---|
| Raw quantile regression | 0.760 |
| **After conformal correction** | **0.901** (target 0.90) |

Raw quantile regression undercovered by 14 points — it claimed 90% intervals
that contained the truth 76% of the time. The conformal correction
(±0.333 log units) restores calibration with a finite-sample guarantee. Without
this step every confidence number in the UI would be quietly wrong.

### hERG cardiotoxicity (Model 2)

10,929 compounds from ChEMBL target CHEMBL240, scaffold-split across 5,451 scaffolds.

| Metric | Value |
|---|---|
| ROC-AUC | 0.799 |
| AUPRC | 0.882 (base rate 0.666 → **1.32× lift**) |
| Best F1 | 0.832 at threshold 0.28 |

The base rate is 66.6% positive because ChEMBL hERG data is publication-biased —
labs measure hERG on compounds they already suspect. ROC-AUC flatters a model on
data like this, so the AUPRC lift over base rate is the number that says whether
the flag is informative. It is, modestly.

### Performance

| Operation | Time |
|---|---|
| Featurize 10,000 SMILES (RDKit) | 29.4 s |
| XGBoost inference on 10,000 | **0.14 s** |
| **Full screen (cached pool)** | **~30 ms** |
| Model load at startup | 3.4 s |

RDKit parsing, not the model, is the bottleneck — by roughly 200×. Sub-second
screening is only achievable by parsing the pool once, offline, into
`backend/models/pool_cache/`. Stages 1 and 2 then cost effectively zero, and
the ~30 ms is almost entirely Pareto sorting in Stage 3.

---

## Where this diverges from the original plan

Five changes, each made because the original was measurably wrong.

**1. Scaffold splitting replaced random splitting.**
The plan reported R² ≈ 0.78–0.82. That is the random-split number and it is
reproduced above (0.785) — but it overstates real-world accuracy. Both are now
reported, and every confidence signal derives from the scaffold number.

**2. "Predicted Kd (nM)" became predicted IC50 (nM).**
The training labels are IC50. Converting an IC50 to a Kd requires the
Cheng-Prusoff relation and the assay ATP concentration, which this dataset does
not contain. Reporting one as the other is a category error, so the pipeline
reports IC50 throughout.

**3. Conformal prediction intervals were added.**
The plan's only uncertainty signal was a binary Tanimoto flag. A screening
decision needs a magnitude, not a boolean: a pIC50 of 7.1 ± 0.2 and 7.1 ± 1.5
are completely different findings. Every prediction now carries a calibrated
90% interval alongside the Tanimoto applicability-domain check — two orthogonal
signals rather than one overloaded one.

**4. The hERG "flag" became a trained model.**
The plan listed hERG and CYP flags with nothing behind them. hERG is now a real
classifier trained on ChEMBL CHEMBL240. Structural alerts come from RDKit's
bundled PAINS/Brenk/NIH catalogs. CYP is not modelled at all, because there is
no honest way to fake it — its absence is stated rather than hidden.

**5. The candidate pool was rebuilt around kinase chemistry.**
This one was found by running the system. Screening 10,000 *random* drug-like
ChEMBL molecules against EGFR returns **zero** hits at the plan's 100 nM cutoff —
the highest prediction across the entire pool is pIC50 6.48, against a training
mean of 6.03.

That is not a model failure. ATP-competitive kinase inhibitors share hinge-binding
motifs that random chemistry does not have, and only 10.4% of a diverse pool
falls inside the applicability domain at all. Real virtual screening campaigns use
target-focused libraries for exactly this reason. The replacement pool is drawn
from compounds with measured activity against other kinases (20 targets, every
training-set compound excluded). This also makes the screen ask a more useful
question: which known kinase chemistry might cross-react with EGFR?

> **Caveat.** That target list was drawn up when the training set was believed to
> be CDK2, so it *includes* EGFR — the target the model is actually trained on.
> Per-compound exclusion means there is no literal train/pool overlap (audited: 0),
> but the pool is no longer the neutral library this section describes. Dropping
> EGFR from the list requires rebuilding the pool cache and changes every result.

**Current status.** `fetch_chembl.py kinase_pool` implements this, but ChEMBL
suffered a multi-hour partial outage during the build and the download did not
complete — it reached 6,634 of a target 10,000 compounds. The cached pool is
therefore still the diverse one, which is why the funnel numbers above use a
relaxed potency cutoff. To finish once ChEMBL is healthy:

```bash
python backend/scripts/fetch_chembl.py kinase_pool   # ~5 min when the API is up
python backend/scripts/build_pool_cache.py           # ~90 s
```

The diverse pool is preserved at `backend/data/candidate_pool_diverse.csv`.

Two hardening changes came out of that outage. ChEMBL can return activity records
with the structure field null rather than erroring — one target yielded 7 usable
compounds from 1,000 records while its neighbour yielded 2,100. The fetcher now
detects and reports that instead of silently producing a thin pool, and it
checkpoints after every target so a stalled run keeps what it already collected.

---

## Architecture

```
backend/
├── app/
│   ├── featurize.py      Morgan count FP + 26 RDKit descriptors (2074 features)
│   ├── splits.py         Bemis-Murcko scaffold splitting, leakage-free
│   ├── predictor.py      Affinity inference + conformal intervals
│   ├── bio_engine.py     Lipinski, Veber, QED, SA, alerts, applicability domain
│   ├── ranker.py         NSGA-II non-dominated sort + desirability functions
│   ├── orchestrator.py   LLM bookends with a deterministic fallback
│   ├── pipeline.py       The three-stage funnel
│   └── server.py         FastAPI + SSE
├── scripts/
│   ├── train_surrogate.py, train_herg.py
│   ├── fetch_chembl.py, build_pool_cache.py
│   └── verify.py         44 end-to-end assertions
└── models/               Trained artifacts + pool cache
```

**Feature versioning.** `featurize.py` stamps a `FEATURE_VERSION` into every
model artifact. Loading a model whose version does not match the running code
raises immediately rather than silently producing garbage from a misaligned
feature vector — the failure mode that is hardest to notice and most expensive
to discover later.

---

## LLM backend

Three backends, selected per request with the `llm` field. Which one actually
answered is returned in every response, because a screening tool that silently
changes where its molecules are sent is one nobody should trust.

| `llm` | Backend | Setup |
|---|---|---|
| `nebius` | `nemotron-3-super-120b-a12b` on Nebius Token Factory | set `NEBIUS_API_KEY` |
| `llama` | llama.cpp, local | `llama-server -m nemotron.gguf --port 8080` |
| `rules` | Deterministic parser | always available, no setup |
| `auto` *(default)* | Nebius if keyed, else local, else rules | — |

Copy `.env.example` to `.env` and fill in what you have. With nothing set, the
app still runs end to end on the rule-based parser.

Nemotron 3 is a reasoning model: it emits `reasoning_content` before `content`,
and both count against `max_tokens`. Budgets here are sized for that — a limit
tuned for a non-reasoning model gets spent entirely on reasoning and returns
empty content, which is indistinguishable from a backend failure.

The local path is not legacy. For unpublished chemistry, "no structure leaves
the machine" is sometimes a hard constraint, and `llm=rules` bypasses every
network path rather than merely preferring not to use one. The hosted path
exists because the tool-calling loop below needs a model larger than an 8B to
stay coherent.

The rule-based parser is the correctness floor, not a degraded mode. It handles
explicit numbers, negation ("I don't care about toxicity" *lowers* the safety
weight rather than raising it), and never invents a parameter. All LLM output is
clamped to validated ranges before the pipeline sees it, so a hallucinated
parameter cannot reach the science.

---

## Prior art and liabilities (Tavily)

Pass `ground=true` and Nemotron gets five tools over three corpora, and decides
for itself what to look up given the result in front of it.

| Tool | Corpus | The question it answers |
|---|---|---|
| `search_literature` | PubMed, PMC, ACS, RSC, Nature, ChEMBL, DrugBank | Is this chemistry already published? |
| `search_patents` | Google Patents, Espacenet, USPTO | Is it already **claimed**? |
| `search_clinical_news` | FierceBiotech, Endpoints, STAT, FDA, EMA | Has this chemotype recently **failed**? |
| `read_sources` | — | Read a source in full before asserting what it does or does not disclose |
| `crawl_source` | — | Follow one authoritative page's own links, steered in plain words |

**Why three corpora and not one search.** "Is it published", "is it claimed"
and "did it fail" are different questions living in different indexes. A
compound can be absent from every journal and still be covered by a patent, so
a literature search that comes back empty is not evidence of novelty. Domains
are applied with `include_domains_mode: "restrict"` rather than the default
`"prefer"`, because for a novelty claim a result from the wrong corpus is not a
weaker answer — it is an answer to a different question. A test asserts the
three domain lists stay disjoint.

**Why `crawl` and not just search.** Search finds pages matching a query; crawl
follows a site's structure. On a curated database entry, the neighbouring pages
are the related compounds and assays — precisely the ones you cannot name in a
query because you do not yet know they exist.

**Why not Tavily's `/research` endpoint.** It would run *its* agent instead of
this one, in a single opaque call. The tool calls here are the product: a reader
who cannot see which searches ran has no way to judge whether the citations are
relevant. Deliberate trade — less capability per call, a legible trace.

Three properties are enforced in code rather than left to the prompt:

**Evidence never touches a number.** Grounding runs after the science is
finished and frozen, and reaches only the report writer. `backend/app/evidence.py`
cannot import the surrogate, the filters or the ranker — and
`tests/test_evidence.py` fails if that ever becomes possible.

**Citation markers must resolve.** A marker naming a source the agent was never
given is stripped before the report is returned. A fabricated citation is worse
than none: it looks like provenance and survives being read aloud. Dropping it
leaves the sentence intact and unattributed, which is the honest state.

**Absence is survivable.** No key, no network, no credits or a Tavily outage all
yield an empty bundle and an ungrounded report. Grounding is an upgrade, never a
dependency.

Calls are capped per run by `EVIDENCE_MAX_TOOL_CALLS`, because an agent that can
search in a loop can also search forever, and credits actually billed are read
back from Tavily's `usage` field and shown in the UI rather than estimated.

Results are cached on disk by query. That is not only latency: this project is
judged weeks after submission, and a cached corpus means the demo still cites
real sources if the key has since expired.

The tool calls stream to the UI as their own SSE event, labelled by corpus, so
you can see which searches ran before reading the prose that cites them, and
each `[S1]` in the summary links to the source it names.

One detail worth naming, because it was found by testing rather than by
design: the model writes ranges, and it typesets them with Unicode dashes —
a real run produced `[S5‑S8]` with U+2011. An ASCII-only pattern declines to
match that, and **a marker that is not matched is a marker that is never
checked**. Ranges are now expanded to their individual ids and validated one at
a time, across six dash characters, with a test per dash.

```bash
cd backend && python -m pytest tests/ -q      # 32 tests, no network, no keys
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model status and available LLM backends |
| `GET /api/model-card` | Full accuracy disclosure and stated limitations |
| `GET /api/pool-stats` | Prediction distribution across the pool |
| `POST /api/parse` | Bookend A alone — inspect parsed parameters |
| `POST /api/screen` | Run the funnel, single JSON response |
| `GET /api/screen/stream` | Run the funnel, SSE per stage |
| `POST /api/predict` | Score an arbitrary SMILES list |

`/api/screen` and `/api/screen/stream` both accept `llm` (`auto`/`nebius`/`llama`/`rules`)
and `ground` (`true` to run literature grounding).

---

## Limitations

Stated plainly, and served at `/api/model-card` so the UI can show them next to
the predictions they qualify:

- These are QSAR predictions from 2D topology. There is no 3D structure, no
  binding pose, and no knowledge of the EGFR active site.
- IC50 is not Kd. No Cheng-Prusoff correction is applied.
- Structural alerts flag substructures associated with assay interference and
  known liabilities — they are not proof of toxicity.
- Out-of-domain molecules receive extrapolated predictions. The flag is not
  optional context.
- No ADME, no kinase selectivity panel, no in-vivo endpoint. CYP metabolism is
  not modelled.
- hERG training data is publication-biased toward positives.

Intended use is hypothesis generation and triage prior to docking or assay.
Not a substitute for experiment.
