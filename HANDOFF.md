> ## ⚠️ SUPERSEDED IN PART — read this first (2026-09-09)
>
> **The dataset this document analyses is not CDK2. It is EGFR (CHEMBL203).**
> 100% of the 2765 training compounds have EGFR IC50 records and 90.4% of the
> labels match the EGFR median *exactly*; only 2.4% have any CDK2 record.
> Every "CDK2" reference below should be read as "EGFR".
>
> The **measurements in this document remain valid** — R², RMSE, the noise floor,
> the feature ablations, the censoring analysis. They were always measuring an
> EGFR model. Only the target attribution was wrong.
>
> One item is resolved by this: §3a's round-number censoring heuristic (22.6%)
> was detecting a dropped `standard_relation` field. ChEMBL marks 24.8% of these
> compounds censored. Next-step #4 ("re-fetch with relation preserved") is
> therefore confirmed as correct — but it must re-fetch the *intended* target.
>
> Full evidence and 6 further defects: `backend/experiments/audit/AUDIT.md`.

# Handoff — CDK2 surrogate optimization (2026-09-08)

Investigation into raising the XGBoost affinity surrogate. **No production files were
changed.** Everything below is measurement; `backend/app/` and `backend/models/` are
untouched. Experiment scripts are preserved in `backend/experiments/`.

---

## TL;DR

The surrogate is **label-limited, not capacity-limited**. Hyperparameter tuning is a
dead end (measured, not assumed). Feature fusion buys a real but small +0.011. The
genuine headroom is in **label quality (22.6% of labels are censored)** and in the fact
that **headline metrics overstate real screening utility**.

---

## Reproducible baseline

Repeated scaffold CV, 3 seeds × 5 folds. Production metadata reports a single seed
(0.7069); the multi-seed number is the honest one.

| | R² | RMSE | Spearman |
|---|---|---|---|
| Baseline (Morgan2 + desc26, d=2074) | **0.7025 ± 0.0052** | 0.8180 | 0.8424 |

**Seed spread is ±0.005 — treat anything under ~0.01 as noise.** This is the single most
important guardrail; most of the deltas being chased in this problem are smaller than it.

---

## 1. Feature engineering — largely saturated

Single fingerprint families, each + desc26 (scaffold CV R²):

| Morgan2 | Morgan3 | Morgan2-4096 | FCFP2 | FCFP3 | Torsion | AtomPair | RDKitFP | Avalon | MACCS |
|---|---|---|---|---|---|---|---|---|---|
| 0.7025 | 0.7009 | 0.7005 | 0.6944 | 0.7005 | 0.7017 | 0.6778 | 0.6870 | 0.6894 | 0.6187 |

No single fingerprint dominates. Morgan2 (the current choice) is already the best.

**The one clear find — the descriptor block is underpowered:**

- `desc26` alone (current curated set): R² = 0.478
- `desc_full` alone (216 RDKit descriptors, Ipc dropped): R² = **0.655**, from only 216
  dense features — nearly matching a 2048-bit fingerprint.

Fusion results:

| Feature set | d | R² | RMSE |
|---|---|---|---|
| Morgan2 + desc26 (baseline) | 2074 | 0.7025 | 0.8180 |
| Morgan2 + desc_full | 2264 | 0.7046 | 0.8151 |
| Morgan2 + FCFP2 + desc_full | 4312 | 0.7108 | 0.8065 |
| **Morgan2 + FCFP2 + Torsion + desc_full** | 6360 | **0.7136 ± 0.0049** | 0.8026 |
| + AtomPair | 8408 | 0.7110 | 0.8062 | 

Best is **+0.011 over baseline at ~3× featurization/train cost**. Adding atom-pairs on
top *hurts* — dimensionality is past its useful point.

> **Cost caveat before adopting:** `desc_full` takes ~7 ms/molecule (20 s for 2765). The
> plan claims "<1 s for 10,000 compounds"; 10k molecules is ~70 s of featurization. The
> `pool_cache` precomputes this, so the claim survives for the cached pool but **not** for
> arbitrary user-supplied SMILES. Check `backend/scripts/build_pool_cache.py` before
> committing to `desc_full`.

## 2. Hyperparameters — measured dead end

Single-seed scaffold CV on the d=6360 fused set. Full log: `backend/experiments/hp_fused.jsonl`.

```
0.7156  n_estimators=3000, lr=0.015
0.7154  n_estimators=1500
0.7150  {} (current hand-tuned production params)
0.7150  n_estimators=1500, max_depth=8
0.7146  n_estimators=1500, min_child_weight=1
0.7119  colsample_bytree=0.2
0.7108  colsample_bytree=0.8
0.7104  max_depth=4
```

Configs spanning depth 4–8, 600–3000 trees, colsample 0.2–0.8 span **0.7104–0.7156** —
entirely inside seed noise. The existing hand-tuned params are already at the plateau.
I stopped the random search rather than burn 40 more configs confirming a flat landscape.
**Do not restart this.**

---

## 3. EDA — where the real problem is

### 3a. 22.6% of labels are censored (the headline finding)

Round-number IC50 pile-ups at the top assay concentration, recorded in ChEMBL with an
`=` relation:

| IC50 | n | % |
|---|---|---|
| 100,000 nM | 214 | 7.7% |
| 10,000 nM | 130 | 4.7% |
| 20,000 nM | 91 | 3.3% |
| 50,000 nM | 46 | 1.7% |

Total flagged by the round-mantissa heuristic (≥1 µM, mantissa ∈ {1, 2, 2.5, 5}):
**624 compounds / 22.6%**. Mean pIC50 4.57 vs 6.45 for clean labels. These are "no
activity at top concentration" — the model is being asked to fit numbers nobody measured,
and they absorb **18.4% of total squared error**.

### 3b. Variance decomposition

- Between-scaffold: **79.5%** / within-scaffold: 20.5%
- 923 scaffolds, 69% singletons; largest series has **261 compounds** (one mega-series
  dominates whichever fold holds it — a source of fold-to-fold variance)
- A scaffold-mean oracle scores R² = **0.795**. We are at 0.71 under scaffold CV, so the
  practical ceiling is closer than the headline gap suggests.

### 3c. Headline metrics overstate real utility

| Spearman | value |
|---|---|
| All compounds | 0.8439 |
| Clean labels only | 0.8066 |
| **Among actives (pIC50 ≥ 6, n=1378)** | **0.6090** |

Much of the headline number is the easy task of separating censored inactives from
everything else. **Ranking within the interesting compounds — exactly what the funnel does
after its pIC50 ≥ 7 cut — is far weaker than 0.84 implies.** If one number should change
in the model card, it is this one.

### 3d. Error scales with distance from training chemistry

| max-Tanimoto to train | n | RMSE |
|---|---|---|
| 0.00–0.40 | 195 | 1.137 |
| 0.40–0.55 | 356 | 0.930 |
| 0.55–0.70 | 1060 | 0.838 |
| 0.70–0.85 | 951 | 0.692 |
| 0.85–1.01 | 203 | 0.557 |

RMSE varies **2×** across this range, but the conformal correction is a **single global
±0.333 scalar**. Intervals are too wide for familiar chemistry and too narrow for novel
chemistry — the regime the screener is actually for.

### 3e. Two things that are NOT problems (checked, ruled out)

- **No top-end shrinkage.** Stratifying residuals on the *true* value showed +0.933 bias
  at pIC50 ≥ 8, but that is a regression-to-the-mean artifact any imperfect model
  produces. Conditioned on *prediction* the model is unbiased everywhere (+0.008 at
  pred ≥ 8). I nearly chased this; don't.
- **Activity cliffs are not the bottleneck.** Only 4.0% of Tanimoto ≥ 0.7 pairs have
  |ΔpIC50| ≥ 2. Model RMSE (0.80, scaffold CV) already beats a leaky global 1-NN (0.89),
  so it is learning structure, not doing lookup.
- Screening itself is healthy: top-50 precision 100%, top-100 96%, enrichment 3.2–3.4×.

---

## Next steps, in priority order

1. **Censoring-aware training.** The experiment was written and queued but *not run* —
   this is the top open item. Compare, scored **only on clean test compounds** (a censored
   label is not ground truth): as-is / downweight (w=0.3) / clip to a floor / drop /
   **`objective="survival:aft"`** with left-censored interval labels
   (`label_lower_bound=0`, `label_upper_bound=y` for censored; `lb=ub=y` otherwise;
   `aft_loss_distribution="normal"`). AFT is the principled option — it says "true pIC50 is
   *at or below* 4.0" instead of forcing an exact fit. Built into XGBoost, rarely used in
   QSAR. Report `rho_actives` alongside R².
2. **Similarity-adaptive conformal intervals** (normalized or Mondrian CQR, conditioned on
   max-Tanimoto-to-train) to replace the global scalar — see 3d.
3. **Adopt the fused feature set** (+0.011) only if the featurization cost in the caveat
   above is acceptable. Bump `FEATURE_VERSION` — this forces retraining hERG and
   rebuilding the pool cache.
4. **Re-fetch CDK2 (CHEMBL301)** with IC50+Ki+Kd and `standard_relation` **preserved**.
   Gives more data *and* real censoring flags instead of the round-number heuristic.
   Biggest single lever, but it changes the input dataset — needs a decision.

## Open decisions (were awaiting the user)

- How invasive may this get? Items 1–3 change `featurize.py`/training, forcing a hERG
  retrain and pool-cache rebuild.
- Is re-fetching the dataset (item 4) in scope?

## Reproducing

```bash
python backend/experiments/build_blocks.py     # ~30 s, writes blocks/ next to the script
python backend/experiments/bench.py            # import and call run([...])
python backend/experiments/eda.py              # data structure analysis
python backend/experiments/censor.py           # censoring + error decomposition
```

Scripts resolve the project root relatively (`Path(__file__).resolve().parents[N]`) and
expect to write a `blocks/` directory beside themselves. Cached `.npy` blocks were in session-temp and are gone;
regenerate with `build_blocks.py` (~30 s).


---

# Addendum — 2026-09-09

## Model search is closed; blending is the win

Next-step framing in this document assumed better single models were available.
Measured, same harness, 3 seeds x 5-fold scaffold CV (noise floor +-0.005):

| model | R2 | RMSE | rho (actives) |
|---|---|---|---|
| XGBoost baseline (Morgan2+desc26) | 0.7025 +- 0.0052 | 0.8180 | 0.614 |
| Tanimoto-kernel ridge alone | 0.7078 +- 0.0056 | 0.8107 | 0.616 |
| XGBoost fused (d=6360) | 0.7124 +- 0.0043 | 0.8042 | 0.618 |
| **XGB + Tanimoto-KRR, 50/50** | **0.7263 +- 0.0040** | **0.7846** | **0.631** |

**+0.024 over baseline, ~5x the noise floor and ~2x the feature-fusion gain.**
The kernel model *ties* XGBoost alone — the gain is decorrelation, not a better
model. Residual correlation is 0.88 between XGB and KRR vs 0.96 between two
XGBoost variants; feature engineering was buying 0.96-correlated variants.
Blend weight is flat across 0.3-0.7, so 0.5 is a default, not a fitted point.

KRR is also the posterior mean of a GP, so its variance grows with Tanimoto
distance natively — that is **next-step #2 (similarity-adaptive intervals)** for
free. Reproduce: `backend/experiments/kernel_bench.py`, `ensemble.py`.

Untested, both need pip installs that could disturb the `requirements.txt` pins:
CheMeleon (97% win rate on MoleculeACE) and TabPFN v2 (86% vs XGBoost/CatBoost).

## Two defects that change how to read every number here

1. **The shipped default returns zero leads.** `ScreeningParams.max_ic50_nm`
   defaults to 100 nM; the pool's *maximum* predicted pIC50 is 6.48 (332 nM).
   `verify.py:159` tests the funnel at 5000 nM, 50x looser — which is why 49/49
   passes while the product is empty.
2. **The pool is 89.6% out of applicability domain** (median max-Tanimoto 0.262).
   Pool predictions compress into 4.10-6.48 against training labels spanning
   2.19-11.52. §3d's warning about a global +-0.333 conformal scalar is
   understated: on the pool it is not just miscalibrated, it is applied to
   predictions that are mostly regression to the mean.

## Validation gap

Everything in this document validates the system against itself. Nothing checked
it against external ground truth, which is why the target error survived. A
15-molecule known-drug panel surfaced it in minutes (gefitinib ranked first).
As an EGFR model the surrogate is good: Spearman 0.726 vs measured IC50, with
dacomitinib and icotinib predicted correctly as fully external compounds.
Suggested: put that panel in `verify.py` as a blocking check.
