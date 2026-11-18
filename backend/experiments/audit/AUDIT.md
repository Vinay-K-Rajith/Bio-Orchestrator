# Pipeline audit — 2026-09-09

Independent end-to-end check of the screening pipeline. Scripts in this
directory reproduce every number. **No production file was modified.**

## Headline

`backend/data/cdk2_chembl_labeled.csv` is **not CDK2 data. It is EGFR
(CHEMBL203) data.** Every CDK2 claim in the README, model card, API and UI is
mislabelled. The machinery around it is sound; the target string is wrong.

## Evidence

Joined the training CSV against ChEMBL's own activity records:

| | CDK2 (CHEMBL301) | EGFR (CHEMBL203) |
|---|---|---|
| training compounds covered | 65 / 2765 (**2.4%**) | 2765 / 2765 (**100%**) |
| label matches target median exactly | — | **90.4%** |
| label within 2x of target median | 27.7% (of the 65) | **97.7%** |

Generously joining across *every* CDK2-related target (complexes, cyclin
pairings, orthologs) and *any* assay type still covers only 5.8%. 120 sampled
"missing" molecules all have ChEMBL IC50 records — overwhelmingly EGFR (199
records), then HER2/erbB-2, Src, VEGFR2.

Two independent corroborations:
- **Gefitinib** (CHEMBL939) carries IC50 28 nM in the file. It has *zero* CDK2
  records in ChEMBL; its EGFR median is exactly 28.0 nM.
- The EGFR records mark **24.8%** of the training set censored (`>`), matching
  HANDOFF's independent round-number estimate of 22.6%. The `standard_relation`
  was dropped, turning "no activity at top concentration" into exact labels.

## The model is good — at EGFR

Known-drug panel, using **out-of-fold** (scaffold-held-out) scores for anything
in training, so nothing is a memorised label:

| drug | measured EGFR | score | source |
|---|---|---|---|
| gefitinib | 22 nM | 7.62 | OOF |
| lapatinib | 27 nM | 7.61 | OOF |
| dacomitinib | 6 nM | **7.40** | **external** |
| icotinib | 5 nM | **7.13** | **external** |
| afatinib | 1 nM | 7.14 | OOF |
| imatinib (not EGFR) | 50 µM | 4.68 | OOF |
| aspirin | inactive | 3.53 | external |

Spearman 0.726 vs measured IC50, MAE 0.94 log. Osimertinib is the one miss
(predicted 5.33 vs true 7.91) and the model correctly flags it out-of-domain —
its aminopyrimidine chemotype is unlike the quinazoline training chemistry.

## Defects found

1. **Wrong target label** (above). Fix is a rename plus re-fetch, not a retrain.
2. **The default configuration returns nothing.** `ScreeningParams.max_ic50_nm`
   defaults to 100 nM; the pool's *maximum* predicted pIC50 is 6.48 (332 nM), so
   every realistic query yields 0 leads. `verify.py:159` tests the funnel at
   5000 nM — 50x looser — which is why 49/49 passes while the product is empty.
3. **Pool is 89.6% out of applicability domain** (median max-Tanimoto 0.262).
   Pool predictions compress into 4.10–6.48 against training labels spanning
   2.19–11.52 — the screen is ranking largely on regression-to-the-mean.
4. **Prospective accuracy is unverified and looks poor.** 18 pool compounds have
   measured EGFR IC50 (13 uncensored): Spearman −0.14, RMSE 1.61, conformal
   coverage 69% vs 90% claimed. **n=13 is too small to conclude from**, but it
   is the only genuinely prospective evidence available and it does not support
   the headline metrics.
5. `_deterministic_report` (orchestrator.py:349) hardcodes "docking against the
   **CDK2** ATP site" regardless of the target argument. `server.py` hardcodes
   CDK2 in 4 more places.
6. `predictor.predict_features` returns 5 keys on empty input but 7 otherwise
   (`ic50_nm_lo`/`ic50_nm_hi` missing) — latent KeyError.
7. No salt stripping: `[Na+].[Cl-]` scores 4.07 instead of being rejected; a
   hydrochloride salt is fingerprinted including its counterion.

## What is genuinely sound

Scaffold splitting (no leakage across 5 folds), conformal calibration
machinery, NSGA-II ranking (8 leads span 6 Murcko scaffolds, median pairwise
Tanimoto 0.32 — the crowding tiebreak works), the LLM/science separation,
determinism, malformed-input rejection, interval ordering on all 10,000 pool
rows, ~10 ms screens, and a live API. Pool/training overlap is 0.

## Novelty run

Screened for NSCLC (EGFR) leads at 1000 nM, in-domain only: 10,000 -> 48 -> 44
-> 8. Top hit **CHEMBL274064 (SB-220025)** at 405 nM. None of the 8 has any
measured EGFR activity in ChEMBL, so all are formally novel — but SB-220025 and
most of the series are **p38 MAP kinase** chemotypes (trisubstituted
4-fluorophenyl imidazoles). All 8 carry 3.1–4.1 log prediction intervals. Treat
as hypothesis generation only.
