# Evaluation protocol of the final iteration (`protocol-v1`)

This file fixes how the 30 new cases are evaluated, before any of them is scored. The tag `protocol-v1` freezes it, together with the calibration in `calibration/`. Nothing here changes after the tag. If we find a real mistake, we fix it in a new PR, tag `protocol-v2`, and write down here what changed and why.

Two rules protect the results:
- **No new case is scored before the tag `protocol-v1` exists.** Otherwise someone could say we tuned our method on the test cases.
- **Bad results are reported as they come out.** No case, run or field is dropped, and no setting below is changed after scoring starts.

These tasks follow this protocol: #37 (true errors), #35 (main experiment), #39 (natural errors), #36 (figures) and #46 (ZIP). The plan is in the roadmap, #47.

## 1. Cases

- The 30 new cases are `case_30` … `case_59` in `cases/confirmation_cases.json`, frozen by the tag `cases-v1` (commit `ce38dca`). `case_30`–`case_39` have k = 1, `case_40`–`case_49` have k = 2 and `case_50`–`case_59` have k = 3.
- Their split is `confirmation`. They are used only for this evaluation, never to choose a setting.
- Case n has three seeds: `seed_points` = 3000 + n (its training and evaluation points), `seed_train` = 4000 + n (weights and mini-batches) and `seed_scoring` = 1000 + n (its scoring points, see section 4).
- No case is replaced, dropped or added.

## 2. Models, checkpoints and failed runs

Each case was trained once (#40) and saved at 7 stages. The 210 checkpoints are in `models/confirmation/case_NN/`, frozen by the tag `models-v1` (commit `1258d9d`). Their fingerprint is `models/confirmation/MANIFEST.sha256` (455 files; the SHA-256 of the fingerprint file itself is `5e2c1d215100c97d34fdbfe655facb93f116159aa1a1d83f84a514727141bc3a`).

| Stage | The model as it was |
|---|---|
| `adam_00100`, `adam_00500`, `adam_01000`, `adam_02500`, `adam_04000`, `adam_05000` | after that many Adam steps (one step is one mini-batch) |
| `lbfgs_final` | after L-BFGS: the final model |

A run **passed** if its `lbfgs_final` model meets the training acceptance test on the case's 64×64×20 evaluation grid: velocity relative L2 error below 0.10, MSE(R_c) below 1e-4 and MSE(R_m) below 1e-3. The checkpoint sidecars record this (`status`, `metrics`), and the result tables call it `passed_acceptance`.

Barak's rules (18 Oct 2026, recorded in #43):
- **Failed runs stay in.** They are marked as failed in every table. If any run fails, we add one table with only the runs that passed.
- A run that **crashes** (an error, a power cut) is restarted with the same seeds and written down in `models/confirmation/RESTARTS.md`. That doesn't count as replacing it.

When this protocol was frozen, all 30 runs had passed and none had been restarted (`models/confirmation/training_summary.log`, `models/confirmation/RESTARTS.md`). So the extra passed-only table is not needed.

## 3. What is scored

**Damaged fields (#35).** We take the `lbfgs_final` model of every case, failed runs included. Each case gets its clean field plus the same 5 damage types at the same 6 strengths as the old benchmark, from `src/hallucinations/perturbations.py` (unchanged):

| Damage type | What it does |
|---|---|
| `velocity_divergence` | adds ε·U0·sin(3x + 0.7)·sin(2y) to u, so the flow is no longer divergence-free |
| `momentum` | adds ripples of size ε·U0 that vary in space and time to u and v |
| `pressure` | adds ε·U0²·cos(5x)·cos(4y) to p |
| `temporal_mismatch` | reports the model's field at time t + ε·τ_decay (at most T) as the field at time t |
| `boundary` | adds a bump of size ε·U0 to u near the edges x = 0 and x = 2π |

The strengths are ε = 0.0001, 0.002, 0.01, 0.02, 0.03 and 0.05. That gives 31 fields per case (1 clean and 5 × 6 damaged), so 930 fields in all.

**Natural errors (#39).** The clean output of all 210 checkpoints, with no damage added.

## 4. Scoring settings

Every field is scored with the default settings of `src/detection/evaluate_phs.py`. #39 imports the same scoring functions rather than copying them. The code that computes the components has not changed since commit `4437907`, whose run produced the calibration's source scores (section 5). As a check, rescoring all 930 old fields at commit `f81010c` (`evaluate_phs.py --calibration`, on a CPU) gave every component of the source scores back exactly, bit for bit (31 Oct 2026).

| Setting | Value |
|---|---|
| Interior points for S_mom and S_div (`--n_interior`) | 20000 |
| Points for S_bc, split into near the edges and far from them (`--bc_local_n_points`) | 20000 |
| Width of the "near the edges" band (`--bc_local_band_width_fraction`) | 0.05 of the domain length 2π |
| Time slices for S_E (`--n_time`) | 20, from 0 to T |
| Grid per time slice for S_E (`--energy_res`) | 32 × 32 |
| Points per batch (`--chunk_size`) | 8000 |
| `--n_bc` | 1000 (the current S_bc doesn't use it) |
| Scoring points | Every field of case n, clean or damaged, is scored at the same random points, drawn with seed 1000 + n. `evaluate_phs.py` computes this seed from the case number; it is the case's `seed_scoring` in `cases/confirmation_cases.json`. `--no_seed_points` is never used. |
| Numbers | float64, on CPU or GPU. The two can differ in the last digits, which matters only for a score within rounding of τ. |

## 5. The five scores and the frozen calibration

Each field gets four raw components (`src/detection/phs.py`):
- S_mom = MSE(R_u) + MSE(R_v), the momentum residual;
- S_div = MSE(R_c), the divergence residual;
- S_bc, the momentum residual near the domain edges divided by the one far from them;
- S_E, the mean over time of the squared gap between the kinetic energy E(t) and its exact decay.

The residuals are scaled for each case (`ResidualScaler`) before they are squared. Each component is then divided by its frozen normalizer N_j: S̄_j = S_j / (N_j + 1e-12). A score is a sum of normalized components:

| Score | Components |
|---|---|
| `Score1_momentum_only` | S̄_mom |
| `Score2_momentum_divergence` | S̄_mom + S̄_div |
| `Score3_PHS_full` | S̄_mom + S̄_div + S̄_bc + S̄_E (this is the PHS) |
| `Score4_PHS_no_bc` | S̄_mom + S̄_div + S̄_E |
| `Score5_PHS_no_E` | S̄_mom + S̄_div + S̄_bc |

A field is **flagged** by a score when the score is above that score's alarm level τ (strictly: score > τ).

**The frozen calibration.** Barak: "Use only the 30 existing clean fields to calculate the component normalizers and each score's empirical 95th-percentile threshold. Freeze these before evaluating the new cases."
- The calibration is fitted once, on the clean fields of the 30 old cases `case_00`–`case_29` (all of them: 20 train, 5 validation and 5 test), and on nothing else. No damaged field and no new case is used.
- N_j is the mean of S_j over these 30 fields. τ is the 95th percentile of each score over the same 30 fields, with numpy's linear interpolation (the default of `np.percentile`).
- The source scores are `calibration/source_phs_components_raw.csv`, the scores of all 930 old fields from the old package. The file is copied byte for byte, not rescored. A rerun at commit `4437907` reproduced it byte for byte on 6 Oct 2026.
- The result is `calibration/frozen_calibration.json`, written by `python src/detection/frozen_calibration.py` at commit `f81010c`. Its values, rounded (the exact ones are in the file):

| | mom | div | bc | E |
|---|---|---|---|---|
| N_j | 7.541959e-06 | 5.150714e-07 | 1.691281 | 1.458200e-06 |

| Score | τ |
|---|---|
| `Score1_momentum_only` | 1.585837 |
| `Score2_momentum_divergence` | 2.888173 |
| `Score3_PHS_full` | 7.739237 |
| `Score4_PHS_no_bc` | 6.853411 |
| `Score5_PHS_no_E` | 4.275268 |

- New fields are scored with these numbers only, never refitted: by `evaluate_phs.py --calibration calibration/frozen_calibration.json` (#35) and by `frozen_calibration.apply()` (#39).
- What to expect from τ: with 30 fields, linear interpolation puts τ between the 3rd and 2nd largest of the 30 calibration scores, so each score flags exactly 2 of its own 30 fields (6.7%). If the new clean fields behave like the old ones, expect a false-alarm rate a little above 5%: on average between 2/31 and 3/31 (about 6.5% to 9.7%), because τ comes from only 30 fields.
- This differs from the old paper's calibration, which took N_j and τ from the 5 clean validation fields only, and had only Score1 to Score3.

## 6. True error (#37)

Each model is compared with the exact Taylor–Green solution (`generate_tgv` in `src/physics/taylor_green.py`) on a fresh, evenly spaced 64 × 64 × 11 grid:
- x and y take the 64 values 2πi/64 for i = 0, …, 63. The edge 2π is left out, since on the periodic domain it is the same point as 0;
- t takes 11 evenly spaced values from 0 to T, both ends included.

That is 45,056 points. They are neither training points nor the 64 × 64 × 20 `eval_grid` of the acceptance test.

Three errors, each a relative L2 error √(Σ (model − exact)² / Σ exact²), where for the velocity the u and v sums are added together:
- `relL2_uv`: the velocity over all points. **This is the main number.**
- `relL2_p`: the pressure, after removing its mean over x and y at each time, from both the model and the exact solution;
- `relL2_uv_T`: the velocity at t = T only.

A model is **inaccurate** if `relL2_uv` is above 0.10 (Barak, 18 Oct 2026).

This grid reproduces the old package's errors of the 30 old models (`04_checkpoints/checkpoints.csv`, 0.14–0.56%) to within 1e-13 (checked on 31 Oct 2026), so #37 can test its code against them.

## 7. Metrics and error bars

**Main experiment (#35).** For each score, overall, by k, by damage type, and by damage type × strength:
- ROC-AUC, and at τ: precision, recall and F1;
- the false-alarm rate: flagged clean fields out of 30 (out of 10 per k);
- **already-flagged cases:** if a score already flags a case's clean field, catching that case's damaged fields is not a new detection. So recall is reported twice: over all cases, and over only the cases whose clean field that score doesn't flag.

Every false alarm and every missed damaged field is listed.

**Natural errors (#39).** For each score, by stage × k: how many checkpoints are inaccurate, flagged, inaccurate but not flagged (missed), and accurate but flagged (false alarm). Every missed failure and every false alarm is listed, and any inaccurate `lbfgs_final` model is listed apart from the Adam stages.

**Error bars** (`src/detection/grouped_eval.py`, #41):
- `case_bootstrap`, for a metric over several fields per case (AUC, precision, F1, recall over all strengths): 2000 resamples. Each resample redraws whole cases with replacement, never single fields, separately within each k, so it always has 10 cases per k. Seed 0. The range is the 2.5th to 97.5th percentile of the resampled values. A resample on which the metric is undefined is left out, and the tables say how many were used (`n_boot`).
- `clopper_pearson`, the exact 95% range of a count, for counts where each case gives at most one field: the false alarms on clean fields, and recall at one damage type and strength.
- With only 10 cases per k, the bootstrap ranges run slightly narrow: in the #41 simulation they held the true value about 92% of the time, not 95%.
- The resamples come from numpy's random generator (`np.random.default_rng`), so the exact ranges depend on the numpy version. We use numpy 2.5.1 (`requirements.txt`).

## 8. What `protocol-v1` freezes

| File | SHA-256 |
|---|---|
| `calibration/source_phs_components_raw.csv` | `ee3dd7d1e2bfa571d9cb752cc92076640bd4c4d5686ba5e6816af57a089d15bd` |
| `calibration/frozen_calibration.json` | `24039706721d554022a42b49fb340538954147f46b2a75cc73efeb6409f359a1` |

Git stores both files byte for byte (`.gitattributes`), so these hashes are the same on Windows, macOS and Linux.

The scoring itself is frozen too: the components and scores in `src/detection/phs.py`, the damage in `src/hallucinations/perturbations.py`, the settings in section 4, and `src/detection/frozen_calibration.py`. After the tag, #35 adds only path flags to `evaluate_phs.py` and `generate_hallucinations.py`, so they can read the new cases and models.
