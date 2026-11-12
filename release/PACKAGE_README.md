# PHS study, final iteration: the package

This is the final round of experiments asked for in October 2026, with everything needed to rebuild its results from the saved models, without retraining. Prepared by Nadav and Alon.

- **What is here:** the repository at the tag `final-v1` of https://github.com/NadavJ180/ML-in-ChemEng-PINN, exported with `git archive`: the code, the 30 old models, the 210 checkpoints of the 30 new cases, the case files with their seeds, the frozen calibration, the evaluation protocol and every result. The build script (`release/build_final_package.py`) added this file, `data/cases_metadata.json`, `pip_freeze.txt`, `reproduction_check.txt` and `repro_log.txt`.
- **Checked:** every result was rebuilt from scratch on a CPU and compared with the copies in this package. The outcome is in `reproduction_check.txt`, and the commit in its first lines.
- The repository's own `README.md` and `PIPELINE.md` document the code; `PROTOCOL.md` is the evaluation protocol, frozen before any new case was scored.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # we checked with Python 3.12; the exact versions are in pip_freeze.txt
python run_pipeline.py --preset final --device cpu
```

- It takes about 80 minutes on a CPU (79 on our 8-thread laptop, 58 of them for the scores). On an NVIDIA GPU, use `--device cuda`. The CPU run is the one we checked; a GPU run should agree to rounding, not bit for bit, which could matter only for a score within rounding of its τ (`PROTOCOL.md` section 4).
- **No model is trained.** The command rebuilds, from the saved models: the new cases' points, the damaged fields, their scores and the benchmark's tables (`results/benchmark/`), the true errors of all 240 models (`results/accuracy/`), the natural-error tables and figures (`results/natural_errors/`), and the final figures and tables (`results/figures/`). `python run_pipeline.py --preset final --list` prints each step and its exact command.
- It writes over the copies in `results/`. To compare yours with ours, unzip a second copy first.
- On another OS or with other library builds, the CSV files can differ in the last digits (about 1e-13). Every count, flag and LaTeX table should come out the same.
- The damaged fields (about 1.8 GB, `data/confirmation/hallucinations/`) are not in the ZIP; the command rebuilds them.
- Training runs on a GPU and is not repeatable bit for bit, so the package ships the trained checkpoints and every result is rebuilt from them. Retraining (`src/models/train_model.py`, see `README.md`) gives similar, not identical, models.

## What was asked for, and where it is

| Item | Where |
|---|---|
| All models and checkpoints | `models/case_00.pth` … `case_29.pth`: the 30 old models, whose clean fields the calibration was fitted on. `models/confirmation/case_30/` … `case_59/`: 7 checkpoints per new case (`adam_00100`, `adam_00500`, `adam_01000`, `adam_02500`, `adam_04000`, `adam_05000`, `lbfgs_final`), each a `.pth` file with a `.json` sidecar (case, seeds, stage, losses, errors on the evaluation grid, pass/fail, commit). Fingerprint of the folder's 455 files: `models/confirmation/MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256` in that folder; on macOS `shasum -a 256 -c`) |
| Metadata and seeds | `cases/confirmation_cases.json`: the 30 new cases, each with its 3 seeds. `data/cases_metadata.json`: the 30 old cases (seed 42). `results/figures/table_cases.csv`: one row per new case with its parameters, seeds, pass/fail and errors |
| Raw scores | `results/benchmark/confirmation_scores_raw.csv` (the 930 fields of the synthetic-damage benchmark) and `results/natural_errors/checkpoint_scores.csv` (the 210 checkpoints): raw and normalized components, the five scores and their flags. `calibration/source_phs_components_raw.csv`: the old pipeline's scores of the 930 fields of the 30 old cases, whose clean rows the calibration was fitted on (its normalized columns and Score1–3 use the old calibration) |
| Reference errors | `results/accuracy/reference_errors_confirmation.csv` (the 210 checkpoints) and `reference_errors_existing.csv` (the 30 old models) |
| Frozen calibration | `calibration/frozen_calibration.json`: the four normalizers and the five τ. The rules: `PROTOCOL.md` |
| Training logs | `models/confirmation/case_NN/loss_history.json` (every loss term at every Adam step and L-BFGS evaluation), `models/confirmation/loss_summary.csv` (final losses, pass/fail and minutes per case), `training_summary.log` (pass/fail), `RESTARTS.md`. The 30 old models: `models/loss_summary.csv` (the same columns; their runs saved only the final model) |
| Complete code | the whole package; `pip_freeze.txt` |
| Figures (PDF and PNG) and tables (CSV and LaTeX) | `results/figures/`: the paper's figures and tables. `results/benchmark/` and `results/natural_errors/`: every table of the two experiments |
| Every false alarm and every miss | Synthetic damage: `results/benchmark/false_alarms.csv`, `missed_detections.csv`, `clean_false_positives.csv`, `preflagged_cases.csv`. Natural errors: `results/natural_errors/false_alarms.csv`, `missed_failures.csv` (empty: there are none), `inaccurate_final_models.csv` (empty: there are none) |
| Instructions to rebuild everything | this file; `PIPELINE.md` |
| Completeness and reproducibility check | `reproduction_check.txt` and the console log of that run, `repro_log.txt` |

## How the study was run

1. **30 new cases**, `case_30` … `case_59`, 10 for each wavenumber k = 1, 2, 3, were drawn and registered before any training (tag `cases-v1`, 22 Oct 2026). Case n has three seeds: 3000 + n for its training and evaluation points, 4000 + n for its weights and mini-batches, and 1000 + n for its scoring points.
2. **Each case was trained once** and saved at 7 stages: after 100, 500, 1000, 2500, 4000 and 5000 Adam steps, and after L-BFGS (the final model, `lbfgs_final`). Tag `models-v1`, 30 Oct.
3. **The calibration was fitted once**, on the clean fields of the 30 old cases only: each component's normalizer is its mean over these 30 fields, and each score's alarm level τ is its 95th percentile over them. It was frozen with the protocol (tag `protocol-v1`, 31 Oct) before any new case was scored. A field is flagged when its score is above τ.
4. **Five scores**, each a sum of normalized components:

   | Score | Components | τ |
   |---|---|---|
   | `Score1_momentum_only` | mom | 1.586 |
   | `Score2_momentum_divergence` | mom + div | 2.888 |
   | `Score3_PHS_full` (the PHS) | mom + div + bc + E | 7.739 |
   | `Score4_PHS_no_bc` | mom + div + E | 6.853 |
   | `Score5_PHS_no_E` | mom + div + bc | 4.275 |

   mom and div are the mean squared residuals of the momentum and continuity equations of the model's own field; bc is the momentum residual near the domain edges divided by the one far from them; E is the mean squared gap between the field's kinetic energy and the exact Taylor–Green decay. E is the only term that uses the exact solution.
5. **Synthetic damage** (#35): the final model of each case, scored clean and with 5 damage types (velocity divergence, momentum, pressure, temporal mismatch, boundary) at 6 strengths ε (0.0001, 0.002, 0.01, 0.02, 0.03, 0.05): 30 clean and 900 damaged fields.
6. **Natural errors** (#39): the clean field of all 210 checkpoints, held against its true error.
7. **True error** (#37): the relative L2 error of the velocity against the exact solution, on a fresh 64 × 64 × 11 grid. A model is **inaccurate** above 10% (decided with Barak on 18 Oct, before scoring).
8. **Error bars**, 95%: for a metric over several fields per case (AUC, precision, F1, pooled recall), a case bootstrap: 2000 resamples of whole cases within each k, seed 0. For a count with one field per case (false alarms on clean fields, recall at one damage type and strength), the exact Clopper–Pearson range.

## Results

### Training: every run passed

- All 30 runs passed the acceptance test; none failed and none was restarted (`models/confirmation/RESTARTS.md`). So the extra "passed runs only" tables that the protocol calls for when a run fails are not needed.
- The final models' velocity errors are 0.15–0.73% (k = 1: 0.15–0.28%, k = 2: 0.26–0.54%, k = 3: 0.18–0.73%). The 30 old models: 0.14–0.56%; all 30 of their runs passed too (10.5 GPU hours, 18.6–22.7 minutes per case, `models/loss_summary.csv`).
- Training took 11.0 GPU hours in all, on an NVIDIA RTX 3050 in float64: 18–23 minutes per case, and 60 minutes for `case_32`, because the GPU ran slower during that run.
- The checkpoints: at the six Adam stages, 30, 30, 30, 22, 16 and 14 of the 30 are inaccurate, and no final model is. That gives 142 inaccurate and 68 accurate checkpoints: 38 Adam checkpoints with errors from 2.96% to 10%, and the 30 final models.

### Synthetic damage (#35)

| Score | AUC | Recall at τ | False alarms on the 30 clean fields |
|---|---|---|---|
| Score1: mom | 0.872 [0.868, 0.880] | 0.714 [0.706, 0.724] | 0 |
| Score2: mom + div | **0.900** [0.899, 0.906] | **0.803** [0.790, 0.822] | 2: case 34, case 45 |
| Score3: PHS | 0.884 [0.873, 0.899] | 0.770 [0.746, 0.801] | 3: case 46, case 48, case 53 |
| Score4: PHS − bc | 0.889 [0.878, 0.905] | 0.764 [0.739, 0.797] | 3: case 46, case 48, case 53 |
| Score5: PHS − E | 0.889 [0.887, 0.895] | 0.773 [0.759, 0.791] | 2: case 34, case 36 |

- Every score flags every damaged field from ε = 0.01 on, except Score1 for boundary damage (15/30 at ε = 0.01, 25/30 at 0.02). The pooled recall above mixes strengths: over the four strengths from 0.01 on it is 0.967–1.000, and over the two below it 0.210–0.410, depending on the score.
- At ε = 0.0001 (a velocity change of at most about 0.01%) almost every flagged field belongs to a case whose clean field the score already flags; there are only 3 new detections, all in cases whose clean score is just below τ.
- Detection falls with k: at ε = 0.002, the damaged fields flagged out of 50 for k = 1, 2, 3 are Score1 38, 22, 2; Score2 50, 43, 20; Score3 40, 27, 11; Score4 33, 28, 12; Score5 49, 25, 10.
- The false alarms, 0 to 3 of 30 per score, are in line with what the protocol expected from a τ fitted on 30 fields: between 2 and 3 of 31 on average.

### Natural errors (#39)

- Every score flags all 142 inaccurate checkpoints: no failure is missed (but see limit 3 below).
- The false alarms are the 38 accurate Adam checkpoints, which every score flags, and the false alarms on the final models (Score1 0, Score2 2, Score3 3, Score4 3, Score5 2), the same fields as the clean false alarms above.
- AUC, inaccurate against accurate checkpoints: Score1 0.960 [0.944, 0.976], Score2 0.959 [0.943, 0.975], Score3 0.993 [0.983, 0.999], Score4 0.993 [0.983, 0.999], Score5 0.959 [0.943, 0.975].
- No checkpoint has an error between 0.73% (the least accurate final model, `case_58`) and 2.96% (the most accurate Adam checkpoint).

### The bad results, and what the results do not show

1. **The PHS does not beat the simpler Score2.** Score2 has the highest AUC (0.900 against 0.884 for the PHS) and the highest recall, with fewer false alarms than the PHS. The difference in AUC, Score2 minus PHS, is +0.016 [+0.003, +0.030]. This range was computed after the fact, not planned in the protocol (paired case bootstrap on `results/benchmark/confirmation_scores_raw.csv`, 2000 resamples within each k, seed 0). Adding bc or E to Score2 lowers its AUC.
2. **False alarms and k.** Section 5 of `PROTOCOL.md` expected more false alarms for k = 1, since clean scores fall with k. That holds for Score1, Score2 and Score5: their clean scores fall with k (Spearman correlation with k: −0.80, −0.78, −0.90), and Score5's two false alarms are both k = 1. It does not hold for Score3 and Score4, whose false alarms are case 46 and case 48 (k = 2) and case 53 (k = 3). Their energy terms are large: S̄_E = 36.1, 20.4 and 9.14. S_E is not scaled by the velocity scale U0, so it grows like U0⁴ times the squared relative energy error. These three cases have U0 = 1.86, 1.66 and 1.91, higher than any old case of the same k (at most 1.56 for k = 2 and 1.62 for k = 3). Case 46 and case 48 also have the largest relative energy errors of all 60 models (0.84% and 0.79%, against at most 0.48% for the old ones). See `results/figures/fig_energy_term_vs_u0`. `PROTOCOL.md` is frozen, so this answer is given here.
3. **A fully trained model that is wrong was never tested.** All 142 inaccurate checkpoints are half-trained Adam checkpoints (errors from 10.5% up), and no final model is inaccurate. So "no failure is missed" holds for half-trained models only.
4. **Among the final models, the flags don't follow the error.** Score2 to Score5 flag final models with errors of 0.17–0.54%, while the least accurate one (`case_58`, 0.73%) is flagged by none. Over the 30 final models, the Spearman correlation of the score with the error is −0.15, −0.13, +0.34, +0.67 and −0.34 for Score1 to Score5 (computed after the fact from `results/natural_errors/checkpoint_scores.csv`). Only Score4 clearly follows it, through its energy term (limit 7). Within one run the scores do follow the error: over the 7 checkpoints of a case, the median Spearman correlation is 0.89–0.98. But the final models' errors span only a factor of 5 (0.15–0.73%), while the score per squared error varies 35 to 41-fold between them (Score1, Score2, Score5), and for Score1 is about 10 times higher for k = 1 than for k = 3.
5. **Comparing k at the same error mixes k with how far training has gone.** At the same Adam stage the k = 3 checkpoints are much less accurate than the k = 1 ones (median error after 5000 Adam steps: 3.8% for k = 1, 12.4% for k = 2, 18.0% for k = 3). So among checkpoints with the same error, the k = 1 ones are earlier in their training than the k = 3 ones.
6. **The bc term does not shrink with the error.** It is a ratio of the residual near the edges to the residual far from them. It makes up a median of 0.10 τ of Score3 and 0.18 τ of Score5 on the final models. On its own it ranks damaged fields below clean ones (AUC 0.420) and inaccurate checkpoints below accurate ones (0.120): the ratio grows during training, from a median of 0.60 after 100 Adam steps to 1.31 for the final models. See `results/figures/table_term_auc`.
7. **The energy term compares the model with the exact solution**, the same reference the true error uses. So Score3's and Score4's high AUC on the checkpoints (0.993; E alone 0.983) is partly built in.
8. **The 10% cutoff lies in an empty gap.** Every cutoff from 0.74% to 2.95% would split the checkpoints the same way, since no checkpoint falls in between. The 10% rule was fixed before scoring; any other cutoff would be chosen after the fact.
9. **A damage breaks the equations more than a real error of the same size.** As velocity changes, the damages are 0.04–0.2% at ε = 0.002, 0.2–1% at 0.01 and 1–5% at 0.05; the pressure damage changes no velocity (`results/figures/table_damage_size`). Among the damages of the same size as the final models' own errors (0.15–0.73%), the median rise of the score over its clean field is 0.79, 15, 6.1, 6.3 and 11 τ for Score1 to Score5. The final models' median scores are 0.62, 0.69, 0.49, 0.38 and 0.67 τ. For Score1 a damage and a real error of the same size count about the same; the scores with the continuity term react 12 to 22 times more to a damage, which breaks continuity, than to a trained model's own error. See `results/figures/fig_score_vs_velocity_error`.

### Reading the error bars

- **AUC ranges sit just above their estimate** (Score2: 0.900 [0.899, 0.906]), and for 3 subsets they lie wholly above it (Score4 for k = 1: 0.927 [0.928, 0.941]). A resample that draws a case twice adds the pairs of that case's clean field with its own damaged fields, which are nearly always in the right order (for Score4 at k = 1, 98.3% of them, against 92.0% of the pairs across cases), so resampled AUCs run high.
- **Some ranges have no width.** Score2's pooled recall for k = 3 is 0.733 [0.733, 0.733], because every k = 3 case has exactly 22 of its 30 damaged fields flagged, so no resample of cases changes it (also Score5 for k = 3: 0.700, and the natural-error recall, 1.000). For these, read the exact ranges per strength in `results/figures/table_recall_by_type_eps_k.csv` and `results/natural_errors/table_stage_by_k.csv`.
- With 10 cases per k the bootstrap ranges run a little narrow: in a simulation they held the true value about 92% of the time, not 95%. All ranges treat the frozen τ as fixed.

## How each figure and table is made

Each step is one script; `python run_pipeline.py --preset final --list` shows the commands. The tables are written as `.csv` and `.tex` (the `.tex` files need `booktabs`, some also `longtable`), the figures as `.pdf` and `.png` (the `fig_*.tex` captions need `graphicx`).

| Folder | Script | Built from |
|---|---|---|
| `results/accuracy/` | `src/models/accuracy_audit.py` | each model, against the exact solution on the 64 × 64 × 11 grid |
| `results/benchmark/` | `src/hallucinations/generate_hallucinations.py` (the damaged fields), `src/detection/evaluate_phs.py --calibration` (their scores), `src/detection/benchmark_metrics.py` (the tables) | the final models, the case file and the frozen calibration |
| `results/natural_errors/` | `src/detection/natural_errors.py` | the 210 checkpoints, scored with the same functions and settings; the true errors; the benchmark's clean rows, as a check (they must match) |
| `results/figures/` | `src/detection/final_figures.py` | the saved tables above, the case file, the training summary and the calibration (it loads no model) |

**`results/figures/`**, the paper's set. Each `fig_*.tex` holds the figure environment with its full caption, written from the numbers.
- `fig_roc`: ROC curves of the five scores on the 930 fields, with τ marked.
- `fig_recall_vs_eps`, `fig_recall_vs_eps_by_k`, `fig_recall_vs_eps_new_detections`: recall against damage strength, by damage type; by k; and over the cases whose clean field the score doesn't flag.
- `fig_clean_scores_vs_tau`: the clean scores of the new and old cases against τ, each false alarm labelled.
- `fig_energy_term_vs_u0`: the energy term against U0 (bad result 2).
- `fig_damage_fields`: what each damage does to a flow and to its residuals.
- `fig_natural_score_vs_error`: the score of each checkpoint against its true error.
- `fig_score_vs_velocity_error`: both experiments on one axis (bad result 9).
- `table_cases`: the 30 new cases. `table_recall_by_type_eps_k`: flagged counts by score, k, damage type and strength (a `longtable`: in a two-column paper, put it after `\onecolumn`). `table_synthetic_vs_natural`: both experiments at τ. `table_term_auc`: each term on its own. `table_damage_size`: the damages as velocity changes.

**`results/benchmark/`**: `confirmation_scores_raw` (every field); `metrics_overall`, `metrics_by_k`, `metrics_by_type`, `metrics_by_type_eps` (AUC, precision, recall, recall without the already-flagged cases, F1 and false-alarm rate, with ranges); `clean_false_positives`, `preflagged_cases`, `false_alarms`, `missed_detections`.

**`results/natural_errors/`**: `checkpoint_scores` (every checkpoint); `table_stage_by_k` (counts and rates by stage and k); `missed_failures`, `false_alarms`, `inaccurate_final_models`; `fig_score_vs_error` (each score against the error); `fig_fields_<case>_<stage>` (the fields of 5 chosen checkpoints).

## Rebuilding part of it

```bash
python run_pipeline.py --preset final --list                        # the 9 steps and their commands
python run_pipeline.py --preset final --stages sample,errors_old,errors_new   # the true errors only (under a minute)
python src/detection/natural_errors.py --from_scores                 # the natural-error tables and figures, from the saved scores
python run_pipeline.py --preset final --stages final_figures         # the final figures, from the saved tables (about 2 min)
```

The steps after `scores` read what the earlier ones wrote, so `--from metrics` needs a full run first (the 930 scores, `data/confirmation/phs_scores/`, are not in the ZIP).

## Notes

- `plots/paper_figures/` holds the first paper's figures and tables, from the 30 old cases. They are not part of this iteration and were not rechecked for it; `python run_pipeline.py --stages sample,hallucinate,phs,figures --device cpu` rebuilds them (checked byte for byte at commit `4437907`, October 2026).
- `calibration/source_phs_components_raw.csv` is that run's table of the 930 old fields, copied byte for byte; `PROTOCOL.md` (sections 4, 5 and 8) records how it and the calibration were checked, and their SHA-256.
- More detail: `PROTOCOL.md` (every setting), `README.md` and `PIPELINE.md` (the code), and issues #35 to #48 of the repository (the work, step by step; the roadmap is #47).
