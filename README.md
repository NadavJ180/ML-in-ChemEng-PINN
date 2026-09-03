# Physical Hallucination Detection in PINN-Generated Flow Fields

> **An Undergraduate Research Project in Chemical Engineering**

## 📌 Project Objective
The goal of this project is to develop a compact and reproducible benchmark for detecting **physical hallucinations** in Physics-Informed Neural Network (PINN)-generated two-dimensional incompressible Navier-Stokes flow fields.

A "physical hallucination" is defined as a generated flow field that appears visually plausible but violates governing physical constraints, such as momentum conservation, incompressibility, periodic boundary consistency, or expected energy decay.

## 🔬 Central Research Question
*Can a normalized physics-based score detect visually plausible but physically inconsistent perturbations of PINN-generated 2D incompressible Navier-Stokes flow fields?*

## Reproducibility
All randomized case generations and network initializations are controlled via a global seed utility (`src.utils.seed.set_global_seed()`) to ensure deterministic behavior.

## Project Workflow & Training Execution

### Phase 3: Baseline PINN Training (Issue #7)
The current pipeline trains individual Physics-Informed Neural Networks for each randomized Taylor-Green Vortex (TGV) case. The training workflow employs a hybrid optimization strategy to ensure convergence and strict adherence to the project's usability criteria.

**Hardware & Execution Environment:**
* **Compute Node:** Local Execution (Alon's Machine)
* **OS:** Ubuntu 26.04
* **GPU:** NVIDIA GeForce RTX 3050 (6GB VRAM)
* **Optimization Strategy:** To accommodate the strict 6GB VRAM ceiling while processing Float64 higher-order PDE derivatives, the training script utilizes dynamic mini-batch tensor slicing via command-line arguments.

**Training Pipeline:**
1. **Adam Pre-training:** Guides the network out of local minima using dynamic spatial/temporal mini-batches.
2. **L-BFGS Fine-tuning:** A full-batch quasi-Newton optimization phase to drive physics residuals below the required $\mathcal{O}(10^{-4})$ thresholds.
3. **Automated Evaluation:** Models are strictly evaluated against the WP3 usability criteria (Relative L2 error, Continuity MSE, Momentum MSE). Failing models trigger the Section 12.4 risk mitigation protocol.
This project does not aim to create a new PINN architecture or a faster CFD solver. Instead, it provides a small, interpretable **detection framework** for identifying physically inconsistent generated flow fields using a Physical Hallucination Score (PHS).

## 🗂️ Repository Structure

```text
├── notebooks/
│   └── 01_tgv_visualization.ipynb   # Visualization of Taylor-Green Vortex
├── src/
│   ├── data/                        # Point sampling and dataset generation
│   │   ├── generate_datasets.py
│   │   ├── point_samplers.py
│   │   └── sampler.py
│   ├── models/                      # PINN architecture and loss functions
│   │   ├── loss.py
│   │   ├── pinn.py
│   │   ├── scaling.py               # ResidualScaler: non-dimensionalization, shared by training & scoring
│   │   ├── train_model.py           # Training entry point (writes plots/loss_history/{case_id}/, see below)
│   │   └── verify_model.py
│   ├── physics/                     # Physical equations and analytical models
│   │   ├── navier_stokes.py
│   │   └── taylor_green.py
│   ├── hallucinations/              # Issues #8-#9: perturbation engine + verification audit
│   │   ├── perturbations.py         # The 5 Section 7 perturbation types + apply_perturbation dispatcher
│   │   ├── generate_hallucinations.py  # Builds data/hallucinations/*.pt + hallucination_index.csv/json
│   │   └── verify_hallucinations.py    # WP4 audit: visual plausibility + violation-activation checks
│   ├── detection/                   # Issue #10: Physical Hallucination Score (Section 8, WP5)
│   │   ├── phs.py                   # Pure formula module: Smom/Sdiv/Sbc/S_bc_local/SE, normalization, scoring
│   │   ├── evaluate_phs.py          # Full detection pipeline: scores every field, calibrates tau, evaluates
│   │   ├── detection_sensitivity.py # Probes detection precision below the canonical epsilon floor
│   │   └── publication_figures.py   # Issues #12-13: curated Figures 1-3 / Tables 1-2 for the paper
│   └── utils/                       # Utility functions (e.g., seeding)
│       └── seed.py
├── tests/                           # Pytest suite for physics, samplers, losses, perturbations, and PHS
│   ├── test_pinn_loss.py
│   ├── test_point_samplers.py
│   ├── test_residuals.py
│   ├── test_perturbations.py
│   └── test_phs.py
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

**Generated (not committed as source, but tracked as deliverables where noted):**
```text
├── data/
│   ├── cases_metadata.json          # Regenerate with `python src/data/sampler.py` (seed=42, deterministic)
│   ├── hallucinations/              # Output of generate_hallucinations.py (Issue #8)
│   │   ├── {case_id}_hallucinations.pt
│   │   └── hallucination_index.csv / .json   # Dataset manifest: (case, perturbation, epsilon, split, label)
│   └── phs_scores/                  # Output of evaluate_phs.py (Issue #10)
│       └── phs_components_raw.csv / .json    # Every scored field's raw + normalized components + scores
├── plots/
│   ├── loss_history/                # Per-case training diagnostics (see train_model.py)
│   │   ├── {case_id}/loss_history.json / .png, rel_l2_tracking.png
│   │   └── _summary/loss_summary.csv / .json, all_cases_loss_overlay.png
│   ├── hallucination_verification/  # Output of verify_hallucinations.py (Issue #9)
│   │   └── {case_id}/contour_eps_0.01_0.02.png, residual_curves.png, residual_summary.csv/json, ...
│   ├── phs_evaluation/               # Output of evaluate_phs.py (Issue #10) and detection_sensitivity.py
│   │   ├── roc_curves.png, score_distributions_comparison.png
│   │   ├── raw_components_vs_epsilon.png, scores_vs_epsilon.png
│   │   ├── recall_by_type.png, recall_by_perturbation_type.csv
│   │   ├── normalizers_and_thresholds.json, detection_metrics_summary.csv / .json
│   │   └── sensitivity_recall_vs_epsilon.png, sensitivity_boundary_summary.csv / .json
│   └── paper_figures/                # Output of publication_figures.py (Issues #12-13), IEEE-sized
│       ├── figure1_valid_hallucinated_diff_residual.png
│       ├── figure2_phs_histogram_and_roc.png
│       ├── figure3_violation_signature_heatmap.png
│       └── table1_experimental_setup.csv / .tex, table2_detection_results.csv / .tex
```
`data/*` and `plots/*` are gitignored by default (see `.gitignore`); the specific files above are force-added (`git add -f`) when they're meant to ship as deliverables, following the convention already used for `models/*.pth` and the original `plots/loss_history/` outputs.

## ⚙️ Setup and Installation
This project is developed and tested on Linux (Ubuntu). To get started, clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone [https://github.com/nadavj180/ml-in-chemeng-pinn.git](https://github.com/nadavj180/ml-in-chemeng-pinn.git)
cd ml-in-chemeng-pinn

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Running Tests
To verify the physical constraints, residual computations, and point sampling logic, run the test suite using `pytest`:
```bash
pytest tests/
```

### Visualizing Analytical Solutions
Launch Jupyter to explore the Taylor-Green Vortex generator and verify the initial flow fields:
```bash
jupyter notebook notebooks/01_tgv_visualization.ipynb
```

### Generating the Hallucination Dataset (Issue #8)
Builds the perturbed-field dataset (5 perturbation types × 5 epsilon values + 1 clean baseline, per trained case):
```bash
python src/hallucinations/generate_hallucinations.py            # all cases
python src/hallucinations/generate_hallucinations.py --case_id case_00   # single case
```
Writes `data/hallucinations/{case_id}_hallucinations.pt` and the dataset manifest
`data/hallucinations/hallucination_index.csv` / `.json`. Note: running with `--case_id` **overwrites**
the manifest rather than merging into it — run once without `--case_id` to (re)index every case in one pass.

### Verifying Perturbations: Visual Plausibility & Violation Activation (Issue #9)
Audits the Issue #8 perturbations against WP4's acceptance criteria — small epsilons stay visually
imperceptible, and every perturbation type activates at least one physical violation:
```bash
python src/hallucinations/verify_hallucinations.py --case_id case_00
python src/hallucinations/verify_hallucinations.py --all_cases
```
Writes per-case contour comparisons, a full epsilon-sweep, a high-epsilon (ε=1.0) "obviously broken"
demo, and a quantitative residual/violation table to `plots/hallucination_verification/{case_id}/`.

### Evaluating the Physical Hallucination Score (Issue #10)
Computes 5 components (Smom, Sdiv, Sbc, S_bc_local, SE) for every field in the hallucination index,
calibrates normalizers/threshold from the validation split, and evaluates detection (ROC-AUC,
Precision, Recall, F1) on the held-out test split against 3 baselines (Score1/2/3 -- Score4 is PHS itself):
```bash
python src/detection/evaluate_phs.py                              # full run, all cases
python src/detection/evaluate_phs.py --n_interior 3000 --n_time 8 --energy_res 16   # fast smoke test
```
Writes the scored dataset to `data/phs_scores/`, and ROC curves, score distributions, per-type recall,
and epsilon-response plots to `plots/phs_evaluation/`. See `src/detection/phs.py`'s module docstring for
the exact formulas and every deliberate deviation from the write-up's literal notation (including the
5th component, `bc_local` -- promoted from an optional add-on to a permanent part of PHS; see the
Findings section below for why, and for an honest look at what it does and doesn't improve).

### Probing Detection Precision Below the Canonical Range
Loads an *existing* calibration from a prior `evaluate_phs.py` run (never refits it) and sweeps epsilon
values below the canonical floor, reporting recall vs. both epsilon and relative L2 error:
```bash
python src/detection/detection_sensitivity.py
```
Writes `sensitivity_recall_vs_epsilon.png` and `sensitivity_boundary_summary.csv`/`.json` (the epsilon/
relative-error values where recall crosses 50%/90%) to `plots/phs_evaluation/`.

### Generating Publication Figures & Tables (Issues #12-13)
Produces the exact curated deliverables WP6 calls for -- Figures 1-3 and Tables 1-2 -- sized for an IEEE
double-column paper, reading from `evaluate_phs.py`'s already-computed outputs wherever possible rather
than recomputing anything:
```bash
python src/detection/publication_figures.py
python src/detection/publication_figures.py --case_id case_05 --perturbation_type boundary --epsilon 0.002
```
Writes `figure1_valid_hallucinated_diff_residual.png`, `figure2_phs_histogram_and_roc.png`,
`figure3_violation_signature_heatmap.png`, and `table1_experimental_setup.csv`/`.tex`,
`table2_detection_results.csv`/`.tex` to `plots/paper_figures/`. Requires `evaluate_phs.py` to have
already been run (Figures 2-3 and Table 2 read its output directly) and a trained model for whichever
case Figure 1 illustrates (`case_00` by default). See the Findings section below for two rendering bugs
that were caught and fixed while building this (a malformed LaTeX subscript, and unescaped underscores
that would have broken real LaTeX compilation).

## 🎯 Key Deliverables & Roadmap
Based on the project blueprint, the following components are implemented or actively being developed:
- [x] **Taylor-Green Vortex Generator:** Analytical flow generation (`src/physics/taylor_green.py`).
- [x] **Residual Verification:** Navier-Stokes residual computation (`src/physics/navier_stokes.py` & tests).
- [x] **Baseline PINN Architecture:** Configurable neural network (`src/models/pinn.py`).
- [x] **Baseline PINN Training:** 30 trained TGV models (`src/models/train_model.py`, `models/*.pth`).
- [x] **Hallucinated Flow Fields (Issue #8):** 5 controlled perturbation types — spatial (velocity-divergence,
      momentum, pressure), temporal (`temporal_mismatch`, now shifting by a physically-scaled amount — see
      Findings below), and boundary — with a shared `apply_perturbation` dispatcher
      (`src/hallucinations/perturbations.py`, `generate_hallucinations.py`).
- [x] **Perturbation Verification (Issue #9):** Visual-plausibility and violation-activation audit against
      WP4's acceptance criteria (`src/hallucinations/verify_hallucinations.py`).
- [x] **Physical Hallucination Score (Issue #10):** Smom/Sdiv/Sbc/S_bc_local/SE components, validation-split
      normalization, and threshold calibration (`src/detection/phs.py`).
- [x] **Detection Metrics & Baseline Comparison (Issue #11):** ROC-AUC/Precision/Recall/F1 for PHS vs. 3
      baselines, plus a per-perturbation-type recall breakdown (`src/detection/evaluate_phs.py`). Note:
      WP5's own acceptance bar (`AUC(PHS) > 0.90`) is not currently met on the harder, boundary-centered
      epsilon range chosen for this project (`AUC(PHS) = 0.836`) — see the Findings section for why that's
      a deliberate tradeoff, not an oversight.
- [x] **Publication Figures & Tables (Issues #12-13):** Figures 1-3 and Tables 1-2, IEEE double-column
      sized (`src/detection/publication_figures.py`, output in `plots/paper_figures/`). One caveat: Figure
      1's residual panel should be regenerated against the actual retrained models before being treated as
      final — see the Findings section.
- [ ] **Technical Report & Archiving (Issue #14):** Not started.

## 📝 Documentation
Please refer to the `PINN suggestion-1.pdf` within the repository for the full academic roadmap, methodological details, and risk management strategies.

## 🔍 Findings & Known Limitations

Documented here as they were discovered, since they affect how the results in this README's plots
should be read and are directly relevant to the paper's Discussion/Limitations section.

### Phase mismatch in early training runs (fixed)
The models committed before commit `be9b8ff` (2026-07-17) were trained without accounting for the
randomly sampled phase (`phi_x`, `phi_y`) in each case's target field — they learned *a* valid TGV
solution, just not necessarily their own case's phase. This was caught by `verify_hallucinations.py`'s
`phase_amplitude_fit_diagnostic`, which fits the best-matching phase against the model's own output and
compares it to the case's true phase. Fixed by retraining with the corrected initial-condition target.
PDE-residual-based checks (everything in Issues #9 and #10) are phase-invariant and were unaffected by
this bug either way, but the earlier phase-mismatched models' Relative-L2-vs-analytical numbers were not
representative of true model quality and should not be reported as final.

### `Sbc` cannot detect the "boundary" perturbation type — by construction, not by bug
The "boundary" perturbation's envelope `m(x) = exp(-x²/σ²) + exp(-(2π-x)²/σ²)` is built from two
mirrored Gaussians straddling the periodic seam, so `m` and *every derivative of m* match exactly at
`x=0` vs `x=2π` (verified numerically up to 2nd order) — `x=0` and `x=2π` are the same physical point on
a periodic domain, and this envelope is smooth-periodic across it by construction. Consequently the
perturbed field remains **exactly** periodic, and `Sbc = MSE(s|x=0 − s|x=2π) + MSE(s|y=0 − s|y=2π)` —
a value comparison, per Section 8 — cannot distinguish it from a clean field at any order. No amount of
tweaking a boundary-*value*-comparison check can fix this; it requires a spatially-localized check
instead. `Smom`/`Sdiv` already detect this perturbation type fine (100% recall in practice, since the
envelope's narrow curvature spikes the momentum residual near the edges even though the value matches),
so `PHS` as a whole was never broken by this — but `Sbc`'s own name didn't match what it could actually
catch for one of the 5 perturbation types it's nominally responsible for. **`S_bc_local`**
(`compute_boundary_localization_violation` in `phs.py`) closes this directly: it compares near-boundary-band
residuals against the same interior collocation sample used for `Smom`/`Sdiv` rather than comparing exact
edge values, and does rise for this perturbation type (confirmed: raw value climbed from `2e-5` to `3.4e-3`
across the epsilon sweep, a ~170x increase, while `Sbc` stayed flat at `~2.9e-7`).

**Current configuration: `S_bc_local` is a permanent 5th component of the official PHS.**
`PHS_COMPONENT_NAMES = ["mom", "div", "bc", "bc_local", "E"]`, and `Score4_PHS_full` (all 5, summed after
independent normalization) is now what "PHS" means throughout this project — a deliberate departure from
Section 8's literal 4-term formula, made after `S_bc_local` was confirmed to close a real, structural gap
`Sbc` cannot close by construction (not a casual drift from the spec). `Score3_without_bc_local` (the
original 4-term formula) is kept as a standing ablation baseline specifically so the two can still be
compared directly.

**An honest result from that comparison, once `EPSILON_VALUES` was widened to the harder range described
below:** on the new, much harder epsilon sweep, `Score4_PHS_full` (AUC=0.850) does **not** clearly
outperform `Score3_without_bc_local` (AUC=0.855) — the two are statistically indistinguishable at this
sample size (5 test cases), and `Score4` even reads marginally *lower* in this particular run. This isn't
a contradiction of the "boundary" finding above: `bc_local` still visibly helps that ONE perturbation
type specifically (confirmed in `raw_components_vs_epsilon.png` — `S_bc_local` is the only component that
separates "boundary" from the pack at low epsilon). What this shows instead is that adding a 5th
independently-calibrated term — itself estimated from only 5 validation cases — introduces its own
calibration noise into the pooled sum, and at this harder epsilon range (where every score's AUC dropped
well below the old 1.000, so signal-to-noise matters more) that added noise roughly cancels out the
added signal in the *overall* pooled number, even though it's a clear net positive for the specific
perturbation type it targets. Reported here rather than smoothed over, since it's a genuinely useful
caveat for the paper: `bc_local`'s value is best understood per-perturbation-type, not from the pooled
AUC alone.

### `temporal_mismatch` was the hardest perturbation type to detect — fixed in place
Across a 14-case evaluation run, the original `temporal_mismatch` definition had **72% recall** vs.
**100%** for the other perturbation types (all misses at the 3 smallest epsilons; 100% recall by ε=0.1).
Root cause: this perturbation shifted time by `epsilon * T`, where `T = min(2.0, τ_decay)` and
`τ_decay = 1/(2νk²)` is the flow's natural decay timescale (`src/physics/taylor_green.py::compute_decay_timescale`).
Checked against all 30 cases' sampled `(Re, U0, k)`: **every single case** has `τ_decay > 2.0`, i.e. `T` is
capped, with `τ_decay / T` ranging from **1.67x to 114x** (median **8.8x**). So even the largest canonical
epsilon (0.1) only shifted time by `0.1 × 2.0 = 0.2s` of real time — a small fraction of how long these
flows actually take to evolve, for every case in the ensemble. This was a property of how `epsilon` was
defined for this perturbation type, not a scoring bug — `phs.py`'s normalization was verified by hand
against stored values and matches Section 8 exactly.

**Fix:** `perturb_temporal_mismatch` (`src/hallucinations/perturbations.py`) now shifts time by
`epsilon * τ_decay` — the flow's own *uncapped* natural timescale — instead of the capped `T`, so a given
epsilon represents roughly the same physical fraction of evolution for every case. The result is clamped
to `[0, T]` before querying the model: without this, `epsilon * τ_decay` could land far outside `[0, T]`
for slow-decaying cases (up to 114x `T`), asking the model about a time it was never trained on — pure
extrapolation, which tends to produce obviously-wrong output and would make the perturbation trivially
easy to detect (defeating the point of a *subtle* hallucination benchmark). For the slowest-decaying
cases, the largest epsilon values may saturate at exactly `t=T` (multiple epsilons producing the identical
shift) — this is an intentional physical ceiling ("as far as we can push this without leaving the domain
the model actually knows"), not a bug. This is a direct, in-place fix, not an added alternative —
`PERTURBATION_NAMES` still lists exactly 5 types. **Existing generated data/plots predating this change
are stale for `temporal_mismatch` specifically and should be regenerated**
(`generate_hallucinations.py` → `verify_hallucinations.py` / `evaluate_phs.py`).

**Result, on the same 14-case run:** recall for `temporal_mismatch` went from 72% to **100% at every
epsilon**, including the smallest (0.005). Detection overall improved from AUC=0.966 to a clean
**AUC=1.000** for both `Score2_momentum_divergence` and `Score4_PHS_full` on the test split — the single
weakest perturbation type had been capping overall performance, and fixing it directly lifted the whole
evaluation to perfect separation. (This AUC=1.000 was measured under the *old* `EPSILON_VALUES` range
described in the next section, before it was replaced with the harder, boundary-centered range —
included here for the historical record of what fixing `temporal_mismatch` itself achieved, holding
everything else constant. It is not the current headline number; see below for that.)

### Diagnostic tooling added: per-type recall breakdown and misclassification table
Two additions to `evaluate_phs.py` make cross-cutting detection issues (like the `temporal_mismatch`
finding above) visible without manually slicing the raw CSV: `evaluate_detection_by_perturbation_type` /
`recall_by_type.png` break recall down by `(perturbation_type, epsilon)` instead of pooling everything
into one number (uses a standard legend with `_styled_line`'s distinct linestyle/marker cycle, matching
every other multi-line plot in this module — an earlier version used per-type colored callout labels with
leader lines instead, specifically to guarantee zero overlap when several types tie at recall=1.0, but
was reverted in favor of a plain legend for consistency), and `diagnose_misclassifications` prints and
saves `plots/phs_evaluation/misclassified_fields.csv` — every field on the wrong side of `tau`, sorted by
how close the call was, with its margin.

**Current configuration: point sampling is seeded per-case by default.** `sample_interior_points` /
`sample_periodic_boundaries` calls inside `phs.py` use a deterministic seed derived from each case_id, so
every field belonging to one case (clean and all its perturbed variants) is scored at identical random
points — this is always on unless `evaluate_phs.py` is run with `--no_seed_points`. Added after measuring
~4% run-to-run swings in `Smom` from unseeded resampling alone — enough to flip a genuinely borderline case
between runs.

### A pre-existing, unrelated test bug: `tests/test_pinn_loss.py`
`test_pinn_architecture_and_loss_graph` calls `BaselinePINN()` with no arguments and fails with
`TypeError: missing 1 required positional argument: 'k'`. This predates all Issue #9/#10 work: the test
was added in commit `39f7bc0` (2026-07-12), and `k` became a required constructor argument two days later
in `d957ff3` (2026-07-14, adding Fourier feature input encoding) — that commit's own message says it
"updated the training file accordingly," but this test file was never updated to match, so it has been
silently broken since. One-line fix (pass `k=1`, matching the `LossEvaluator(... k=1.0)` already in the
same test) whenever someone gets to it — left alone here since it's unrelated to this session's work.

### The canonical epsilon range was rebuilt around the actual detection boundary
An earlier version of this project used `EPSILON_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]`, matching the
write-up's own Section 11 example values. That range gave a flat AUC=1.000 that didn't say where detection
actually became unreliable — it meant the smallest value (0.005) was already comfortably past the point
where detection is reliable, not that it sat at the edge of it. Two different notions of "subtle" were
being conflated: WP4's "visually imperceptible" and "hard for PHS to detect" are not the same threshold —
PDE residuals amplify small, especially high-frequency, perturbations far more than the human eye does, so
a change invisible in a contour plot can still be trivially separable in residual space.

Probed directly with `src/detection/detection_sensitivity.py` (loads an *existing* calibration, never
refits it, so it can't leak into the numbers it checks): pooled across all 5 perturbation types, recall
crossed **50% at ε≈0.0003 / relative L2 error ≈0.004%**, and **90% at ε≈0.0016 / relative L2 error
≈0.079%** — both well below the old smallest canonical value. Relative L2 error
(`compute_relative_error` in `phs.py`) matters here because `epsilon` itself isn't comparable across
perturbation types — a fraction of `U0` means something different from a fraction of a boundary-bump
amplitude or a fraction of a decay timescale — so it's reported as a second, physically comparable axis:
"how much did this change the field," in the same units regardless of mechanism. (That function pools
`(u, v, p)` together, not just `(u, v)` — the `pressure` perturbation only touches `p`, so a velocity-only
version would be identically zero for every one of its epsilons.)

**Decision made from this finding: `EPSILON_VALUES` was rebuilt around the boundary itself.** An
intermediate 10-value version, `[0.0001, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01]`,
was tried first, then trimmed to 5 — `[0.0001, 0.0005, 0.001, 0.005, 0.01]` — after checking directly
(by filtering the 10-value run's own already-computed data down to just these 5) that the resulting
recall curve still told the same story clearly: **0.40 → 0.52 → 0.76 → 1.00 → 1.00**, a genuine
floor-to-ceiling transition with fewer, cheaper points, at the cost of some resolution on exactly how the
climb happens between 0.001 and 0.005. This was a deliberate call that the write-up's Section 11 example
values are a starting point, not a constraint: smaller epsilon is *harder* to detect, not easier, so a
sweep weighted toward the boundary is a more demanding test of PHS, not a relaxed one, and finding that
boundary is closer to the actual point of this project than reproducing the write-up's specific numeric
examples. `VISUAL_CHECK_EPSILONS = [0.01, 0.02]` in `verify_hallucinations.py` (the required
visual-plausibility plot) is independently fixed and untouched by this change — it doesn't read from
`EPSILON_VALUES`.

**Consequence: the headline AUC is no longer 1.000, and that's the point.** On the 5-value range,
Score1=0.786, Score2=0.858, Score3=0.822, Score4/PHS=0.818 (test split, one representative run) — a real,
informative spread rather than every score maxing out together. `evaluate_phs.py`'s standard output
(`recall_by_type.png`, `scores_vs_epsilon.png`, etc.) now shows the actual sensitivity curve directly, as
part of the normal pipeline, rather than needing a separate probe to see it. `detection_sensitivity.py`
still exists for going even lower than the new floor (0.0001) — its own `SENSITIVITY_EPSILON_VALUES` now
starts at 0.00001 — or for checking specific values outside the standard grid without regenerating the
whole dataset.

**Direct consequence, and the answer to "why do the score distribution plots show so many false
negatives now": they're supposed to.** Checked exactly rather than assumed: on the test split, **54 of 250
hallucinated fields (22%) fall below tau** — and every single one of them is at ε≤0.002 (the bottom half
of the sweep); zero false negatives occur at ε≥0.003. They're also spread fairly evenly across all 5
perturbation types (7-15 each), not concentrated in one specific mechanism. This is the direct, intended
result of deliberately testing PHS against hallucinations below and around its actual detection boundary
(~ε=0.0003-0.0016) instead of only well above it — a real detection limit necessarily produces false
negatives when you test at or below that limit, by definition.

`score_distributions_comparison.png` was redesigned around exactly this: the original version pooled every
epsilon into one "hallucinated" histogram per score, which hid the structure above entirely — a wide,
spread-out mass with no visual indication that "easy" and "hard" fields were mixed together. It's now a
strip plot (every individual field's score plotted as a real point, not binned) with epsilon as categorical
x-axis positions — clean, then each epsilon value in increasing order — colored on a light-to-dark
gradient for increasing epsilon, with clean fields in a completely distinct color. This makes the
dose-response structure directly visible: you can see each epsilon's own cluster relative to tau, count how
many of its points sit below it (a visual version of `recall_by_type.png`'s numbers), and watch every
score's discriminative power narrow or widen across the 4 panels. A strip plot rather than a histogram,
box plot, or violin was a deliberate choice for this sample size (n=5 clean, n=25 hallucinated per
epsilon) — a violin plot's smoothed density can visually oversell how much data backs it this small, and a
histogram bins away exactly the individual-field detail that matters when n is this size.

One caveat worth keeping in mind when reading the recall curves: at the very smallest epsilon the
pooled recall floors around ~40% rather than 0%, which traces back to the k-imbalance false positives
below — 2 of the 5 test cases already sit above tau at their *clean* baseline, so a barely-perturbed
version of those two also reads as "detected," which is really the case's own baseline showing through,
not genuine sensitivity to that particular epsilon.

Also worth flagging for context regardless: AUC computed from only 5 clean test fields is a real result,
but a small one — it says "no clean field outranked any hallucinated field observed" (when AUC=1.0) or
reflects the ranking of a small pooled comparison otherwise, both narrower guarantees than the same
statistic computed from thousands of examples.

A related, more conceptual point for the paper's Discussion/Limitations: PHS was designed to check
momentum, divergence, boundary, and energy consistency, and Section 7's 5 perturbations were designed to
violate exactly those things — so strong detection here partly reflects that the attack and the detector
were built to match each other, not necessarily that PHS would catch an arbitrary physically-wrong field
generated by some other mechanism (e.g. a snapshot mislabeled from a different case, or an adversarially
optimized field that minimizes these specific residuals while still being wrong some other way). Not
something to fix now, but worth naming as a scope boundary rather than leaving it implicit.

### Publication figures & tables generated (Issues #12, #13), and a bug found along the way
`src/detection/publication_figures.py` (new) produces the exact deliverables WP6/Issues #12-13 ask
for — Figures 1-3 and Tables 1-2 — as a script separate from the diagnostic tools (`verify_hallucinations.py`,
`evaluate_phs.py`), since those are meant to be exhaustive (every type, every epsilon) while these are
meant to be a small, curated, print-ready set. Output lives in `plots/paper_figures/`:

- **Figure 1** (`figure1_valid_hallucinated_diff_residual.png`): valid field, hallucinated field,
  difference map, and — the one piece that didn't exist anywhere in the repo before — a residual
  **heatmap**, all for one representative example (`case_00`, `velocity_divergence`, ε=0.001 by default,
  configurable via `--case_id`/`--perturbation_type`/`--epsilon`). Residual statistics were already
  computed everywhere; this is the first time they're rendered as a spatial map. One honest caveat:
  the residual panel looks somewhat speckled/noisy in the current render — this is likely because the
  sandbox this was generated in is still working from an earlier, pre-retraining clone of the models
  (see the phase-mismatch entry above); it's worth regenerating against the actual retrained models to
  see if the panel cleans up, rather than assuming the speckle is a real physical signature.
- **Figure 2** (`figure2_phs_histogram_and_roc.png`): histogram of log10(PHS) and its ROC curve, both
  restricted to `Score4_PHS_full` specifically, matching the write-up's singular "the PHS" framing. The
  fuller 4-score ROC comparison remains available separately in `evaluate_phs.py`'s own `roc_curves.png`
  for the ablation discussion — this is the clean, single-score headline version for the paper, not a
  replacement for that.
- **Figure 3** (`figure3_violation_signature_heatmap.png`): perturbation type × PHS component heatmap
  (mean normalized `S̄_j` at the largest epsilon, test split) — genuinely new, and turned out to be a
  clean, single-glance summary of several findings from this project's whole history at once: `Sbc`'s
  column stays flat even for "boundary" (the blind spot), `bc_local`'s column responds broadly rather
  than being boundary-exclusive (it's a spatially-restricted momentum-residual check, not a
  boundary-only detector — any broadly-active perturbation like `pressure`/`momentum` shows up there
  too), and `SE`'s column lights up only for `temporal_mismatch`. One rendering bug fixed along the
  way: a raw underscore inside an already math-mode-subscripted label (`bc_local`) triggers a *second*,
  nested LaTeX subscript and renders as a stray vertical bar — fixed by substituting a comma for display.
  **A second display fix**: cells were originally annotated with the raw `S̄_j` value, spanning ~1 to
  ~800+ across the grid — genuinely correct (`S̄_j` is normalized *per component* against that
  component's own clean-field mean, not normalized *across components* to a common scale, so wildly
  different activation strengths between components is real information, not a normalization bug), but
  hard to read at a glance with single- and triple-digit numbers side by side. Cell text now shows
  `log10(S̄_j)` instead (colorbar relabeled to match); the color mapping and underlying data are
  unchanged, only the displayed digits are log-scaled.
- **Table 1** (`table1_experimental_setup.csv`/`.tex`) and **Table 2** (`table2_detection_results.csv`/`.tex`):
  computed directly from `cases_metadata.json` and `perturbations.py`'s own constants (Table 1) or read
  directly from `evaluate_phs.py`'s `detection_metrics_summary.csv` (Table 2), so neither can silently
  drift out of sync with the actual pipeline. A second bug fixed here: raw underscores in table values
  pulled from Python identifiers (e.g. `velocity_divergence`) will fail to compile or mis-render in
  actual LaTeX (underscore is a reserved math-mode character in text mode) — the `.tex` outputs now
  escape them (`\_`); the `.csv` outputs correctly keep plain underscores, since CSVs have no such
  reserved character.

**IEEE double-column sizing** is used throughout this new script (`IEEE_COL_WIDTH = 3.5in`,
`IEEE_PAGE_WIDTH = 7.16in`, explicit 8pt figure font, 300 DPI) rather than matplotlib's on-screen
defaults, which look oversized once a figure is scaled down to these physical print dimensions. This is
a starting convention, not a full pass over every figure in the repo — the diagnostic plots in
`verify_hallucinations.py`/`evaluate_phs.py` are still sized for on-screen inspection, which is
appropriate for their exhaustive/debugging role; only the 3 curated paper figures use it so far.

**One redundant plot removed** rather than left alongside a newer one, per the general
"replace, don't just add" principle: `phs_vs_epsilon.png` (Score4 only, all perturbation types overlaid
in one panel) was a strict subset of what `scores_vs_epsilon.png` already shows (every score including
Score4, per perturbation type, across separate panels) — removed, including its one unique feature (a
horizontal clean-baseline reference line), since `score_distributions_comparison.png`'s strip plot
already shows the actual clean-field points as their own category, which is more informative than a
single averaged reference line.

### `detection_sensitivity.py`'s own grid extended to reach a genuine 90% crossing, plus graceful fallback reporting
`SENSITIVITY_EPSILON_VALUES` originally topped out at 0.001, where recall was still climbing (~76-80%) --
so its "90% recall" boundary summary printed "not reached in this range," which was accurate but
unsatisfying on its own. Added `0.0015` and `0.002` (confirmed directly: `evaluate_phs.py`'s own
canonical-range run shows recall=0.92 at ε=0.002) so this script's own run now finds a genuine, self-contained
90% crossing at ε≈0.00186, rather than requiring a cross-reference to a different script's output. The
"6 values maximum" agreed for the canonical `EPSILON_VALUES` doesn't apply here -- that constraint was
specifically about the size of the full, expensive hallucination dataset every other script depends on;
this script is a lightweight, standalone probe against an already-existing calibration.

Also made the boundary-crossing report more informative when a level genuinely isn't reached (for
whatever future range gets chosen): rather than a bare "not reached in this range," it now also reports
the highest recall actually observed in-range and the epsilon it occurred at (e.g. "not reached in this
range (max recall = 0.80 at epsilon=0.001)"), computed once per group in `find_boundary_crossings` rather
than left as a dead end. One thing worth being clear about, since it came up while reviewing this
script's output: the reported "N epsilons" and the epsilon grid itself are computed dynamically from
`len(SENSITIVITY_EPSILON_VALUES)` and printed at the top of every run -- not a stale or hardcoded count.
The number changing between runs (7 before this fix, 9 after) reflects an intentional, separate grid for
this script, not inconsistent output.

### The canonical epsilon sweep now shows a complete rise-and-stabilization story in 6 points
`EPSILON_VALUES` was widened from 5 values to the agreed maximum of 6:
`[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]` — inserting `0.002` into the climbing region specifically
(the earlier 10-value exploratory run showed 0.001→0.76, 0.0015→0.84, 0.002→0.92, 0.003→1.00 recall, so
0.002 sits right where the curve is still visibly rising but close to the ceiling) rather than extending
either end further. The result, confirmed directly rather than assumed: pooled recall by epsilon is now
**0.40 (floor) → [0.0005, 0.001, 0.002 climbing] → 1.00, 1.00 (two points confirming stabilization, not
just one)** — a complete, small, six-point S-curve. Current AUCs on this range: Score1=0.800,
Score2=0.880, Score3=0.841, Score4/PHS=0.836 (test split) — the same Score2-leads-pooled-AUC pattern
documented above still holds at this slightly different range, for the same reasons.

### Score2 (momentum+divergence) currently beats Score3/Score4 in pooled AUC — a real trend, not a strong one
On the current epsilon range, pooled test-split AUC ranks Score2=0.880 > Score3=0.841 > Score4/PHS=0.836 >
Score1=0.800 — adding `bc`, `bc_local`, and `E` on top of momentum+divergence does not currently improve,
and mildly hurts, the pooled ranking metric. This was first found (and the mechanism below investigated)
on an earlier 5-value epsilon range, where the same ranking held with different absolute numbers
(Score2=0.858 > Score3=0.822 > Score4=0.818 > Score1=0.786) — cited here for the record, since the
bootstrap analysis below was run against that range and hasn't been repeated on the current 6-value one;
the qualitative pattern (and the point estimates above) both still hold at the new range, but treat the
specific confidence numbers as approximate rather than re-verified. Checked rather than assumed at the
time: a case-level bootstrap (2000 resamples of the 5 test cases) gives `P(Score2 > Score4) = 92.5%` and
`P(Score2 > Score3) = 92%` — a real, consistent direction, but well short of a strong statistical result
at this sample size (the 95% CIs overlap substantially). The same pattern held even broken down by
perturbation type, including "boundary," where `bc_local` was expected to show a clear win and didn't.

Likely mechanism: `Smom`/`Sdiv` respond robustly across all 5 perturbation types, while `Sbc`, `bc_local`,
and `SE` are each strongly informative for only some types and mostly contribute their own calibration
noise (each independently normalized from just 5 validation cases) for the rest. Every score here is a
naive equal-weighted sum, so adding a component that isn't informative for a given perturbation type
doesn't just fail to help that type's ranking — it adds variance on top of an already-good signal.

This does not mean `bc`/`bc_local`/`E` add no value in an absolute sense — `bc_local` rising ~170x for
the boundary perturbation specifically, and `SE` rising meaningfully for `temporal_mismatch` specifically,
are both real, independently-confirmed physical signals (see the sections above). It means a naive sum
doesn't currently translate that into a pooled-AUC improvement on this particular 5-case benchmark. Left
as an open, documented finding rather than acted on: the Score4-as-PHS decision from earlier stands (it
was made on structural grounds — closing Sbc's boundary blind spot — that this finding doesn't undo), and
no weighted-combination alternative has been attempted, since fitting weights would need labeled
hallucinated examples in calibration, a bigger shift away from the current "threshold from clean fields
only" design than seemed worth taking on here.

### One clean field can still score well above tau — traced to a real, fixable calibration gap
Even with `temporal_mismatch` fixed and 100% recall, one or two clean test fields still land above tau by
a wide margin (~1.0–1.4, not a borderline sliver). Investigated directly rather than guessed at: all
normalized components are elevated fairly uniformly for the affected fields (not one outlier check), and
tracing it against case parameters turned up a clean, non-coincidental cause — **wavenumber `k` correlates
with natural residual scale even for well-trained, clean fields** (mean *scaled* `Smom` across 14 sampled
cases: k=1 → 8.8e-6, k=2 → 6.8e-6, k=3 → 4.5e-6, monotonically decreasing), and the random 20/5/5 split
happened to allocate `k` unevenly: **validation is 20% k=1 (1 of 5 cases), test is 60% k=1 (3 of 5 cases)**.
The threshold was calibrated mostly from the "quieter" k=2/k=3 family and then applied to a test split
skewed toward the "naturally noisier" k=1 family — a real, mechanistic explanation, not calibration noise.

**Root mechanism, checked directly rather than assumed:** the *raw*, un-scaled residual runs the OPPOSITE
direction — it *increases* with k (higher spatial frequency is genuinely harder to fit to the same absolute
precision, consistent with the well-documented "spectral bias" of neural networks, see Rahaman et al. 2019
and the Fourier-feature literature this project's own `k`-parameterized input encoding is designed to
counteract). `ResidualScaler`'s non-dimensionalization divides by `scale_ns = U0² · k`, which grows with k
*faster* than the raw fitting error does here, so after scaling the direction flips: k=1 ends up looking
like the "noisiest" family, even though in absolute physical terms it's actually the easiest to fit. So
this isn't "k=1 systems are intrinsically dirtier" — it's that a scaling formula motivated by making
residuals *physically* dimensionless (the right goal for interpreting a residual's physical meaning) ends
up over-correcting for a *fitting-difficulty* effect it wasn't designed to address, and the resulting net
effect happens to point the "wrong" way for calibration purposes.

Options for generalizing past this (not implemented here — flagging as ideas, roughly in order of effort):
- **Stratify the train/validation/test split by `k`** (and ideally by `Re`/`U0` too) so each split gets
  proportional representation — the most direct fix for this specific imbalance, standard ML practice.
- **Covariate-aware normalization**: instead of one pooled `mean(Sj)` across all validation-clean fields
  regardless of case parameters, normalize each field against an expected baseline that accounts for its
  own `(Re, U0, k)` — e.g. a simple regression of "expected clean Sj" against case parameters — so the
  normalizer reflects that case's own natural scale rather than an average that can be skewed by which
  parameter values happened to land in the calibration set. Given the mechanism above, a **separate
  threshold per k** (as directly suggested during this investigation) is a reasonable, simple special case
  of this — one normalizer/tau per k value, rather than pooling k=1/2/3 into a single number — though with
  only 1-2 validation cases per k in the current split, each k-specific estimate would itself be a fragile,
  single-sample number until the split is also rebalanced.
- **Cross-validation across many random splits** rather than trusting one particular 20/5/5 partition —
  would directly reveal how much detection performance and false-positive rate depend on split luck, and
  give a more honest, averaged estimate of both.
- **Revisit `ResidualScaler`'s scaling exponent for this purpose specifically.** The current
  `scale_ns = U0² · k` is correct for the physics (non-dimensionalizing the Navier-Stokes momentum
  equation), but nothing requires PHS's calibration to use the *same* scaling a training loss uses — a
  scaling empirically fit to equalize *clean-field fitting error* across k (rather than one derived from
  the equations' own dimensional analysis) would address the root cause directly rather than working around
  it downstream via per-k thresholds.

**A broader hypothesis worth naming in the paper's Discussion, even if out of scope here:** is `k`
controlling "chaos"? Not quite, technically — this specific 2D TGV solution is exactly, analytically
solvable and non-chaotic in the dynamical-systems sense (no sensitive dependence on initial conditions) at
any `k` tested here, so "chaos" isn't the precise term. But the *spirit* of the hypothesis holds up well
under the mechanism found above: `k` is a physical parameter that genuinely controls something the network
finds harder or easier to represent exactly (spatial frequency / spectral complexity), and this shows up
measurably in fitting quality even for a well-trained model. The generalizable version of this idea is
probably the more interesting one for a Discussion section: **for any family of physical systems spanning
a parameter space, the "natural noise floor" of a well-converged solution is unlikely to be uniform across
that space** — some parameter regions may be intrinsically harder for a given architecture to fit to the
same precision, independent of training quality. A physics-informed anomaly/hallucination detector
calibrated by pooling across such a space (as PHS currently does) implicitly assumes that floor is roughly
constant; where it isn't, the calibration inherits whatever covariate imbalance happens to exist between
however the calibration and evaluation data were split. This is a general caution for any PINN-family
detector spanning a physical parameter space, not specific to Taylor-Green vortices or to `k`.

## 🚀 Future Research Directions

Having established a robust baseline pipeline and trained 30 independent Taylor-Green Vortex (TGV) models[cite: 3], the Physical Hallucination Score (PHS) framework opens up several exciting, high-impact avenues for subsequent academic research:

---

### 1. Adversarial Hallucination Generation (Automating the Fakes)
* **Objective:** Replace manual mathematical perturbations with automated, target-driven fakes[cite: 3].
* **Approach:** Implement an adversarial neural network (similar to a GAN generator) whose sole objective is to generate flow fields that successfully bypass the global PHS while minimizing visible spatial distortion[cite: 3]. This tests if the PHS framework can survive active adversarial "attacks" from an AI optimizing for physical evasion[cite: 3].

### 2. Spatially Localized Hallucination Mapping (Finding Where It Broke)
* **Objective:** Transition from a single, global diagnostic metric to a spatially resolved diagnostic tool[cite: 3].
* **Approach:** Expand the global scalar PHS into a pixel-wise Spatial Hallucination Heatmap[cite: 3]. By plotting the localized residual fields of the Navier-Stokes and divergence equations[cite: 3], the auditor can pinpoint exactly *where* in the physical domain (e.g., near boundaries or vortex centers) the model is hallucinating[cite: 3].

### 3. Robustness to Experimental and Sensor Noise
* **Objective:** Validate the detector's capability under real-world, imperfect engineering conditions[cite: 3].
* **Approach:** Inject varying levels of Gaussian white noise into the pristine validation and test datasets[cite: 3]. The research goal is to demonstrate that the PHS can mathematically distinguish between standard, expected experimental measurement noise and smooth, structurally incorrect deep-learning hallucinations[cite: 3].

### 4. Cross-PDE Methodological Generalization
* **Objective:** Prove that the normalization and thresholding framework is universally applicable to physics-informed models[cite: 3].
* **Approach:** Port the exact PHS methodology (calculating component-wise residuals, normalizing them using clean validation cases, and establishing a unified detection threshold tau)[cite: 3] to entirely different partial differential equations, such as the Heat Equation, Burgers' Equation, or 3D flow systems.