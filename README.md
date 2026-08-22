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
│   │   ├── phs.py                   # Pure formula module: Smom/Sdiv/Sbc/SE, normalization, scoring
│   │   └── evaluate_phs.py          # Full detection pipeline: scores every field, calibrates tau, evaluates
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
│   └── phs_evaluation/              # Output of evaluate_phs.py (Issue #10)
│       ├── roc_curves.png, score_distributions.png
│       ├── phs_vs_epsilon.png, raw_components_vs_epsilon.png, scores_vs_epsilon.png
│       ├── recall_by_type.png, recall_by_perturbation_type.csv
│       ├── normalizers_and_thresholds.json, detection_metrics_summary.csv / .json
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
Computes the 4 Section 8 components (Smom, Sdiv, Sbc, SE) for every field in the hallucination index,
calibrates normalizers/threshold from the validation split, and evaluates detection (ROC-AUC,
Precision, Recall, F1) on the held-out test split against 2 residual-only baselines:
```bash
python src/detection/evaluate_phs.py                              # full run, all cases
python src/detection/evaluate_phs.py --n_interior 3000 --n_time 8 --energy_res 16   # fast smoke test
python src/detection/evaluate_phs.py --include_bc_local           # also score the optional S_bc_local (see below)
```
Writes the scored dataset to `data/phs_scores/`, and ROC curves, score distributions, per-type recall,
and epsilon-response plots to `plots/phs_evaluation/`. See `src/detection/phs.py`'s module docstring for
the exact formulas and every deliberate deviation from the write-up's literal notation.

## 🎯 Key Deliverables & Roadmap
Based on the project blueprint, the following components are implemented or actively being developed:
- [x] **Taylor-Green Vortex Generator:** Analytical flow generation (`src/physics/taylor_green.py`).
- [x] **Residual Verification:** Navier-Stokes residual computation (`src/physics/navier_stokes.py` & tests).
- [x] **Baseline PINN Architecture:** Configurable neural network (`src/models/pinn.py`).
- [x] **Baseline PINN Training:** 30 trained TGV models (`src/models/train_model.py`, `models/*.pth`).
- [x] **Hallucinated Flow Fields (Issue #8):** 5 controlled perturbation types — spatial (velocity-divergence,
      momentum, pressure), temporal, and boundary — with a shared `apply_perturbation` dispatcher
      (`src/hallucinations/perturbations.py`, `generate_hallucinations.py`).
- [x] **Perturbation Verification (Issue #9):** Visual-plausibility and violation-activation audit against
      WP4's acceptance criteria (`src/hallucinations/verify_hallucinations.py`).
- [x] **Physical Hallucination Score (Issue #10):** Smom/Sdiv/Sbc/SE components, validation-split
      normalization, and threshold calibration (`src/detection/phs.py`).
- [x] **Detection Metrics & Baseline Comparison:** ROC-AUC/Precision/Recall/F1 for PHS vs. 2 residual-only
      baselines, plus a per-perturbation-type recall breakdown (`src/detection/evaluate_phs.py`).

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
so `PHS` as a whole is unaffected. An optional 5th component, **`S_bc_local`**
(`compute_boundary_localization_violation` in `phs.py`), was added to close this gap directly: it
compares near-boundary-band residuals against the same interior collocation sample used for `Smom`/`Sdiv`
rather than comparing exact edge values, and does rise for this perturbation type (confirmed: raw value
climbed from `2e-5` to `3.4e-3` across the epsilon sweep, a ~170x increase, while `Sbc` stayed flat at
`~2.9e-7`). It is off by default (Section 8 defines exactly 4 components) — enable with
`evaluate_phs.py --include_bc_local` to also compute `Score4_PHS_plus_bc_local` for comparison.

### `temporal_mismatch` is the hardest perturbation type to detect
Across a 14-case evaluation run, `temporal_mismatch` had **72% recall** vs. **100%** for the other 4
perturbation types (all misses at the 3 smallest epsilons; 100% recall by ε=0.1). Root cause: this
perturbation shifts time by `epsilon * T`, where `T = min(2.0, τ_decay)` and `τ_decay = 1/(2νk²)` is the
flow's natural decay timescale (`src/physics/taylor_green.py::compute_T`). Checked against all 30 cases'
sampled `(Re, U0, k)`: **every single case** has `τ_decay > 2.0`, i.e. `T` is capped, with
`τ_decay / T` ranging from **1.67x to 114x** (median **8.8x**). So even the largest canonical epsilon
(0.1) only shifts time by `0.1 × 2.0 = 0.2s` of real time — a small fraction of how long these flows
actually take to evolve, for every case in the ensemble. This is a property of how `epsilon` is defined
for this perturbation type (`src/hallucinations/perturbations.py::perturb_temporal_mismatch`, an Issue #8
file) combined with the WP2 case-sampling ranges, not a scoring bug — `phs.py`'s normalization was
verified by hand against stored values and matches Section 8 exactly for every component, including this
one. Options going forward, not yet decided: (a) report as a limitation (this perturbation type is
intrinsically subtle for the current case ensemble — arguably a legitimate finding, not just a weakness);
(b) redefine the perturbation's epsilon to scale with `τ_decay` instead of the capped `T`, which would
need Issue #8 sign-off since it changes `perturbations.py`; (c) narrow WP2's `Re` sampling range so
`τ_decay` stays closer to `T`. `evaluate_phs.py`'s `evaluate_detection_by_perturbation_type` /
`recall_by_type.png` make this breakdown visible any time the pipeline is re-run, rather than only
showing up as a slightly-lower pooled recall number.

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