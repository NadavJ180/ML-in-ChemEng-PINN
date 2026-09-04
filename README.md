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
│   │   ├── phs.py                   # Pure formula module: Smom/Sdiv/Sbc(ratio)/SE, normalization, scoring
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
│   │   ├── normalized_components_vs_epsilon.png, scores_vs_epsilon.png
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
git clone https://github.com/nadavj180/ml-in-chemeng-pinn.git
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
Builds the perturbed-field dataset (5 perturbation types × 6 epsilon values + 1 clean baseline, per trained case):
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
Computes 4 components (Smom, Sdiv, Sbc, SE) for every field in the hallucination index,
calibrates normalizers/threshold from the validation split, and evaluates detection (ROC-AUC,
Precision, Recall, F1) on the held-out test split against 2 residual-only baselines plus PHS itself
(Score1/2/3, Score3 = PHS):
```bash
python src/detection/evaluate_phs.py                              # full run, all cases
python src/detection/evaluate_phs.py --n_interior 3000 --n_time 8 --energy_res 16   # fast smoke test
```
Writes the scored dataset to `data/phs_scores/`, and ROC curves, score distributions, per-type recall,
and epsilon-response plots to `plots/phs_evaluation/`. See `src/detection/phs.py`'s module docstring for
the exact formulas and every deliberate deviation from the write-up's literal notation (including `Sbc`
now being a near/far residual RATIO rather than the write-up's literal boundary-value comparison; see
[`FINDINGS.md`](FINDINGS.md) for why the original formula was replaced entirely, not kept alongside a
replacement).

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
case Figure 1 illustrates (`case_00` by default). See [`FINDINGS.md`](FINDINGS.md) for two rendering bugs
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
      [`FINDINGS.md`](FINDINGS.md)), and boundary — with a shared `apply_perturbation` dispatcher
      (`src/hallucinations/perturbations.py`, `generate_hallucinations.py`).
- [x] **Perturbation Verification (Issue #9):** Visual-plausibility and violation-activation audit against
      WP4's acceptance criteria (`src/hallucinations/verify_hallucinations.py`).
- [x] **Physical Hallucination Score (Issue #10):** Smom/Sdiv/Sbc(near/far ratio)/SE components,
      validation-split normalization, and threshold calibration (`src/detection/phs.py`).
- [x] **Detection Metrics & Baseline Comparison (Issue #11):** ROC-AUC/Precision/Recall/F1 for PHS vs. 2
      baselines, plus a per-perturbation-type recall breakdown (`src/detection/evaluate_phs.py`). The
      acceptance bar (`AUC(PHS) > 0.90`) is met on the current epsilon range (`AUC(Score3_PHS_full) =
      0.908`) — see [`FINDINGS.md`](FINDINGS.md) for the full investigation, including why
      `Score2_momentum_divergence` (`AUC = 0.928`) still edges out PHS in this pooled ranking, and why that's
      an expected consequence of testing PHS against its own detection boundary rather than a flaw.
- [x] **Publication Figures & Tables (Issues #12-13):** Figures 1-3 and Tables 1-2, IEEE double-column
      sized (`src/detection/publication_figures.py`, output in `plots/paper_figures/`). One caveat: Figure
      1's residual panel should be regenerated against the actual retrained models before being treated as
      final — see [`FINDINGS.md`](FINDINGS.md).
- [ ] **Technical Report & Archiving (Issue #14):** Not started.

## 📝 Documentation
Please refer to the `PINN suggestion-1.pdf` within the repository for the full academic roadmap, methodological details, and risk management strategies. [`FINDINGS.md`](FINDINGS.md) is the project's research log — every bug, design change, and investigation, documented as it happened; read it for the full "why" behind a design decision. For a practical run-order guide (which script to run when, what it needs, what it costs, and when it's safe to skip), see [`PIPELINE.md`](PIPELINE.md); [`run_pipeline.py`](run_pipeline.py) automates running every stage end to end.

## 🚀 Future Research Directions

Having established a robust baseline pipeline and trained 30 independent Taylor-Green Vortex (TGV) models, the Physical Hallucination Score (PHS) framework opens up several exciting, high-impact avenues for subsequent academic research:

---

### 1. Adversarial Hallucination Generation (Automating the Fakes)
* **Objective:** Replace manual mathematical perturbations with automated, target-driven fakes.
* **Approach:** Implement an adversarial neural network (similar to a GAN generator) whose sole objective is to generate flow fields that successfully bypass the global PHS while minimizing visible spatial distortion. This tests if the PHS framework can survive active adversarial "attacks" from an AI optimizing for physical evasion.

### 2. Spatially Localized Hallucination Mapping (Finding Where It Broke)
* **Objective:** Transition from a single, global diagnostic metric to a spatially resolved diagnostic tool.
* **Approach:** Expand the global scalar PHS into a pixel-wise Spatial Hallucination Heatmap. By plotting the localized residual fields of the Navier-Stokes and divergence equations, the auditor can pinpoint exactly *where* in the physical domain (e.g., near boundaries or vortex centers) the model is hallucinating.

### 3. Robustness to Experimental and Sensor Noise
* **Objective:** Validate the detector's capability under real-world, imperfect engineering conditions.
* **Approach:** Inject varying levels of Gaussian white noise into the pristine validation and test datasets. The research goal is to demonstrate that the PHS can mathematically distinguish between standard, expected experimental measurement noise and smooth, structurally incorrect deep-learning hallucinations.

### 4. Cross-PDE Methodological Generalization
* **Objective:** Prove that the normalization and thresholding framework is universally applicable to physics-informed models.
* **Approach:** Port the exact PHS methodology (calculating component-wise residuals, normalizing them using clean validation cases, and establishing a unified detection threshold tau) to entirely different partial differential equations, such as the Heat Equation, Burgers' Equation, or 3D flow systems.