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
│   │   └── pinn.py
│   ├── physics/                     # Physical equations and analytical models
│   │   ├── navier_stokes.py
│   │   └── taylor_green.py
│   └── utils/                       # Utility functions (e.g., seeding)
│       └── seed.py
├── tests/                           # Pytest suite for physics, samplers, and losses
│   ├── test_pinn_loss.py
│   ├── test_point_samplers.py
│   └── test_residuals.py
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

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

## 🎯 Key Deliverables & Roadmap
Based on the project blueprint, the following components are implemented or actively being developed:
- [x] **Taylor-Green Vortex Generator:** Analytical flow generation (`src/physics/taylor_green.py`).
- [x] **Residual Verification:** Navier-Stokes residual computation (`src/physics/navier_stokes.py` & tests).
- [x] **Baseline PINN Architecture:** Configurable neural network (`src/models/pinn.py`).
- [ ] **Baseline PINN Training:** Generating valid PINN flow fields for normalization.
- [ ] **Hallucinated Flow Fields:** Introducing controlled perturbations (spatial, temporal, boundary).
- [ ] **Physical Hallucination Score (PHS):** Implementation and normalization of the detection metric.
- [ ] **Detection Metrics & Baseline Comparison:** Statistical evaluation of the PHS.

## 📝 Documentation
Please refer to the `PINN suggestion-1.pdf` within the repository for the full academic roadmap, methodological details, and risk management strategies.

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