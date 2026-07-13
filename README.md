# Physical Hallucination Detection in PINN-Generated 2D Navier-Stokes Flow Fields

**0560412: Machine Learning for Chemical Engineering - Final Project**
**Authors:** Nadav Jean & Alon Solomiak

## Project Objective
Developing a compact benchmark to detect physical hallucinations in PINN-generated 2D incompressible Navier-Stokes flow fields using a Taylor-Green Vortex generator.

## Repository Structure
* `/src` - Object-oriented source code (models, physics, data generation, utilities).
* `/notebooks` - Exploratory analysis and scratchpads.
* `/tests` - Unit tests for analytical residual verification.
* `/data` - Directory for synthetic cases and evaluation grids.

## Environment Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment.
4. Install dependencies: `pip install -r requirements.txt`

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