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
