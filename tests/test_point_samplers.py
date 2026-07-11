import json
import pytest
import torch
import numpy as np
from pathlib import Path

# Import your actual functions
from src.data.point_samplers import generate_case_dataset
from src.physics.taylor_green import compute_T 


def test_sampler_acceptance_criteria():
    """
    Verifies that all generated spatiotemporal points strictly respect 
    the domain boundaries and that evaluation grids have consistent shapes.
    """
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "data" / "cases_metadata.json"
    
    if not metadata_path.exists():
        pytest.skip("Dataset not found. Run sampler.py first.")
        
    with open(metadata_path, "r") as f:
        dataset = json.load(f)
        
    all_cases = dataset["train"] + dataset["validation"] + dataset["test"]
    
    for case in all_cases:
        case_id = case["case_id"]
        T = compute_T(case["U0"], case["Re"], case["k"]) 
        
        data = generate_case_dataset(T=T, N_interior=10000, N_ic=2000, N_bc=1000)
        
        interior = data["interior"]
        ic = data["ic"]
        bc_x_left, bc_x_right = data["bc"]["x_bounds"]
        bc_y_bottom, bc_y_top = data["bc"]["y_bounds"]
        eval_grid = data["eval_grid"]
        
        # --- ACCEPTANCE CRITERIA 1: Check Shapes ---
        assert eval_grid.shape == (81920, 3), f"[{case_id}] Grid shape mismatch"
        assert interior.shape == (10000, 3), f"[{case_id}] Interior shape mismatch"
        
        # --- ACCEPTANCE CRITERIA 2: Verify Strict Boundary Limits ---
        tol = 1e-6
        
        assert torch.all((interior[:, 0] >= -tol) & (interior[:, 0] <= 2 * np.pi + tol)), "X bounds failed"
        assert torch.all((interior[:, 1] >= -tol) & (interior[:, 1] <= 2 * np.pi + tol)), "Y bounds failed"
        assert torch.all((interior[:, 2] >= -tol) & (interior[:, 2] <= T + tol)), "T bounds failed"
        
        assert torch.all(ic[:, 2] == 0.0), "IC time must be strictly 0"
        
        assert torch.all(bc_x_left[:, 0] == 0.0), "Left BC must be at x=0"
        assert torch.all(bc_x_right[:, 0] == 2 * np.pi), "Right BC must be at x=2π"
        assert torch.allclose(bc_x_left[:, 1:], bc_x_right[:, 1:]), "X boundary Y/T coordinates must match"