"""
Dataset Generator

Executes the point sampling pipeline for all TGV cases and exports
the resulting dictionaries of tensors as .pt files for PINN training.
"""

import json
import torch
import sys
from pathlib import Path

# 1. Define the project root
project_root = Path(__file__).parent.parent.parent

# 2. Force Python to add the project root to its search path
sys.path.append(str(project_root))

# Import your actual physics and data functions
from src.data.point_samplers import generate_case_dataset
from src.physics.taylor_green import compute_T

def main():
    # Lock the random number generator so everyone gets the exact same points
    torch.manual_seed(42)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / "data" / "cases_metadata.json"

    # We will save the tensors in a dedicated subfolder to keep data/ clean
    output_dir = project_root / "data" / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run sampler.py first.")

    with open(metadata_path, "r") as f:
        dataset = json.load(f)

    all_cases = dataset["train"] + dataset["validation"] + dataset["test"]

    print(f"Generating PyTorch tensors for {len(all_cases)} cases...")

    for case in all_cases:
        case_id = case["case_id"]
        T = compute_T(
            Re=case["Re"],
            k=case["k"],
            U0=case["U0"]
        )

        # Generate the unified dictionary of spatial/temporal points
        case_data = generate_case_dataset(T=T, N_interior=100000, N_ic=2000, N_bc=1000)

        # Cast the dictionary values to float64 for consistency with the PINN architecture
        for key, val in case_data.items():
            if isinstance(val, torch.Tensor):
                case_data[key] = val.to(torch.float64)
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, tuple): # Handling the bc dictionary structure
                        case_data[key][subkey] = tuple(t.to(torch.float64) for t in subval)

        # Calculate and store ic_true directly in the dictionary
        U0_val = case["U0"]
        k_val = case["k"]
        # FIX: phi_x and phi_y were never read from `case` here, so ic_true was
        # always built with phi_x=0, phi_y=0 regardless of each case's actual
        # target phase in cases_metadata.json. Every model was therefore
        # trained against the wrong initial condition.
        phi_x_val = case["phi_x"]
        phi_y_val = case["phi_y"]
        x_ic = case_data["ic"][:, 0:1]
        y_ic = case_data["ic"][:, 1:2]

        u_true = U0_val * torch.sin(k_val * x_ic + phi_x_val) * torch.cos(k_val * y_ic + phi_y_val)
        v_true = -U0_val * torch.cos(k_val * x_ic + phi_x_val) * torch.sin(k_val * y_ic + phi_y_val)
        case_data["ic_true"] = torch.cat([u_true, v_true], dim=1)

        # Save the dictionary as a compressed PyTorch file
        save_path = output_dir / f"{case_id}.pt"
        torch.save(case_data, save_path)

    print(f"✅ Successfully generated and exported {len(all_cases)} datasets to {output_dir}")

if __name__ == "__main__":
    main()