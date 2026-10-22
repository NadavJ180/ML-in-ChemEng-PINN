"""
Dataset Generator

Executes the point sampling pipeline for all TGV cases and exports
the resulting dictionaries of tensors as .pt files for PINN training.

Usage:
    python src/data/generate_datasets.py      # data/cases_metadata.json -> data/tensors/
    python src/data/generate_datasets.py --metadata cases/confirmation_cases.json --output_dir <folder>
"""

import argparse
import json
import torch
import sys
from pathlib import Path

# 1. Define the project root
project_root = Path(__file__).parent.parent.parent

# 2. Force Python to add the project root to its search path
sys.path.append(str(project_root))

# Import the physics and data generation functions
from src.data.point_samplers import generate_case_dataset
from src.physics.taylor_green import compute_T, compute_nu, generate_tgv
from src.utils.seed import set_global_seed
from src.utils.console import enable_utf8_output

# Case lists a case file can hold: data/cases_metadata.json has the first three,
# cases/confirmation_cases.json has "confirmation". Cases are generated in this order.
SPLITS = ("train", "validation", "test", "confirmation")


def parse_args():
    """
    Parses command-line arguments.

    Outputs:
        args (argparse.Namespace): metadata and output_dir (None means the default path).
    """
    parser = argparse.ArgumentParser(description="Generate the .pt point clouds for every case in a case file")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Case file to read (default: data/cases_metadata.json). Every case in its "
                             "train/validation/test/confirmation lists is generated.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Folder to write {case_id}.pt into (default: data/tensors)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lock the random number generator so everyone gets the exact same points
    set_global_seed(42)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    metadata_path = Path(args.metadata) if args.metadata else project_root / "data" / "cases_metadata.json"

    # We will save the tensors in a dedicated subfolder to keep data/ clean
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run sampler.py first.")

    with open(metadata_path, "r") as f:
        dataset = json.load(f)

    all_cases = [case for split in SPLITS for case in dataset.get(split, [])]
    if not all_cases:
        raise ValueError(f"No cases in {metadata_path}: expected a list under one of {SPLITS}.")

    print(f"Generating PyTorch tensors for {len(all_cases)} cases...")

    for case in all_cases:
        case_id = case["case_id"]

        # A pre-registered case carries its own seed for its points, so its
        # tensors are the same whichever other cases are generated with it.
        # The original 30 cases have no seed_points and keep drawing from the
        # single seed-42 stream above, so their tensors do not change.
        if "seed_points" in case:
            set_global_seed(case["seed_points"])

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

        # Calculate and store ic_true directly in the dictionary, via the same
        # analytical TGV formula used everywhere else (generate_tgv), evaluated at t=0
        U0_val = case["U0"]
        k_val = case["k"]
        phi_x_val = case["phi_x"]
        phi_y_val = case["phi_y"]
        nu_val = compute_nu(U0_val, case["Re"], k_val)
        x_ic = case_data["ic"][:, 0:1]
        y_ic = case_data["ic"][:, 1:2]
        t_ic = torch.zeros_like(x_ic)

        u_true, v_true, _ = generate_tgv(x_ic, y_ic, t_ic, U0_val, k_val, phi_x_val, phi_y_val, nu_val)
        case_data["ic_true"] = torch.cat([u_true, v_true], dim=1)

        # Save the dictionary as a compressed PyTorch file
        save_path = output_dir / f"{case_id}.pt"
        torch.save(case_data, save_path)

    print(f"✅ Successfully generated and exported {len(all_cases)} datasets to {output_dir}")

if __name__ == "__main__":
    enable_utf8_output()
    main()