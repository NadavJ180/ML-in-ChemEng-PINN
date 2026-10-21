"""
Data Sampling Engine for Taylor-Green Vortex (TGV) Simulations

This module generates randomized parameter sets for TGV simulations using specified 
statistical distributions (Uniform, Categorical, LogUniform). It then strictly 
partitions the generated cases into training/calibration, validation, and testing 
datasets, exporting the final metadata to a JSON file for use in Physics-Informed 
Neural Network (PINN) training.

With --confirmation it instead writes the 30 pre-registered confirmation cases
(test only, 10 per k, each with its own seeds) to cases/confirmation_cases.json.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.seed import set_global_seed
from src.utils.console import enable_utf8_output


def generate_randomized_cases(num_cases: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates randomized TGV parameter sets using specified distributions.

    Args:
        num_cases: The total number of simulation cases to generate. Defaults to 30.
        seed     : The random seed used for reproducibility across runs. Defaults to 42.

    Returns:
        List: A list of dictionaries, where each dictionary contains 
              the physical parameters (U0, k, Re, phi_x, phi_y) and metadata for a single case.
    """
    # Set seed for strict reproducibility across runs
    set_global_seed(seed)
    
    cases = []
    for i in range(num_cases):
        # Calculate LogUniform limits for Re
        log_re_min = np.log(20)
        log_re_max = np.log(300)
        
        case = {
            "case_id": f"case_{i:02d}",
            "seed": seed + i,  # Unique sub-seed for potential individual case generation
            "U0": float(np.random.uniform(0.5, 2.0)),
            "k": int(np.random.choice([1, 2, 3])),
            "Re": float(np.exp(np.random.uniform(log_re_min, log_re_max))), #LogUniform sampeling
            "phi_x": float(np.random.uniform(0, 2 * np.pi)),
            "phi_y": float(np.random.uniform(0, 2 * np.pi))
        }
        cases.append(case)
        
    return cases


def split_dataset(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Partitions the dataset into training, validation, and testing splits.

    Args:
        cases         : The complete list of generated case dictionaries.

    Returns:
        Dict[str, Any]: A dictionary containing the dataset metadata and partitioned cases.
        
    Raises:
        ValueError    : If the total number of cases provided is not exactly 30.
    """
    if len(cases) != 30:
        raise ValueError(f"Expected exactly 30 cases, got {len(cases)}")
        
    return {
        "metadata": {
            "description": "Randomized TGV simulation cases for PINN calibration.",
            "total_cases": len(cases)
        },
        "train": cases[:20],         
        "validation": cases[20:25],  
        "test": cases[25:30]         
    }


def generate_confirmation_cases(seed: int = 2026, first_case: int = 30, cases_per_k: int = 10) -> Dict[str, Any]:
    """
    Generates the pre-registered confirmation cases, which are used for testing only.

    k is fixed by position (cases_per_k cases for each k = 1, 2, 3, in that order); U0, Re,
    phi_x and phi_y are drawn from the same distributions as generate_randomized_cases().
    Every case carries its own seeds, where n is the case number:
        seed_points  = 3000 + n : collocation, IC and BC points in the tensors
        seed_train   = 4000 + n : weight initialization and mini-batches
        seed_scoring = 1000 + n : PHS scoring points (the rule evaluate_phs.py already uses)

    Args:
        seed        : The sampler seed. Defaults to 2026.
        first_case  : The number of the first case. Defaults to 30.
        cases_per_k : The number of cases for each k. Defaults to 10.

    Returns:
        Dict[str, Any]: The metadata plus the cases under the single split "confirmation",
                        in the same layout as split_dataset() returns.
    """
    # A local generator, so calling this (e.g. from the tests) leaves the global RNGs untouched
    rng = np.random.default_rng(seed)
    log_re_min = np.log(20)
    log_re_max = np.log(300)

    cases = []
    for k in (1, 2, 3):
        for _ in range(cases_per_k):
            n = first_case + len(cases)
            cases.append({
                "case_id": f"case_{n:02d}",
                "split": "confirmation",
                "k": k,
                "U0": float(rng.uniform(0.5, 2.0)),
                "Re": float(np.exp(rng.uniform(log_re_min, log_re_max))),  # LogUniform sampling
                "phi_x": float(rng.uniform(0, 2 * np.pi)),
                "phi_y": float(rng.uniform(0, 2 * np.pi)),
                "seed_points": 3000 + n,
                "seed_train": 4000 + n,
                "seed_scoring": 1000 + n
            })

    return {
        "metadata": {
            "description": "Pre-registered confirmation cases (test only)",
            "total_cases": len(cases),
            "sampler_seed": seed,
            "created_by": "python src/data/sampler.py --confirmation"
        },
        "confirmation": cases
    }


def export_dataset(dataset: Dict[str, Any], output_path: Path, newline: str = None) -> None:
    """
    Exports the dataset dictionary to a structured JSON file.

    Args:
        dataset (Dict[str, Any]): The fully partitioned dataset dictionary.
        output_path (Path): The file path where the resulting JSON file will be saved.
        newline (str): Passed to open(). None (the default) writes the platform's line
                       endings; "\\n" writes the same bytes on every platform.
    """
    with open(output_path, "w", newline=newline) as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Successfully exported dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Generate the randomized TGV case parameters.")
    parser.add_argument("--confirmation", action="store_true",
                        help="Write the 30 pre-registered confirmation cases (test only) to "
                             "cases/confirmation_cases.json instead of data/cases_metadata.json")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    if args.confirmation:
        cases_dir = project_root / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        # LF on every platform, so the tracked file regenerates to the same bytes anywhere
        export_dataset(generate_confirmation_cases(), cases_dir / "confirmation_cases.json", newline="\n")
    else:
        # Define the top-level data directory
        data_dir = project_root / "data"

        # Safely create the data directory if it does not exist
        data_dir.mkdir(parents=True, exist_ok=True)

        output_file = data_dir / "cases_metadata.json"

        # Execute the pipeline
        tgv_cases = generate_randomized_cases(num_cases=30, seed=42)
        partitioned_dataset = split_dataset(tgv_cases)
        export_dataset(partitioned_dataset, output_file)