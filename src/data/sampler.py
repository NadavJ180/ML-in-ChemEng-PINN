"""
Data Sampling Engine for Taylor-Green Vortex (TGV) Simulations

This module generates randomized parameter sets for TGV simulations using specified 
statistical distributions (Uniform, Categorical, LogUniform). It then strictly 
partitions the generated cases into training/calibration, validation, and testing 
datasets, exporting the final metadata to a JSON file for use in Physics-Informed 
Neural Network (PINN) training.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


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
    np.random.seed(seed)
    
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


def export_dataset(dataset: Dict[str, Any], output_path: Path) -> None:
    """
    Exports the dataset dictionary to a structured JSON file.

    Args:
        dataset (Dict[str, Any]): The fully partitioned dataset dictionary.
        output_path (Path): The file path where the resulting JSON file will be saved.
    """
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Successfully exported dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    # Define the top-level data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    # Safely create the data directory if it does not exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "cases_metadata.json"
    
    # Execute the pipeline
    tgv_cases = generate_randomized_cases(num_cases=30, seed=42)
    partitioned_dataset = split_dataset(tgv_cases)
    export_dataset(partitioned_dataset, output_file)