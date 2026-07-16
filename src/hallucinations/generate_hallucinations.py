"""
Hallucination Dataset Generator (Section 7 execution script)

For every case_id that has a trained, saved PINN (models/{case_id}_best.pth),
this script:

  1. Loads the model and its evaluation grid (reusing the exact grid the
     model was trained/evaluated on when available, for consistency).
  2. Runs one clean forward pass to obtain the valid (u, v, p) baseline.
  3. Applies all 5 Section 7 perturbation functions at all 5 epsilon
     strengths {0.005, 0.01, 0.02, 0.05, 0.1} -> 25 hallucinated variants.
  4. Saves a single .pt file per case containing the clean baseline plus all
     25 perturbed variants, each tagged with clear categorization labels.
  5. Appends every (case, perturbation, epsilon) combination as a row to a
     global CSV/JSON index so downstream detection code (PHS scoring) can
     enumerate the full hallucinated dataset without re-opening every file.

Usage:
    python src/hallucinations/generate_hallucinations.py
    python src/hallucinations/generate_hallucinations.py --case_id case_00
    python src/hallucinations/generate_hallucinations.py --device cpu
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# 1. Define the project root 
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.physics.taylor_green import compute_nu, compute_T
from src.data.point_samplers import generate_evaluation_grid
from src.hallucinations.perturbations import (
    apply_perturbation,
    EPSILON_VALUES,
    PERTURBATION_NAMES,
)


def parse_args():
    """
    Parses command-line arguments controlling which cases to process, which
    device to run on, and how large each VRAM-safe forward-pass chunk is.

    Inputs:
        None (reads directly from sys.argv).

    Outputs:
        args (argparse.Namespace): Parsed arguments with fields
            case_id (str | None), device (str), chunk_size (int).
    """
    parser = argparse.ArgumentParser(description="Generate the physical hallucination dataset.")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Only generate hallucinations for this specific case_id.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=8000,
                        help="Chunk size for forward passes over the evaluation grid (VRAM safety).")
    return parser.parse_args()


def load_case_metadata(metadata_path: Path):
    """
    Loads cases_metadata.json and flattens the train/validation/test splits
    into a single lookup dictionary keyed by case_id.

    Inputs:
        metadata_path (Path): Path to data/cases_metadata.json.

    Outputs:
        dict: Mapping of case_id (str) -> case metadata dict
              (containing "Re", "U0", "k", "phi_x", "phi_y", "seed", etc.).
    """
    with open(metadata_path, "r") as f:
        dataset = json.load(f)

    all_cases = dataset["train"] + dataset["validation"] + dataset["test"]
    return {c["case_id"]: c for c in all_cases}


def get_evaluation_grid(case_id: str, T: float, tensors_dir: Path) -> torch.Tensor:
    """
    Reuses the exact evaluation grid the model saw during training/validation
    (data/tensors/{case_id}.pt) when available, since that keeps the
    hallucination grid perfectly consistent with the WP3 acceptance-criteria
    grid. Falls back to regenerating a fresh 64x64x20 grid otherwise.

    Inputs:
        case_id (str): The case identifier (e.g. "case_00"), used to locate
                        the case's saved tensor file.
        T (float): The case's final simulation time, used only if the tensor
                   file is missing and the grid must be regenerated.
        tensors_dir (Path): Directory containing the {case_id}.pt tensor files
                             produced by generate_datasets.py.

    Outputs:
        torch.Tensor: Flattened (x, y, t) evaluation grid coordinates of
                      shape (N, 3), dtype float64.
    """
    pt_path = tensors_dir / f"{case_id}.pt"
    if pt_path.exists():
        case_data = torch.load(pt_path, map_location="cpu")
        return case_data["eval_grid"].to(torch.float64)
    return generate_evaluation_grid(T).to(torch.float64)


def run_model_in_chunks(model, coords: torch.Tensor, device: str, chunk_size: int):
    """
    Runs a no-grad forward pass of the trained PINN over `coords` in
    VRAM-safe chunks, moving each chunk to `device` for inference and back
    to CPU afterward, then concatenates the results into full-size tensors.

    Inputs:
        model (nn.Module): The trained BaselinePINN to run inference with.
        coords (torch.Tensor): Flattened (x, y, t) coordinates, shape (N, 3).
        device (str): Target hardware device for inference ('cuda' or 'cpu').
        chunk_size (int): Number of points to process per forward-pass chunk.

    Outputs:
        u_pred (torch.Tensor): Predicted x-velocity, shape (N, 1), on CPU.
        v_pred (torch.Tensor): Predicted y-velocity, shape (N, 1), on CPU.
        p_pred (torch.Tensor): Predicted pressure, shape (N, 1), on CPU.
    """
    u_chunks, v_chunks, p_chunks = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords[i:i + chunk_size].to(dtype=torch.float64, device=device)
            preds = model(chunk)
            u_chunks.append(preds[:, 0:1].cpu())
            v_chunks.append(preds[:, 1:2].cpu())
            p_chunks.append(preds[:, 2:3].cpu())

    return torch.cat(u_chunks, dim=0), torch.cat(v_chunks, dim=0), torch.cat(p_chunks, dim=0)


def generate_case_hallucinations(case_id, case_meta, project_root, device, chunk_size):
    """
    Builds the full clean + 25-variant hallucination bundle for a single case:
    loads the trained model, runs one clean forward pass on the evaluation
    grid, then applies all 5 perturbations at all 5 epsilon
    strengths to produce the 25 hallucinated variants.

    Inputs:
        case_id (str): The case identifier (e.g. "case_00").
        case_meta (dict): This case's metadata dict, containing "Re", "U0", "k".
        project_root (Path): Repository root, used to locate models/ and data/.
        device (str): Target hardware device for inference ('cuda' or 'cpu').
        chunk_size (int): Number of points to process per forward-pass chunk.

    Outputs:
        bundle (dict): Full data structure to be saved as
                        {case_id}_hallucinations.pt, containing "case_id",
                        "Re", "U0", "k", "nu", "T", "coords", "clean", and
                        "perturbed" (nested by perturbation name -> epsilon key).
        index_rows (list[dict]): One row per (perturbation, epsilon) combo
                                  (plus one "clean" row), each with case_id,
                                  perturbation_type, epsilon, Re, U0, k, T,
                                  and label, for the global categorization index.
    """
    Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
    nu = compute_nu(U0, Re, k)
    T = compute_T(U0, Re, k)

    # 1. Load the trained model
    model_path = project_root / "models" / f"{case_id}_best.pth"
    model = BaselinePINN(k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.to(torch.float64)

    # 2. Load / build the evaluation grid and get the clean baseline
    tensors_dir = project_root / "data" / "tensors"
    eval_grid = get_evaluation_grid(case_id, T, tensors_dir)

    x = eval_grid[:, 0:1]
    y = eval_grid[:, 1:2]
    t = eval_grid[:, 2:3]

    u_clean, v_clean, p_clean = run_model_in_chunks(model, eval_grid, device, chunk_size)

    # Note: temporal_mismatch re-queries the model directly, so we do that on
    # the same device/model, but everything is stored back on CPU for saving.
    coords = {"x": x, "y": y, "t": t}
    fields = {"u": u_clean, "v": v_clean, "p": p_clean}
    params = {"U0": U0, "k": k, "T": T}

    bundle = {
        "case_id": case_id,
        "Re": Re,
        "U0": U0,
        "k": k,
        "nu": nu,
        "T": T,
        "coords": coords,
        "clean": fields,
        "perturbed": {},
    }

    index_rows = []

    for perturbation_name in PERTURBATION_NAMES:
        bundle["perturbed"][perturbation_name] = {}
        for epsilon in EPSILON_VALUES:
            # temporal_mismatch needs the actual model (on `device`) to re-query
            # at a shifted time; all other perturbations are pure value-space math.
            model_for_perturbation = model if perturbation_name == "temporal_mismatch" else None

            if model_for_perturbation is not None:
                # Run the shifted-time query on-device in chunks for VRAM safety,
                # then bring the result back to CPU to match the rest of the bundle.
                hallucinated = _temporal_mismatch_chunked(model, coords, params, epsilon, device, chunk_size)
                # Only redefines (u, v) for this perturbation; pressure
                # stays at its clean value.
                hallucinated["p"] = fields["p"].clone()
            else:
                hallucinated = apply_perturbation(
                    perturbation_name, fields, coords, params, epsilon, model=None
                )

            eps_key = f"eps_{epsilon}"
            bundle["perturbed"][perturbation_name][eps_key] = hallucinated

            index_rows.append({
                "case_id": case_id,
                "perturbation_type": perturbation_name,
                "epsilon": epsilon,
                "Re": Re,
                "U0": U0,
                "k": k,
                "T": T,
                "label": "hallucinated",
            })

    # Also register the clean baseline in the index for completeness
    index_rows.append({
        "case_id": case_id,
        "perturbation_type": "none",
        "epsilon": 0.0,
        "Re": Re,
        "U0": U0,
        "k": k,
        "T": T,
        "label": "clean",
    })

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return bundle, index_rows


def _temporal_mismatch_chunked(model, coords, params, epsilon, device, chunk_size):
    """
    Chunked version of the temporal_mismatch perturbation (see
    src.hallucinations.perturbations.perturb_temporal_mismatch) so the
    shifted-time re-query never exceeds VRAM limits on large evaluation grids.

    Inputs:
        model (nn.Module): The trained BaselinePINN to re-query.
        coords (dict): Evaluation grid coordinates with keys "x", "y", "t",
                        each a torch.Tensor of shape (N, 1).
        params (dict): Case-specific physical constants; requires "T".
        epsilon (float): Perturbation strength; the time shift is epsilon * T.
        device (str): Target hardware device for inference ('cuda' or 'cpu').
        chunk_size (int): Number of points to process per forward-pass chunk.

    Outputs:
        dict: {"u": hallucinated u (N, 1), "v": hallucinated v (N, 1)} on CPU.
              Does NOT include "p" — the caller is responsible for filling
              it in with the clean pressure field, since only
              redefined (u, v) for this perturbation.
    """
    T = params["T"]
    x, y, t = coords["x"], coords["y"], coords["t"]
    t_shifted = t + epsilon * T

    u_chunks, v_chunks = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk_size):
            x_c = x[i:i + chunk_size].to(dtype=torch.float64, device=device)
            y_c = y[i:i + chunk_size].to(dtype=torch.float64, device=device)
            t_c = t_shifted[i:i + chunk_size].to(dtype=torch.float64, device=device)
            shifted_coords = torch.cat([x_c, y_c, t_c], dim=1)
            preds = model(shifted_coords)
            u_chunks.append(preds[:, 0:1].cpu())
            v_chunks.append(preds[:, 1:2].cpu())

    u_tilde = torch.cat(u_chunks, dim=0)
    v_tilde = torch.cat(v_chunks, dim=0)

    # "p" is intentionally omitted here; the caller fills it in with the clean
    # pressure field, since only redefined (u, v) for this perturbation.
    return {"u": u_tilde, "v": v_tilde}


def main():
    """
    Entry point for the hallucination sweep. Loads cases_metadata.json,
    iterates over every case_id that has a trained model saved in models/,
    builds and saves its clean + 25-variant hallucination bundle via
    generate_case_hallucinations(), and writes a combined categorization
    index (CSV + JSON) covering every case's clean and hallucinated entries.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes one {case_id}_hallucinations.pt file per processed case
        plus hallucination_index.csv / hallucination_index.json to
        data/hallucinations/, and prints a summary to stdout.
    """
    args = parse_args()

    metadata_path = project_root / "data" / "cases_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata at {metadata_path}")

    case_meta_by_id = load_case_metadata(metadata_path)

    models_dir = project_root / "models"
    output_dir = project_root / "data" / "hallucinations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_id:
        case_ids = [args.case_id]
    else:
        case_ids = sorted(case_meta_by_id.keys())

    all_index_rows = []
    n_generated, n_skipped = 0, 0

    print("=" * 60)
    print("🧪 GENERATING PHYSICAL HALLUCINATION DATASET")
    print(f"Perturbations: {PERTURBATION_NAMES}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print("=" * 60)

    for case_id in case_ids:
        model_path = models_dir / f"{case_id}_best.pth"
        if not model_path.exists():
            print(f"⏭️  Skipping {case_id}: no trained model found at {model_path}")
            n_skipped += 1
            continue

        case_meta = case_meta_by_id.get(case_id)
        if case_meta is None:
            print(f"⏭️  Skipping {case_id}: not found in cases_metadata.json")
            n_skipped += 1
            continue

        print(f"\n[{case_id}] Generating 25 hallucinated variants + clean baseline...")
        bundle, index_rows = generate_case_hallucinations(
            case_id, case_meta, project_root, args.device, args.chunk_size
        )

        save_path = output_dir / f"{case_id}_hallucinations.pt"
        torch.save(bundle, save_path)
        print(f"💾 Saved {save_path.relative_to(project_root)}")

        all_index_rows.extend(index_rows)
        n_generated += 1

    # --- Persist the global categorization index (CSV + JSON) ---
    if all_index_rows:
        index_csv_path = output_dir / "hallucination_index.csv"
        with open(index_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_index_rows)

        index_json_path = output_dir / "hallucination_index.json"
        with open(index_json_path, "w") as f:
            json.dump(all_index_rows, f, indent=2)

        print(f"\n📊 Wrote categorization index: {index_csv_path.relative_to(project_root)}")
        print(f"📊 Wrote categorization index: {index_json_path.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print(f"✅ Generated hallucinations for {n_generated} case(s). Skipped {n_skipped}.")
    print(f"Each generated case contains 1 clean baseline + "
          f"{len(PERTURBATION_NAMES)} x {len(EPSILON_VALUES)} = "
          f"{len(PERTURBATION_NAMES) * len(EPSILON_VALUES)} hallucinated variants.")
    print("=" * 60)


if __name__ == "__main__":
    main()
