"""
Accuracy audit of the final iteration (#37): the true error of every model.

Barak asks us to "evaluate each checkpoint against the analytical solution".
The Taylor-Green vortex has an exact solution (generate_tgv in
src/physics/taylor_green.py), so the true error of any model can be measured.
These errors are the ruler that the natural-error study (#39) and the figures
(#36) hold the detection scores against.

Each model is compared with the exact solution on a fresh, evenly spaced
64 x 64 x 11 grid (PROTOCOL.md section 6):
    x, y = 2*pi*i/64 for i = 0 ... 63   (2*pi is left out: on the periodic domain it is the point 0)
    t    = 11 evenly spaced times from 0 to T, both ends included
That is 45,056 points, which are neither training points nor the 64 x 64 x 20
eval_grid of the acceptance test.

Three relative L2 errors, sqrt(sum (model - exact)^2 / sum exact^2), where for
the velocity the u and v sums are added together:
    relL2_uv     the velocity over all points. This is the main number.
    relL2_p      the pressure, after removing its mean over x and y at each time
                 from both the model and the exact solution, since the equations
                 fix a PINN's pressure only up to a constant
    relL2_uv_T   the velocity at t = T only
A model is inaccurate when relL2_uv > 0.10 (Barak, 18 Oct 2026).

The models are found from the case file: a case folder <models_dir>/<case_id>/
holds the 7 checkpoints of a final-iteration run (#40), and otherwise the case
has one old model, <models_dir>/<case_id>.pth. Every case in the case file must
have its models: none is skipped.

Output: one CSV row per model, with the columns of CSV_COLUMNS. #39 joins it
with its checkpoint scores on (case_id, stage).

Usage:
    # The 30 old models, checked against the old package's 04_checkpoints/checkpoints.csv
    python src/models/accuracy_audit.py --metadata data/cases_metadata.json --models_dir models --out results/accuracy/reference_errors_existing.csv
    # The 210 checkpoints of the 30 new cases
    python src/models/accuracy_audit.py --metadata cases/confirmation_cases.json --models_dir models/confirmation --out results/accuracy/reference_errors_confirmation.csv
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.checkpoints import (FINAL_STAGE, LEGACY_CHECKPOINT, SPLITS, STAGES, check_sidecar,
                                    load_checkpoint)
from src.physics.taylor_green import compute_nu, compute_T, generate_tgv
from src.utils.console import enable_utf8_output

GRID_N = 64               # Values of x, and of y
GRID_NT = 11              # Values of t
INACCURATE_ABOVE = 0.10   # A model is inaccurate when relL2_uv > this (Barak, 18 Oct 2026)

# The columns of the output CSV, in order:
#   case_id, k, split   from the case file
#   stage               "adam_00100" ... "adam_05000", "lbfgs_final"; "legacy" for an old model
#   adam_step           Adam steps taken when the checkpoint was saved (its sidecar's adam_steps;
#                       5000 for lbfgs_final). Empty for an old model, which has no sidecar.
#   relL2_uv, relL2_p, relL2_uv_T   the three errors above
#   passed_acceptance   whether the case's run passed: its lbfgs_final sidecar says PASSED
#                       (PROTOCOL.md section 2). The same on all 7 rows of a case, so a failed run
#                       stays visible at every stage. Empty for an old model, which has no sidecar.
#   inaccurate          relL2_uv > INACCURATE_ABOVE
CSV_COLUMNS = ("case_id", "k", "split", "stage", "adam_step", "relL2_uv", "relL2_p", "relL2_uv_T",
               "passed_acceptance", "inaccurate")


def accuracy_grid(T: float):
    """
    Returns the audit grid of a case whose time window is [0, T].

    Inputs:
        T (float): The case's final time (compute_T).

    Outputs:
        (t, x, y): Three float64 tensors of shape (GRID_NT, GRID_N, GRID_N), indexed
                   [time, x, y], so [-1] is the slice at t = T.
    """
    xy = 2 * torch.pi * torch.arange(GRID_N, dtype=torch.float64) / GRID_N
    t = torch.linspace(0.0, T, GRID_NT, dtype=torch.float64)
    return torch.meshgrid(t, xy, xy, indexing="ij")


def relative_l2(pred, exact) -> float:
    """
    Returns sqrt(sum (pred - exact)^2 / sum exact^2), with the sums taken over every
    point of every tensor in the pair of sequences (e.g. (u, v) and (u_exact, v_exact)).
    """
    num = sum(torch.sum((a - b) ** 2).item() for a, b in zip(pred, exact))
    den = sum(torch.sum(b ** 2).item() for b in exact)
    return math.sqrt(num / den)


def reference_errors(model, case: dict, device: str = "cpu") -> dict:
    """
    Compares one model with the exact Taylor-Green solution of its case on the audit grid.

    Inputs:
        model (nn.Module): Maps raw (x, y, t) points, shape (N, 3), to (u, v, p), shape (N, 3).
        case (dict): The case's entry in the case file (U0, Re, k, phi_x, phi_y).
        device (str): Where to evaluate the model.

    Outputs:
        dict: relL2_uv, relL2_p and relL2_uv_T (floats).
    """
    U0, Re, k = case["U0"], case["Re"], case["k"]
    t, x, y = accuracy_grid(compute_T(U0=U0, Re=Re, k=k))
    coords = torch.stack([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(coords.to(device)).to(device="cpu", dtype=torch.float64).reshape(*t.shape, 3)
    u, v, p = pred[..., 0], pred[..., 1], pred[..., 2]
    u_exact, v_exact, p_exact = generate_tgv(x, y, t, U0, k, case["phi_x"], case["phi_y"], compute_nu(U0, Re, k))

    # Only pressure differences are fixed by the equations, so each time slice is compared without its mean
    p = p - p.mean(dim=(1, 2), keepdim=True)
    p_exact = p_exact - p_exact.mean(dim=(1, 2), keepdim=True)
    return {
        "relL2_uv": relative_l2((u, v), (u_exact, v_exact)),
        "relL2_p": relative_l2((p,), (p_exact,)),
        "relL2_uv_T": relative_l2((u[-1], v[-1]), (u_exact[-1], v_exact[-1])),
    }


def read_cases(metadata_path) -> list:
    """
    Returns every case of a case file, in file order, each with its split:
    train, validation and test for data/cases_metadata.json, confirmation for
    cases/confirmation_cases.json.
    """
    with open(metadata_path, "r") as f:
        dataset = json.load(f)
    cases = [{**case, "split": split} for split in SPLITS for case in dataset.get(split, [])]
    if not cases:
        raise ValueError(f"No cases in {metadata_path}: expected a list under one of {SPLITS}.")
    return cases


def case_checkpoints(case_id: str, models_dir) -> list:
    """
    Returns the model files of one case as (stage, path) pairs, in training order.

    A folder <models_dir>/<case_id>/ must hold all 7 checkpoints of STAGES. Without
    that folder, the case must have one old model, <models_dir>/<case_id>.pth.
    """
    models_dir = Path(models_dir)
    case_dir = models_dir / case_id
    if case_dir.is_dir():
        checkpoints = [(stage, case_dir / f"{stage}.pth") for stage in STAGES]
        missing = [path.name for _, path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{case_dir} is missing {', '.join(missing)}; every case keeps all "
                                    f"{len(STAGES)} checkpoints (PROTOCOL.md section 2)")
        return checkpoints
    legacy_path = models_dir / f"{case_id}.pth"
    if legacy_path.exists():
        return [(LEGACY_CHECKPOINT, legacy_path)]
    raise FileNotFoundError(f"No model for {case_id}: neither {case_dir}/ nor {legacy_path} exists")


def audit(metadata_path, models_dir, device: str = "cpu") -> pd.DataFrame:
    """
    Computes the true errors of every model of every case in a case file.

    Inputs:
        metadata_path (str | Path): The case file.
        models_dir (str | Path): The folder with the cases' models (see case_checkpoints).
        device (str): Where to evaluate the models.

    Outputs:
        pd.DataFrame: One row per model, with CSV_COLUMNS, cases in case-file order
                      and each case's models in training order.
    """
    rows = []
    for case in read_cases(metadata_path):
        case_rows = []
        final_status = None
        for stage, path in case_checkpoints(case["case_id"], models_dir):
            model, info = load_checkpoint(path, device=device, k=case["k"], metadata_paths=[metadata_path])
            check_sidecar(info, case, stage, path)
            if stage == FINAL_STAGE:
                final_status = info["status"]
            case_rows.append({
                "case_id": case["case_id"],
                "k": case["k"],
                "split": case["split"],
                "stage": stage,
                "adam_step": info["adam_steps"],
                **reference_errors(model, case, device),
            })
        passed = None if final_status is None else final_status == "PASSED"
        rows.extend({**row, "passed_acceptance": passed} for row in case_rows)

    df = pd.DataFrame(rows)
    df["adam_step"] = df["adam_step"].astype("Int64")  # Integers, empty for an old model
    df["inaccurate"] = df["relL2_uv"] > INACCURATE_ABOVE
    return df[list(CSV_COLUMNS)]


def print_summary(df: pd.DataFrame) -> None:
    """Prints, for each stage, how many models are inaccurate and the range of relL2_uv."""
    print(f"  {'stage':12s} {'models':>6s} {'inaccurate':>10s}   relL2_uv min / median / max")
    for stage, group in df.groupby("stage", sort=False):
        errors = group["relL2_uv"]
        print(f"  {stage:12s} {len(group):6d} {int(group['inaccurate'].sum()):10d}   "
              f"{errors.min():.3e} / {errors.median():.3e} / {errors.max():.3e}")


def parse_args(argv=None):
    """
    Parses the command line.

    Inputs:
        argv (list[str] | None): Arguments; None reads sys.argv.

    Outputs:
        argparse.Namespace: metadata, models_dir, out and device (str).
    """
    parser = argparse.ArgumentParser(description="Measure the true error of every model against the exact "
                                                 "Taylor-Green solution (#37).")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Case file: data/cases_metadata.json (the 30 old cases) or "
                             "cases/confirmation_cases.json (the 30 new ones).")
    parser.add_argument("--models_dir", type=str, required=True,
                        help="Folder with the models: models (old case_XX.pth files) or models/confirmation "
                             "(one folder of 7 checkpoints per case).")
    parser.add_argument("--out", type=str, required=True, help="CSV to write, e.g. results/accuracy/....csv")
    parser.add_argument("--device", type=str, default="cpu", help="Where to evaluate the models (default: cpu).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Audits every model of the case file and writes the CSV.

    Inputs:
        argv (list[str] | None): Command-line arguments; None reads sys.argv.

    Outputs:
        pd.DataFrame: The rows written to --out.
    """
    args = parse_args(argv)
    df = audit(args.metadata, args.models_dir, device=args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform; the floats are written in full, so they read back exactly
    # with pd.read_csv(out, float_precision="round_trip")
    df.to_csv(out, index=False, lineterminator="\n")

    print(f"Wrote {out}: {len(df)} models of {df['case_id'].nunique()} cases")
    print_summary(df)
    return df


if __name__ == "__main__":
    enable_utf8_output()
    main()
