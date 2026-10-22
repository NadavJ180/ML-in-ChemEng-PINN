"""
Checkpoint helper: save a model with a small JSON sidecar, and load it back.

A checkpoint is two files that share a name:
    <name>.pth   The model's state_dict, in the same format as the old
                 models/case_XX.pth files, so torch.load + load_state_dict
                 still opens it.
    <name>.json  The sidecar: which case it is, which phase and step it was
                 saved at, the seeds, the training settings and, for
                 lbfgs_final, PASSED/FAILED with the acceptance metrics.

load_checkpoint() opens both these checkpoints and the old models/case_XX.pth
files (no sidecar), so later stages can treat old and new models alike.
"""

import json
import os
import subprocess
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.models.pinn import BaselinePINN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SIDECAR_VERSION = 1

# Every sidecar has exactly these keys, in this order. A key that does not
# apply to a checkpoint is null, never missing.
SIDECAR_FIELDS = (
    "format_version",  # SIDECAR_VERSION
    "case_id",
    "checkpoint",      # File stem: "adam_00100", ..., "lbfgs_final"; "legacy" for an old models/case_XX.pth
    "phase",           # "adam" or "lbfgs"
    "adam_steps",      # Adam steps taken when the copy was made
    "lbfgs_evals",     # L-BFGS closure evaluations taken (0 during Adam)
    "status",          # "PASSED" or "FAILED" for lbfgs_final, else null
    "metrics",         # {"rel_l2", "mse_rc", "mse_rm"} on the eval grid for lbfgs_final, else null
    "losses",          # Loss components of the last training step (evaluated before that step's update)
    "k",
    "Re",
    "U0",
    "phi_x",
    "phi_y",
    "split",
    "seed_points",     # Seed of the case's tensors (null for the original 30 cases)
    "seed_train",      # Seed of weight init + mini-batches (null for the original 30 cases)
    "config",          # Training settings of the run (epochs, L-BFGS limits, batch sizes, device)
    "architecture",
    "dtype",
    "torch_version",
    "git_commit",
    "created_utc",
)

STATUSES = (None, "PASSED", "FAILED")

# Case files that list cases by split; used to find k for an old checkpoint.
DEFAULT_METADATA_PATHS = (
    PROJECT_ROOT / "data" / "cases_metadata.json",
    PROJECT_ROOT / "cases" / "confirmation_cases.json",
)
SPLITS = ("train", "validation", "test", "confirmation")


def _git_commit():
    """Returns the short hash of the checked-out commit, or None outside a git checkout."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _atomic_write(path: Path, write) -> None:
    """
    Writes a file through a temporary file in the same folder, then renames it
    into place, so a crash never leaves a half-written file under the final name.

    Inputs:
        path (Path): The final file path.
        write (callable): Called with the temporary path (str); writes the content there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_checkpoint(model, path, case: dict, *, phase: str, adam_steps: int, lbfgs_evals: int = 0,
                    status=None, metrics=None, losses=None, config=None):
    """
    Saves the model's weights to <path>.pth and a sidecar to <path>.json.

    The sidecar is written first and the .pth last, each through a temporary
    file, so an existing .pth always has a complete sidecar next to it. That
    matters because train_model.py treats an existing lbfgs_final.pth as
    "this case is done".

    Saving only reads the model (the weights are copied to the CPU), so it
    does not change training.

    Inputs:
        model (nn.Module): The BaselinePINN to save.
        path (str | Path): Checkpoint path, e.g. models/confirmation/case_30/adam_00100.pth.
                           The suffix is forced to .pth.
        case (dict): The case's entry from the case file (case_id, k, Re, U0,
                     phi_x, phi_y and, for pre-registered cases, split and seeds).
        phase (str): "adam" or "lbfgs".
        adam_steps (int): Adam steps taken so far.
        lbfgs_evals (int): L-BFGS closure evaluations taken so far.
        status (str | None): "PASSED" or "FAILED" for the final model, else None.
        metrics (dict | None): Acceptance metrics for the final model.
        losses (dict | None): Loss components of the last training step.
        config (dict | None): Training settings of the run.

    Outputs:
        (Path, Path): The .pth path and the sidecar path.
    """
    if phase not in ("adam", "lbfgs"):
        raise ValueError(f"phase must be 'adam' or 'lbfgs', got {phase!r}")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")

    pth_path = Path(path).with_suffix(".pth")
    sidecar_path = pth_path.with_suffix(".json")

    sidecar = {
        "format_version": SIDECAR_VERSION,
        "case_id": case["case_id"],
        "checkpoint": pth_path.stem,
        "phase": phase,
        "adam_steps": int(adam_steps),
        "lbfgs_evals": int(lbfgs_evals),
        "status": status,
        "metrics": metrics,
        "losses": losses,
        "k": case["k"],
        "Re": case.get("Re"),
        "U0": case.get("U0"),
        "phi_x": case.get("phi_x"),
        "phi_y": case.get("phi_y"),
        "split": case.get("split"),
        "seed_points": case.get("seed_points"),
        "seed_train": case.get("seed_train"),
        "config": config,
        "architecture": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    assert tuple(sidecar) == SIDECAR_FIELDS

    state_dict = OrderedDict((name, tensor.detach().cpu()) for name, tensor in model.state_dict().items())

    def write_sidecar(tmp):
        with open(tmp, "w") as f:
            json.dump(sidecar, f, indent=2)

    _atomic_write(sidecar_path, write_sidecar)
    _atomic_write(pth_path, lambda tmp: torch.save(state_dict, tmp))
    return pth_path, sidecar_path


def _find_case(case_id: str, metadata_paths):
    """Returns the case dict with this case_id from the first case file that has it, or None."""
    for metadata_path in metadata_paths:
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r") as f:
            dataset = json.load(f)
        for split in SPLITS:
            for case in dataset.get(split, []):
                if case["case_id"] == case_id:
                    return {"split": split, **case}
    return None


def _legacy_info(pth_path: Path, k, metadata_paths) -> dict:
    """
    Builds a sidecar-shaped dict for an old models/case_XX.pth, which has no
    sidecar. The case parameters come from the case files when the file name
    is a known case_id.
    """
    info = dict.fromkeys(SIDECAR_FIELDS)
    info["case_id"] = pth_path.stem
    info["checkpoint"] = "legacy"
    info["architecture"] = BaselinePINN.__name__
    info["dtype"] = "float64"

    case = _find_case(pth_path.stem, metadata_paths)
    if case is not None:
        for field in ("k", "Re", "U0", "phi_x", "phi_y", "split", "seed_points", "seed_train"):
            info[field] = case.get(field)
    if k is not None:
        info["k"] = k
    if info["k"] is None:
        searched = ", ".join(str(p) for p in metadata_paths) or "none given"
        raise ValueError(
            f"{pth_path} has no sidecar and its case_id '{pth_path.stem}' was not found in the case "
            f"files ({searched}), so its wave number k is unknown. Pass k=..., or regenerate "
            f"data/cases_metadata.json with `python src/data/sampler.py`."
        )
    return info


def load_checkpoint(path, device: str = "cpu", k=None, metadata_paths=DEFAULT_METADATA_PATHS):
    """
    Loads a model saved by save_checkpoint(), or an old models/case_XX.pth.

    k (the wave number BaselinePINN needs) is read from the sidecar. An old
    checkpoint has no sidecar, so k is taken from the `k` argument or looked
    up by case_id (the file name) in the case files.

    Inputs:
        path (str | Path): The .pth file.
        device (str): Device to put the model on. Defaults to "cpu".
        k (float | None): Wave number; only needed for an old checkpoint whose
                          case is not in the case files.
        metadata_paths (iterable of Path): Case files to look an old checkpoint's
                          case_id up in. Defaults to data/cases_metadata.json and
                          cases/confirmation_cases.json.

    Outputs:
        model (BaselinePINN): The model in eval mode, float64, on `device`.
        info (dict): The sidecar (for an old checkpoint, a dict with the same
                     keys, checkpoint="legacy" and the case parameters when found).
    """
    pth_path = Path(path)
    sidecar_path = pth_path.with_suffix(".json")

    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            info = json.load(f)
        if k is not None and k != info["k"]:
            raise ValueError(f"k={k} was passed but the sidecar of {pth_path} says k={info['k']}")
    else:
        info = _legacy_info(pth_path, k, list(metadata_paths))

    model = BaselinePINN(k=info["k"])
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model, info
