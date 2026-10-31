"""
Checkpoint helper: save a model with a small JSON sidecar, and load it back.

A checkpoint is two files that share a name:
    <name>.pth   The model's state_dict, in the same format as the old
                 models/case_XX.pth files, so torch.load + load_state_dict
                 still opens it.
    <name>.json  The sidecar: which case it is, which phase and step it was
                 saved at, the seeds, the training settings, the model's
                 acceptance metrics on the case's eval grid and, for
                 lbfgs_final, PASSED/FAILED.

load_checkpoint() opens both these checkpoints and the old models/case_XX.pth
files (no sidecar), so later stages can treat old and new models alike.
load_final_model() finds a case's final model in a models folder of either
layout, and refuses a missing model or a sidecar of another case.
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

# The Adam steps a run saves by default (train_model.py --save_steps), and the
# checkpoint names of one case folder in that run, in training order.
ADAM_SAVE_STEPS = (100, 500, 1000, 2500, 4000, 5000)
FINAL_STAGE = "lbfgs_final"


def adam_stage(step: int) -> str:
    """Returns the checkpoint name of the model saved right after Adam step `step`, e.g. "adam_00100"."""
    return f"adam_{step:05d}"


STAGES = tuple(adam_stage(step) for step in ADAM_SAVE_STEPS) + (FINAL_STAGE,)

# The checkpoint name load_checkpoint() gives an old models/case_XX.pth, which has no sidecar
LEGACY_CHECKPOINT = "legacy"

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
    "metrics",         # {"rel_l2", "mse_rc", "mse_rm", "passed"} of this model on the case's eval grid
    "losses",          # Loss components of the last training step (its mini-batch, evaluated before
                       # that step's update); use "metrics" for the model's accuracy
    "k",
    "Re",
    "U0",
    "phi_x",
    "phi_y",
    "split",
    "seed_points",     # Seed of the case's tensors (null for the original 30 cases)
    "seed_train",      # Seed of weight init + mini-batches (null for the original 30 cases)
    "seed_scoring",    # Seed of the case's PHS scoring points (null for the original 30 cases)
    "config",          # Training settings of the run (epochs, L-BFGS limits, batch sizes, device)
    "architecture",
    "dtype",
    "torch_version",
    "git_commit",      # Full hash of the checked-out commit
    "git_dirty",       # True if tracked files had uncommitted changes (untracked outputs don't count)
    "created_utc",
    "wall_time_s",     # Seconds since the case's training started
)

STATUSES = (None, "PASSED", "FAILED")

# Sidecar entries that must equal the case file's, so a checkpoint is never used as another case
CASE_FIELDS = ("case_id", "k", "Re", "U0", "phi_x", "phi_y", "split", "seed_points", "seed_train", "seed_scoring")

# Case files that list cases by split; used to find k for an old checkpoint.
DEFAULT_METADATA_PATHS = (
    PROJECT_ROOT / "data" / "cases_metadata.json",
    PROJECT_ROOT / "cases" / "confirmation_cases.json",
)
SPLITS = ("train", "validation", "test", "confirmation")


def _git(*args):
    """Runs a git command in the project root; returns its stdout, or None outside a git checkout."""
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_state():
    """
    Returns (commit, dirty): the full hash of the checked-out commit, and whether
    tracked files have uncommitted changes. Untracked files are not counted, since
    a training run writes its own untracked outputs. (None, None) outside a git checkout.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if commit is None or status is None:
        return None, None
    return commit.strip(), bool(status.strip())


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
                    status=None, metrics=None, losses=None, config=None, wall_time_s=None):
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
        metrics (dict | None): Acceptance metrics of this model on the case's eval grid.
        losses (dict | None): Loss components of the last training step.
        config (dict | None): Training settings of the run.
        wall_time_s (float | None): Seconds since the case's training started.

    Outputs:
        (Path, Path): The .pth path and the sidecar path.
    """
    if phase not in ("adam", "lbfgs"):
        raise ValueError(f"phase must be 'adam' or 'lbfgs', got {phase!r}")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")

    pth_path = Path(path).with_suffix(".pth")
    sidecar_path = pth_path.with_suffix(".json")
    git_commit, git_dirty = _git_state()

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
        "seed_scoring": case.get("seed_scoring"),
        "config": config,
        "architecture": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_s": None if wall_time_s is None else float(wall_time_s),
    }
    assert tuple(sidecar) == SIDECAR_FIELDS

    state_dict = OrderedDict((name, tensor.detach().cpu()) for name, tensor in model.state_dict().items())

    def write_sidecar(tmp):
        # LF on every platform, so a sidecar has the same bytes wherever it was written
        with open(tmp, "w", newline="\n") as f:
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
    info["checkpoint"] = LEGACY_CHECKPOINT
    info["architecture"] = BaselinePINN.__name__
    info["dtype"] = "float64"

    case = _find_case(pth_path.stem, metadata_paths)
    if case is not None:
        for field in ("k", "Re", "U0", "phi_x", "phi_y", "split", "seed_points", "seed_train", "seed_scoring"):
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


def check_sidecar(info: dict, case: dict, checkpoint: str, path) -> None:
    """
    Refuses a model that is not checkpoint `checkpoint` of `case`: raises
    ValueError if its sidecar names another checkpoint, or if any of its
    CASE_FIELDS differs from the case file. An old model has no sidecar, so
    only its name ("legacy") is checked.

    Inputs:
        info (dict): The sidecar, as load_checkpoint() returns it.
        case (dict): The case's entry in the case file.
        checkpoint (str): The expected name, e.g. "lbfgs_final" or "legacy".
        path (str | Path): The model file, for the error message.
    """
    if info["checkpoint"] != checkpoint:
        raise ValueError(f"{path}: its sidecar says checkpoint {info['checkpoint']!r}, expected {checkpoint!r}")
    if checkpoint == LEGACY_CHECKPOINT:
        return
    wrong = {key: (info.get(key), case.get(key)) for key in CASE_FIELDS if info.get(key) != case.get(key)}
    if wrong:
        raise ValueError(f"{path} does not match its case in the case file ((sidecar, case file) values: {wrong})")


def final_model_path(case_id: str, models_dir):
    """
    Finds the final model of a case in a models folder of either layout.

    Inputs:
        case_id (str): The case, e.g. "case_30".
        models_dir (str | Path): A folder with one folder of checkpoints per case
                                 (models/confirmation), or with the old case_XX.pth
                                 files (models).

    Outputs:
        (str, Path): "lbfgs_final" and <models_dir>/<case_id>/lbfgs_final.pth when the
                     case has a folder, else "legacy" and <models_dir>/<case_id>.pth:
                     the (name, path) pairs of accuracy_audit.case_checkpoints().
                     Raises FileNotFoundError if that file does not exist, so a case
                     with no model stops a run instead of being skipped (PROTOCOL.md
                     section 1: no case is dropped).
    """
    models_dir = Path(models_dir)
    case_dir = models_dir / case_id
    if case_dir.is_dir():
        checkpoint, path = FINAL_STAGE, case_dir / f"{FINAL_STAGE}.pth"
    else:
        checkpoint, path = LEGACY_CHECKPOINT, models_dir / f"{case_id}.pth"
    if not path.exists():
        if checkpoint == LEGACY_CHECKPOINT:
            raise FileNotFoundError(f"No final model for {case_id}: neither {case_dir}/ nor {path} exists")
        raise FileNotFoundError(f"No final model for {case_id}: {path} does not exist")
    return checkpoint, path


def load_final_model(case: dict, models_dir, device: str = "cpu"):
    """
    Loads the final model of a case (see final_model_path), and refuses it if its
    sidecar describes another case (see check_sidecar).

    Inputs:
        case (dict): The case's entry in the case file (case_id, k, Re, ...).
        models_dir (str | Path): The models folder, of either layout.
        device (str): Device to put the model on. Defaults to "cpu".

    Outputs:
        model, info: As load_checkpoint() returns them.
    """
    checkpoint, path = final_model_path(case["case_id"], models_dir)
    model, info = load_checkpoint(path, device=device, k=case["k"])
    check_sidecar(info, case, checkpoint, path)
    return model, info
