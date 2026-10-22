"""
Unit tests for the checkpoint helper (src/models/checkpoints.py).

Verifies that:
  - A saved model loads back with identical weights and identical outputs.
  - The sidecar has every field, with the values that describe the model.
  - The final checkpoint's sidecar records PASSED or FAILED and the metrics.
  - load_checkpoint() also opens the old sidecar-less models/case_XX.pth files,
    including the 30 committed ones.
  - Saving checkpoints during Adam does not change training, and a checkpoint
    saved after step N holds the weights after exactly N steps.
"""

import argparse
import json

import pytest
import torch
from pathlib import Path

from src.data.sampler import generate_randomized_cases
from src.models.checkpoints import SIDECAR_FIELDS, load_checkpoint, save_checkpoint
from src.models.pinn import BaselinePINN
from src.models.train_model import train_adam

PROJECT_ROOT = Path(__file__).parent.parent

CASE = {
    "case_id": "case_30", "split": "confirmation", "k": 2, "U0": 1.25, "Re": 120.5,
    "phi_x": 0.5, "phi_y": 1.5, "seed_points": 3030, "seed_train": 4030, "seed_scoring": 1030,
}

# Written out in full so that changing the sidecar is a deliberate act: later steps read these names.
EXPECTED_FIELDS = [
    "format_version", "case_id", "checkpoint", "phase", "adam_steps", "lbfgs_evals",
    "status", "metrics", "losses", "k", "Re", "U0", "phi_x", "phi_y", "split",
    "seed_points", "seed_train", "config", "architecture", "dtype", "torch_version",
    "git_commit", "created_utc",
]


@pytest.fixture
def model():
    """A freshly initialized float64 BaselinePINN for CASE."""
    torch.manual_seed(0)
    return BaselinePINN(k=CASE["k"])


@pytest.fixture
def coords():
    """256 random (x, y, t) points in the domain."""
    generator = torch.Generator().manual_seed(1)
    return torch.rand(256, 3, dtype=torch.float64, generator=generator) * 2 * torch.pi


def outputs(model, coords):
    model.eval()
    with torch.no_grad():
        return model(coords)


def assert_same_weights(model_a, model_b):
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert list(state_a) == list(state_b)
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), f"weights differ in {name}"


def test_round_trip_outputs_identical(tmp_path, model, coords):
    pth_path, sidecar_path = save_checkpoint(model, tmp_path / "case_30" / "adam_00100.pth", CASE,
                                             phase="adam", adam_steps=100)

    assert pth_path == tmp_path / "case_30" / "adam_00100.pth"
    assert sidecar_path == tmp_path / "case_30" / "adam_00100.json"
    # Nothing but the two files: no temporary files are left behind
    assert sorted(p.name for p in pth_path.parent.iterdir()) == ["adam_00100.json", "adam_00100.pth"]

    loaded, info = load_checkpoint(pth_path)

    assert isinstance(loaded, BaselinePINN)
    assert loaded.k == CASE["k"]
    assert not loaded.training
    assert all(p.dtype == torch.float64 for p in loaded.parameters())
    assert_same_weights(model, loaded)
    assert torch.equal(outputs(model, coords), outputs(loaded, coords))
    assert info == json.loads(sidecar_path.read_text())


def test_pth_is_a_plain_state_dict(tmp_path, model, coords):
    """New checkpoints still open the old way (torch.load + load_state_dict)."""
    pth_path, _ = save_checkpoint(model, tmp_path / "adam_00100.pth", CASE, phase="adam", adam_steps=100)

    plain = BaselinePINN(k=CASE["k"])
    plain.load_state_dict(torch.load(pth_path, map_location="cpu"))

    assert torch.equal(outputs(model, coords), outputs(plain, coords))


def test_sidecar_has_every_field(tmp_path, model):
    losses = {"Total": 0.5, "L_NS": 0.1, "L_div": 0.01, "L_IC": 0.02, "L_BC": 0.03, "L_p": 0.0, "L_NS_raw": 0.2}
    config = {"adam_epochs": 5000, "lbfgs_iters": 5000, "n_int": 4000, "device": "cpu"}
    _, sidecar_path = save_checkpoint(model, tmp_path / "adam_00500", CASE, phase="adam", adam_steps=500,
                                      losses=losses, config=config)
    sidecar = json.loads(sidecar_path.read_text())

    assert list(SIDECAR_FIELDS) == EXPECTED_FIELDS
    assert list(sidecar) == EXPECTED_FIELDS

    assert sidecar["format_version"] == 1
    assert sidecar["case_id"] == "case_30"
    assert sidecar["checkpoint"] == "adam_00500"
    assert sidecar["phase"] == "adam"
    assert sidecar["adam_steps"] == 500
    assert sidecar["lbfgs_evals"] == 0
    assert sidecar["status"] is None
    assert sidecar["metrics"] is None
    assert sidecar["losses"] == losses
    for field in ("k", "Re", "U0", "phi_x", "phi_y", "split", "seed_points", "seed_train"):
        assert sidecar[field] == CASE[field], field
    assert sidecar["config"] == config
    assert sidecar["architecture"] == "BaselinePINN"
    assert sidecar["dtype"] == "float64"
    assert sidecar["torch_version"] == torch.__version__
    assert sidecar["created_utc"].endswith("Z")


@pytest.mark.parametrize("status", ["PASSED", "FAILED"])
def test_final_checkpoint_records_status(tmp_path, model, coords, status):
    metrics = {"rel_l2": 0.05, "mse_rc": 1e-5, "mse_rm": 2e-4}
    pth_path, sidecar_path = save_checkpoint(model, tmp_path / "lbfgs_final.pth", CASE, phase="lbfgs",
                                             adam_steps=5000, lbfgs_evals=812, status=status, metrics=metrics)
    sidecar = json.loads(sidecar_path.read_text())

    assert list(sidecar) == EXPECTED_FIELDS
    assert sidecar["checkpoint"] == "lbfgs_final"
    assert sidecar["phase"] == "lbfgs"
    assert (sidecar["adam_steps"], sidecar["lbfgs_evals"]) == (5000, 812)
    assert sidecar["status"] == status
    assert sidecar["metrics"] == metrics

    loaded, _ = load_checkpoint(pth_path)
    assert torch.equal(outputs(model, coords), outputs(loaded, coords))


def test_rejects_bad_status_and_phase(tmp_path, model):
    with pytest.raises(ValueError):
        save_checkpoint(model, tmp_path / "lbfgs_final.pth", CASE, phase="lbfgs", adam_steps=5000, status="OK")
    with pytest.raises(ValueError):
        save_checkpoint(model, tmp_path / "x.pth", CASE, phase="sgd", adam_steps=1)
    assert list(tmp_path.iterdir()) == []


def test_loads_old_style_checkpoint(tmp_path, model, coords):
    """An old models/case_XX.pth is a bare state_dict with no sidecar."""
    old_path = tmp_path / "case_07.pth"
    torch.save(model.state_dict(), old_path)

    # k unknown: no sidecar, not in any case file, not passed
    with pytest.raises(ValueError, match="k"):
        load_checkpoint(old_path, metadata_paths=[])

    # k passed explicitly
    loaded, info = load_checkpoint(old_path, k=CASE["k"], metadata_paths=[])
    assert torch.equal(outputs(model, coords), outputs(loaded, coords))
    assert list(info) == EXPECTED_FIELDS
    assert (info["case_id"], info["checkpoint"], info["k"]) == ("case_07", "legacy", CASE["k"])

    # k looked up by case_id in a case file
    metadata_path = tmp_path / "cases_metadata.json"
    metadata_path.write_text(json.dumps({
        "metadata": {}, "train": [], "validation": [],
        "test": [{"case_id": "case_07", "seed": 49, "U0": 1.25, "k": CASE["k"], "Re": 120.5,
                  "phi_x": 0.5, "phi_y": 1.5}],
    }))
    loaded, info = load_checkpoint(old_path, metadata_paths=[metadata_path])
    assert torch.equal(outputs(model, coords), outputs(loaded, coords))
    assert (info["k"], info["Re"], info["split"], info["seed_train"]) == (CASE["k"], 120.5, "test", None)


def test_loads_committed_old_models(coords):
    """The 30 committed models/case_XX.pth open with load_checkpoint()."""
    # data/cases_metadata.json is not tracked, so the cases are regenerated (deterministic, seed 42)
    cases = generate_randomized_cases(num_cases=30, seed=42)
    loaded_any = False
    for case in cases:
        pth_path = PROJECT_ROOT / "models" / f"{case['case_id']}.pth"
        if not pth_path.exists():
            continue
        model, info = load_checkpoint(pth_path, k=case["k"])
        assert info["checkpoint"] == "legacy" and info["case_id"] == case["case_id"]
        assert torch.isfinite(outputs(model, coords)).all()
        loaded_any = True
    if not loaded_any:
        pytest.skip("no models/case_XX.pth in this checkout")


# --- Checkpointing must not change training ---------------------------------

def tiny_case_data(n=32):
    """A small synthetic dataset with the structure of a generate_datasets.py .pt file."""
    generator = torch.Generator().manual_seed(2)

    def points(t_zero=False):
        p = torch.rand(n, 3, dtype=torch.float64, generator=generator) * 2 * torch.pi
        if t_zero:
            p[:, 2] = 0.0
        return p

    return {
        "interior": points(),
        "ic": points(t_zero=True),
        "bc": {"x_bounds": (points(), points()), "y_bounds": (points(), points())},
        "ic_true": torch.rand(n, 2, dtype=torch.float64, generator=generator),
    }


def run_adam(adam_epochs, save_steps=(), checkpoint_fn=None):
    """Trains a fresh model with train_adam() from a fixed seed."""
    args = argparse.Namespace(adam_epochs=adam_epochs, n_int=16, n_ic=8, n_bc=8, device="cpu",
                              save_steps=list(save_steps))
    case_data = tiny_case_data()
    torch.manual_seed(4030)
    model = BaselinePINN(k=CASE["k"])
    model, _, history = train_adam(model, case_data, CASE, args, checkpoint_fn=checkpoint_fn)
    return model, history


def test_checkpoints_do_not_change_training(tmp_path, coords):
    saved_steps = []

    def checkpoint_fn(step, model, metrics):
        saved_steps.append(step)
        save_checkpoint(model, tmp_path / f"adam_{step:05d}.pth", CASE, phase="adam", adam_steps=step,
                        losses=dict(metrics))

    with_saving, history_saving = run_adam(4, save_steps=[2, 4, 99], checkpoint_fn=checkpoint_fn)
    without_saving, history_plain = run_adam(4)

    assert saved_steps == [2, 4]
    assert_same_weights(with_saving, without_saving)
    assert history_saving == history_plain

    # adam_00002 holds the weights after exactly 2 steps
    after_two, _ = run_adam(2)
    checkpoint_two, _ = load_checkpoint(tmp_path / "adam_00002.pth")
    assert_same_weights(after_two, checkpoint_two)

    # adam_00004 is the model train_adam returned
    checkpoint_four, info = load_checkpoint(tmp_path / "adam_00004.pth")
    assert_same_weights(with_saving, checkpoint_four)
    assert info["losses"]["Total"] == history_saving["Total"][3]
