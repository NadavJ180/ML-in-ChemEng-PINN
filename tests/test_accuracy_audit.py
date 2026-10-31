"""
Unit tests for the accuracy audit (src/models/accuracy_audit.py, #37).

Verifies that:
  - The audit grid is PROTOCOL.md section 6's 64 x 64 x 11 grid, and not the
    acceptance test's eval_grid.
  - A "model" that returns the exact solution has an error of 0, a known error
    comes out exactly, the pressure is compared without its mean at each time,
    and relL2_uv_T looks at t = T only.
  - A model is inaccurate only when relL2_uv is above 0.10.
  - Both model layouts are read (one folder of 7 checkpoints per case, and the old
    case_XX.pth files), with the stage, Adam step and run result of each model,
    and a missing checkpoint, a missing case or a checkpoint of another case is refused.
  - The committed results/accuracy/ CSVs have one row per model, follow from the
    committed models and case files, and match the old package for the old models.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.point_samplers import generate_evaluation_grid
from src.data.sampler import generate_randomized_cases, split_dataset
from src.models import accuracy_audit
from src.models.accuracy_audit import (CSV_COLUMNS, accuracy_grid, audit, case_checkpoints, main, read_cases,
                                       reference_errors)
from src.models.checkpoints import ADAM_SAVE_STEPS, FINAL_STAGE, STAGES, load_checkpoint, save_checkpoint
from src.models.pinn import BaselinePINN
from src.physics.taylor_green import compute_nu, compute_T, generate_tgv

REPO = Path(__file__).parent.parent
CONFIRMATION_CASES = REPO / "cases" / "confirmation_cases.json"
CONFIRMATION_MODELS = REPO / "models" / "confirmation"
CONFIRMATION_CSV = REPO / "results" / "accuracy" / "reference_errors_confirmation.csv"
EXISTING_CSV = REPO / "results" / "accuracy" / "reference_errors_existing.csv"
ERRORS = ("relL2_uv", "relL2_p", "relL2_uv_T")

CASE = {
    "case_id": "case_30", "split": "confirmation", "k": 2, "U0": 1.25, "Re": 120.5,
    "phi_x": 0.5, "phi_y": 1.5, "seed_points": 3030, "seed_train": 4030, "seed_scoring": 1030,
}

# The old package's 04_checkpoints/checkpoints.csv (rel_L2_velocity, rel_L2_pressure,
# rel_L2_velocity_final_time), for one old case of each k
OLD_PACKAGE = {
    "case_00": (0.0015942185981635, 0.0046466208914462, 0.002527425915233),   # k = 1
    "case_02": (0.0042288539773768, 0.0120208945643179, 0.0087999189908749),  # k = 2
    "case_01": (0.0048902353582143, 0.0087156928720022, 0.0093722676728392),  # k = 3
}


class ExactModel(torch.nn.Module):
    """
    Returns the exact solution of a case, optionally changed: the velocity times
    1 + velocity_error(t), and the pressure times pressure_scale plus pressure_offset(t).
    """

    def __init__(self, case, velocity_error=lambda t: 0.0, pressure_scale=1.0, pressure_offset=lambda t: 0.0):
        super().__init__()
        self.case = case
        self.velocity_error = velocity_error
        self.pressure_scale = pressure_scale
        self.pressure_offset = pressure_offset

    def forward(self, coords):
        c = self.case
        x, y, t = coords[:, 0], coords[:, 1], coords[:, 2]
        u, v, p = generate_tgv(x, y, t, c["U0"], c["k"], c["phi_x"], c["phi_y"], compute_nu(c["U0"], c["Re"], c["k"]))
        factor = 1 + self.velocity_error(t)
        return torch.stack([u * factor, v * factor, p * self.pressure_scale + self.pressure_offset(t)], dim=1)


def read_csv(path):
    """Reads an audit CSV back exactly."""
    return pd.read_csv(path, float_precision="round_trip")


def write_case_file(path, cases, split="confirmation"):
    """Writes a case file with `cases` under `split`, and returns its path."""
    path.write_text(json.dumps({"metadata": {}, split: cases}))
    return path


def write_run(models_dir, case, status="PASSED"):
    """Saves the 7 checkpoints of a made-up run of `case`, as train_model.py --models_dir does."""
    torch.manual_seed(0)
    case_dir = models_dir / case["case_id"]
    for step, stage in zip(ADAM_SAVE_STEPS, STAGES):
        save_checkpoint(BaselinePINN(k=case["k"]), case_dir / stage, case, phase="adam", adam_steps=step)
    save_checkpoint(BaselinePINN(k=case["k"]), case_dir / FINAL_STAGE, case, phase="lbfgs",
                    adam_steps=ADAM_SAVE_STEPS[-1], lbfgs_evals=10, status=status)
    return case_dir


@pytest.fixture(scope="module")
def old_case_file(tmp_path_factory):
    """data/cases_metadata.json, which is not tracked, regenerated (it is deterministic, seed 42)."""
    path = tmp_path_factory.mktemp("cases") / "cases_metadata.json"
    path.write_text(json.dumps(split_dataset(generate_randomized_cases(num_cases=30, seed=42))))
    return path


# --- The grid -----------------------------------------------------------------

def test_grid_is_the_protocol_grid():
    T = 0.7
    t, x, y = accuracy_grid(T)

    assert t.shape == x.shape == y.shape == (11, 64, 64)
    assert t.numel() == 45_056
    assert all(g.dtype == torch.float64 for g in (t, x, y))
    # x and y are 2*pi*i/64 for i = 0 ... 63: 0 is in, 2*pi (the same point as 0) is not
    xy = torch.tensor([2 * math.pi * i / 64 for i in range(64)], dtype=torch.float64)
    assert torch.allclose(x[0, :, 0], xy, rtol=0, atol=1e-15)
    assert torch.allclose(y[0, 0, :], xy, rtol=0, atol=1e-15)
    assert x[0, 0, 0] == 0 and x.max() < 2 * math.pi
    # 11 evenly spaced times, both ends included
    times = t[:, 0, 0]
    assert times[0] == 0 and times[-1] == T
    assert torch.allclose(torch.diff(times), torch.full((10,), T / 10, dtype=torch.float64), rtol=1e-12)
    # Every point once
    points = torch.stack([g.reshape(-1) for g in (x, y, t)], dim=1)
    assert torch.unique(points, dim=0).shape[0] == 45_056


def test_grid_is_not_the_eval_grid():
    T = 0.7
    _, x, _ = accuracy_grid(T)
    eval_grid = generate_evaluation_grid(T)

    # The eval grid has 20 times and x values that reach 2*pi; the two share only x = 0
    assert eval_grid.shape[0] == 64 * 64 * 20
    eval_x = set(torch.unique(eval_grid[:, 0].to(torch.float64)).tolist())
    assert eval_x & set(x[0, :, 0].tolist()) == {0.0}


# --- The three errors ----------------------------------------------------------

def test_exact_model_has_zero_error():
    errors = reference_errors(ExactModel(CASE), CASE)
    assert set(errors) == set(ERRORS)
    for name in ERRORS:
        assert errors[name] == pytest.approx(0.0, abs=1e-14), name


def test_pressure_is_compared_without_its_mean_at_each_time():
    # A pressure that is right up to a different constant at each time has no pressure error
    errors = reference_errors(ExactModel(CASE, pressure_offset=lambda t: 3.0 + 5.0 * t), CASE)
    assert errors["relL2_p"] == pytest.approx(0.0, abs=1e-12)
    assert errors["relL2_uv"] == pytest.approx(0.0, abs=1e-14)


def test_known_errors_come_out_exactly():
    errors = reference_errors(ExactModel(CASE, velocity_error=lambda t: 0.05, pressure_scale=1.2), CASE)
    assert errors["relL2_uv"] == pytest.approx(0.05, rel=1e-12)
    assert errors["relL2_uv_T"] == pytest.approx(0.05, rel=1e-12)
    assert errors["relL2_p"] == pytest.approx(0.2, rel=1e-12)


def test_final_time_error_is_taken_at_t_equal_T():
    # The velocity error grows from 0 at t = 0 to 30% at t = T
    T = compute_T(U0=CASE["U0"], Re=CASE["Re"], k=CASE["k"])
    errors = reference_errors(ExactModel(CASE, velocity_error=lambda t: 0.3 * t / T), CASE)
    assert errors["relL2_uv_T"] == pytest.approx(0.3, rel=1e-12)
    assert 0 < errors["relL2_uv"] < 0.3


def test_inaccurate_means_strictly_above_ten_percent(tmp_path, monkeypatch):
    case_file = write_case_file(tmp_path / "cases.json", [CASE])
    write_run(tmp_path / "models", CASE)
    errors = iter([0.05, 0.10, 0.1000001, 0.2, 0.0999999, 0.5, 0.01])
    monkeypatch.setattr(accuracy_audit, "reference_errors",
                        lambda model, case, device: dict.fromkeys(ERRORS, next(errors)))

    df = audit(case_file, tmp_path / "models")
    assert list(df["inaccurate"]) == [False, False, True, True, False, True, False]


# --- Reading the models ----------------------------------------------------------

@pytest.mark.parametrize("status", ["PASSED", "FAILED"])
def test_reads_a_run_folder(tmp_path, status):
    case_file = write_case_file(tmp_path / "cases.json", [CASE])
    case_dir = write_run(tmp_path / "models", CASE, status=status)
    out = tmp_path / "results" / "errors.csv"

    main(["--metadata", str(case_file), "--models_dir", str(tmp_path / "models"), "--out", str(out)])
    df = read_csv(out)

    assert list(df.columns) == list(CSV_COLUMNS)
    assert list(df["stage"]) == list(STAGES)
    assert list(df["adam_step"]) == [100, 500, 1000, 2500, 4000, 5000, 5000]
    assert (df["case_id"] == "case_30").all() and (df["k"] == 2).all() and (df["split"] == "confirmation").all()
    # The run's result is on every stage, so a failed run is visible at each of them
    assert (df["passed_acceptance"] == (status == "PASSED")).all()
    assert (df["inaccurate"] == (df["relL2_uv"] > 0.10)).all()
    for stage, row in zip(STAGES, df.itertuples()):
        model, _ = load_checkpoint(case_dir / f"{stage}.pth")
        assert reference_errors(model, CASE) == {name: getattr(row, name) for name in ERRORS}
    # The CSV has LF line endings on every platform
    assert b"\r" not in out.read_bytes()


def test_reads_old_models(tmp_path):
    old_case = {"case_id": "case_07", "seed": 49, "U0": 1.1, "k": 1, "Re": 40.0, "phi_x": 0.3, "phi_y": 2.0}
    case_file = write_case_file(tmp_path / "cases_metadata.json", [old_case], split="test")
    (tmp_path / "models").mkdir()
    model = BaselinePINN(k=1)
    torch.save(model.state_dict(), tmp_path / "models" / "case_07.pth")

    df = audit(case_file, tmp_path / "models")

    assert len(df) == 1
    row = df.iloc[0]
    assert (row["case_id"], row["k"], row["split"], row["stage"]) == ("case_07", 1, "test", "legacy")
    # An old model has no sidecar, so its Adam step and run result are unknown
    assert pd.isna(row["adam_step"]) and pd.isna(row["passed_acceptance"])
    assert {name: row[name] for name in ERRORS} == reference_errors(model, {**old_case, "split": "test"})


def test_refuses_a_missing_checkpoint(tmp_path):
    case_file = write_case_file(tmp_path / "cases.json", [CASE])
    case_dir = write_run(tmp_path / "models", CASE)
    (case_dir / "adam_02500.pth").unlink()
    with pytest.raises(FileNotFoundError, match="adam_02500.pth"):
        audit(case_file, tmp_path / "models")


def test_refuses_a_case_without_models(tmp_path):
    other = {**CASE, "case_id": "case_31", "seed_points": 3031, "seed_train": 4031, "seed_scoring": 1031}
    case_file = write_case_file(tmp_path / "cases.json", [CASE, other])
    write_run(tmp_path / "models", CASE)
    with pytest.raises(FileNotFoundError, match="case_31"):
        audit(case_file, tmp_path / "models")


def test_refuses_a_checkpoint_of_another_case(tmp_path):
    write_run(tmp_path / "models", CASE)
    # The case file describes another flow than the one the models were trained on
    case_file = write_case_file(tmp_path / "cases.json", [{**CASE, "U0": 1.5}])
    with pytest.raises(ValueError, match="does not match its case"):
        audit(case_file, tmp_path / "models")


def test_refuses_a_checkpoint_under_another_stage_name(tmp_path):
    case_file = write_case_file(tmp_path / "cases.json", [CASE])
    case_dir = write_run(tmp_path / "models", CASE)
    # adam_00100 is a copy of adam_00500
    for suffix in (".pth", ".json"):
        (case_dir / f"adam_00100{suffix}").write_bytes((case_dir / f"adam_00500{suffix}").read_bytes())
    with pytest.raises(ValueError, match="adam_00500"):
        audit(case_file, tmp_path / "models")


def test_case_checkpoints_are_in_training_order(tmp_path):
    case_dir = write_run(tmp_path, CASE)
    assert case_checkpoints("case_30", tmp_path) == [(stage, case_dir / f"{stage}.pth") for stage in STAGES]


# --- The committed results -------------------------------------------------------

def test_committed_confirmation_errors_cover_every_checkpoint():
    df = read_csv(CONFIRMATION_CSV)
    cases = read_cases(CONFIRMATION_CASES)

    assert list(df.columns) == list(CSV_COLUMNS)
    assert len(df) == 210
    assert list(zip(df["case_id"], df["stage"])) == [(c["case_id"], s) for c in cases for s in STAGES]
    assert list(df["k"]) == [c["k"] for c in cases for _ in STAGES]
    assert (df["split"] == "confirmation").all()
    assert (df["inaccurate"] == (df["relL2_uv"] > 0.10)).all()
    for row in df.itertuples():
        sidecar = json.loads((CONFIRMATION_MODELS / row.case_id / f"{row.stage}.json").read_text())
        final = json.loads((CONFIRMATION_MODELS / row.case_id / f"{FINAL_STAGE}.json").read_text())
        assert row.adam_step == sidecar["adam_steps"]
        assert row.passed_acceptance == (final["status"] == "PASSED")


@pytest.mark.parametrize("case_id", ["case_30", "case_40", "case_50"])  # One case of each k
def test_committed_confirmation_errors_follow_from_the_models(case_id):
    df = read_csv(CONFIRMATION_CSV).set_index(["case_id", "stage"])
    case = next(c for c in read_cases(CONFIRMATION_CASES) if c["case_id"] == case_id)
    for stage, path in case_checkpoints(case_id, CONFIRMATION_MODELS):
        model, _ = load_checkpoint(path, k=case["k"])
        errors = reference_errors(model, case)
        for name in ERRORS:
            assert errors[name] == pytest.approx(df.loc[(case_id, stage), name], rel=1e-10), (stage, name)


def test_committed_old_errors_follow_from_the_models(old_case_file):
    committed = read_csv(EXISTING_CSV)
    recomputed = audit(old_case_file, REPO / "models")

    assert list(committed.columns) == list(CSV_COLUMNS)
    assert len(committed) == 30
    for column in ("case_id", "k", "split", "stage", "inaccurate"):
        assert list(committed[column]) == list(recomputed[column]), column
    assert (committed["stage"] == "legacy").all()
    assert committed["adam_step"].isna().all() and committed["passed_acceptance"].isna().all()
    for name in ERRORS:
        assert list(committed[name]) == pytest.approx(list(recomputed[name]), rel=1e-10), name


def test_committed_old_errors_match_the_old_package():
    committed = read_csv(EXISTING_CSV).set_index("case_id")
    for case_id, values in OLD_PACKAGE.items():
        for name, value in zip(ERRORS, values):
            assert committed.loc[case_id, name] == pytest.approx(value, abs=1e-13), (case_id, name)
    # The old package's range, and every old model is accurate
    assert 0.0013 < committed["relL2_uv"].min() and committed["relL2_uv"].max() < 0.0056
    assert not committed["inaccurate"].any()
