"""
Unit tests for how the two scripts of the perturbation benchmark (#35),
src/hallucinations/generate_hallucinations.py and src/detection/evaluate_phs.py,
find their cases and models.

Verifies that:
  - load_case_metadata() reads both case files: the 30 old cases with their
    train/validation/test splits, as before, and the 30 new cases of
    cases/confirmation_cases.json.
  - Both scripts load each case's model from the models folder they are given,
    in either layout: the old case_XX.pth files, or the lbfgs_final.pth of each
    case folder (models/confirmation).
  - A missing model, or a case missing from the case file, stops a run instead
    of being skipped, and evaluate_phs.py stops before scoring any field.
  - Every field of a case is scored at the case's seed_scoring.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from src.data.sampler import generate_randomized_cases, split_dataset
from src.detection import evaluate_phs
from src.hallucinations.generate_hallucinations import (generate_case_hallucinations, load_case_metadata,
                                                        run_model_in_chunks)
from src.hallucinations.perturbations import EPSILON_VALUES, PERTURBATION_NAMES
from src.models.checkpoints import load_checkpoint

REPO = Path(__file__).parent.parent
CONFIRMATION_CASES = REPO / "cases" / "confirmation_cases.json"
CONFIRMATION_MODELS = REPO / "models" / "confirmation"
OLD_MODELS = REPO / "models"


def old_case_file(path):
    """Writes data/cases_metadata.json, which is not tracked, regenerated (it is deterministic, seed 42)."""
    dataset = split_dataset(generate_randomized_cases(num_cases=30, seed=42))
    path.write_text(json.dumps(dataset))
    return dataset


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """The 30 old and the 30 new cases, as load_case_metadata() reads them."""
    old_path = tmp_path_factory.mktemp("cases") / "cases_metadata.json"
    old_case_file(old_path)
    return {**load_case_metadata(old_path), **load_case_metadata(CONFIRMATION_CASES)}


@pytest.fixture
def args(monkeypatch):
    """evaluate_phs.py's default settings, on the CPU."""
    monkeypatch.setattr(sys, "argv", ["evaluate_phs.py", "--device", "cpu"])
    return evaluate_phs.parse_args()


@pytest.fixture
def scored(monkeypatch):
    """Replaces the PHS computation with a stub; returns the list of (case_id, model, seed) it was called with."""
    calls = []

    def stub(model, case_meta, *args, seed, **kwargs):
        calls.append((case_meta["case_id"], model, seed))
        return {"mom": 1.0, "div": 1.0, "bc": 1.0, "E": 1.0}

    monkeypatch.setattr(evaluate_phs, "compute_phs_components", stub)
    return calls


def index_rows(case_ids):
    """Two hallucination-index rows per case: its clean field and one damaged field."""
    return [{"case_id": case_id, "perturbation_type": kind, "epsilon": epsilon, "label": label}
            for case_id in case_ids
            for kind, epsilon, label in [("none", 0.0, "clean"), ("boundary", 0.1, "hallucinated")]]


def assert_same_weights(model, path, k):
    expected, _ = load_checkpoint(path, k=k)
    state, expected_state = model.state_dict(), expected.state_dict()
    assert list(state) == list(expected_state)
    assert all(torch.equal(state[name], expected_state[name]) for name in state)


# --- The case files -------------------------------------------------------------

def test_reads_the_new_case_file():
    with open(CONFIRMATION_CASES) as f:
        listed = json.load(f)["confirmation"]
    cases = load_case_metadata(CONFIRMATION_CASES)
    assert list(cases) == [case["case_id"] for case in listed] == [f"case_{n}" for n in range(30, 60)]
    assert all(cases[case["case_id"]] == {**case, "split": "confirmation"} for case in listed)


def test_reads_the_old_case_file_as_before(tmp_path):
    path = tmp_path / "cases_metadata.json"
    dataset = old_case_file(path)
    # What it returned before it also read the confirmation list
    before = {case["case_id"]: {**case, "split": split}
              for split in ("train", "validation", "test") for case in dataset[split]}
    cases = load_case_metadata(path)
    assert cases == before and list(cases) == list(before)
    assert [case["split"] for case in cases.values()].count("validation") == 5


def test_refuses_a_case_file_without_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"metadata": {}}))
    with pytest.raises(ValueError, match="No cases"):
        load_case_metadata(path)


# --- generate_hallucinations.py -----------------------------------------------

@pytest.mark.parametrize("case_id, models_dir, model_file", [
    ("case_30", CONFIRMATION_MODELS, CONFIRMATION_MODELS / "case_30" / "lbfgs_final.pth"),  # One folder per case
    ("case_00", OLD_MODELS, OLD_MODELS / "case_00.pth"),                                     # The old layout
])
def test_hallucinations_use_the_model_of_the_models_folder(cases, tmp_path, case_id, models_dir, model_file):
    case = cases[case_id]
    # tmp_path has no tensor file, so the eval grid is rebuilt
    bundle, rows = generate_case_hallucinations(case_id, case, models_dir, tmp_path, "cpu", chunk_size=8000)

    expected, _ = load_checkpoint(model_file, k=case["k"])
    coords = torch.cat([bundle["coords"][axis] for axis in ("x", "y", "t")], dim=1)
    for field, values in zip(("u", "v", "p"), run_model_in_chunks(expected, coords, "cpu", chunk_size=8000)):
        assert torch.equal(bundle["clean"][field], values)
    assert bundle["split"] == case["split"]
    assert len(rows) == len(PERTURBATION_NAMES) * len(EPSILON_VALUES) + 1
    assert {row["split"] for row in rows} == {case["split"]}


def test_hallucinations_stop_on_a_missing_model(cases, tmp_path):
    with pytest.raises(FileNotFoundError, match="case_30"):
        generate_case_hallucinations("case_30", cases["case_30"], tmp_path, tmp_path, "cpu", chunk_size=8000)


# --- evaluate_phs.py --------------------------------------------------------------

@pytest.mark.parametrize("case_ids, models_dir", [
    (["case_31", "case_30"], CONFIRMATION_MODELS),
    (["case_00", "case_07"], OLD_MODELS),
])
def test_scoring_uses_the_model_of_the_models_folder(cases, args, scored, case_ids, models_dir):
    df = evaluate_phs.score_all_fields(index_rows(case_ids), cases, models_dir, args)

    assert list(df["case_id"]) == [case_id for case_id in sorted(case_ids) for _ in range(2)]
    assert [case_id for case_id, _, _ in scored] == list(df["case_id"])
    for case_id, model, seed in scored:
        case_dir = models_dir / case_id
        model_file = case_dir / "lbfgs_final.pth" if case_dir.is_dir() else models_dir / f"{case_id}.pth"
        assert_same_weights(model, model_file, cases[case_id]["k"])
        # PROTOCOL.md section 4: every field of case n is scored at seed 1000 + n, a new case's seed_scoring
        assert seed == 1000 + int(case_id.split("_")[1])
        if "seed_scoring" in cases[case_id]:
            assert seed == cases[case_id]["seed_scoring"]


def test_scoring_stops_before_any_field_on_a_missing_model(cases, args, scored, tmp_path):
    # case_30 has its model and case_31 has none: nothing is scored, not even case_30
    (tmp_path / "case_30").mkdir()
    for suffix in (".pth", ".json"):
        shutil.copy(CONFIRMATION_MODELS / "case_30" / f"lbfgs_final{suffix}", tmp_path / "case_30")
    with pytest.raises(FileNotFoundError, match="case_31"):
        evaluate_phs.score_all_fields(index_rows(["case_30", "case_31"]), cases, tmp_path, args)
    assert scored == []


def test_scoring_stops_before_any_field_on_a_case_missing_from_the_case_file(cases, args, scored):
    with pytest.raises(ValueError, match="case_99"):
        evaluate_phs.score_all_fields(index_rows(["case_30", "case_99"]), cases, CONFIRMATION_MODELS, args)
    assert scored == []
