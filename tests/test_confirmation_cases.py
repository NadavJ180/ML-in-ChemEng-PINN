"""
Unit tests for the pre-registered confirmation cases (cases/confirmation_cases.json, #44).

Verifies that the committed file:
  - Holds case_30 ... case_59 in the single split "confirmation", 10 cases for each k.
  - Keeps every parameter inside its sampling range, and every seed on its rule.
  - Shares no seed and no case_id with the existing 30 cases.
  - Keeps the field names that training (#38) and scoring read.
  - Is what `python src/data/sampler.py --confirmation` regenerates: byte for byte,
    except the last bits of Re, which depend on the platform.
"""

import json
import math
import numpy as np
import pytest
from pathlib import Path

from src.data.sampler import generate_confirmation_cases, generate_randomized_cases, export_dataset

CASES_PATH = Path(__file__).parent.parent / "cases" / "confirmation_cases.json"


@pytest.fixture
def confirmation():
    """
    Loads the committed confirmation file.

    Outputs:
        dict: {"metadata": {...}, "confirmation": [30 case dicts]}.
    """
    with open(CASES_PATH, "r") as f:
        return json.load(f)


def test_split_and_case_ids(confirmation):
    cases = confirmation["confirmation"]

    assert set(confirmation) == {"metadata", "confirmation"}
    assert confirmation["metadata"]["total_cases"] == 30
    assert confirmation["metadata"]["sampler_seed"] == 2026
    assert [c["case_id"] for c in cases] == [f"case_{n}" for n in range(30, 60)]
    assert [c["k"] for c in cases] == [1] * 10 + [2] * 10 + [3] * 10
    assert all(c["split"] == "confirmation" for c in cases)


def test_parameters_in_range(confirmation):
    for c in confirmation["confirmation"]:
        assert 0.5 <= c["U0"] <= 2.0, f"[{c['case_id']}] U0 out of range"
        assert 20 <= c["Re"] <= 300, f"[{c['case_id']}] Re out of range"
        assert 0 <= c["phi_x"] < 2 * np.pi, f"[{c['case_id']}] phi_x out of range"
        assert 0 <= c["phi_y"] < 2 * np.pi, f"[{c['case_id']}] phi_y out of range"


def test_seed_rules(confirmation):
    for c in confirmation["confirmation"]:
        n = int(c["case_id"].split("_")[1])
        assert c["seed_points"] == 3000 + n
        assert c["seed_train"] == 4000 + n
        assert c["seed_scoring"] == 1000 + n


def test_no_overlap_with_existing_cases(confirmation):
    """
    The existing seeds are the sampler seed 42 (also the global seed of
    generate_datasets.py and train_model.py), the per-case "seed" field 42-71,
    and the scoring seeds 1000-1029 (evaluate_phs.py's 1000 + case number).
    data/cases_metadata.json is not tracked, so the existing cases are regenerated.
    """
    existing = generate_randomized_cases(num_cases=30, seed=42)
    existing_ids = {c["case_id"] for c in existing}
    existing_seeds = ({42}
                      | {c["seed"] for c in existing}
                      | {1000 + int(c["case_id"].split("_")[1]) for c in existing})

    cases = confirmation["confirmation"]
    new_ids = {c["case_id"] for c in cases}
    new_seeds = [c[field] for c in cases for field in ("seed_points", "seed_train", "seed_scoring")]
    new_seeds.append(confirmation["metadata"]["sampler_seed"])

    assert not existing_ids & new_ids
    assert not existing_seeds & set(new_seeds)
    assert len(set(new_seeds)) == len(new_seeds), "A seed is reused within the new cases"


def test_field_names(confirmation):
    """#38's training code reads these names: change them only after telling Alon."""
    assert list(confirmation["metadata"]) == ["description", "total_cases", "sampler_seed", "created_by"]
    for c in confirmation["confirmation"]:
        assert list(c) == ["case_id", "split", "k", "U0", "Re", "phi_x", "phi_y",
                           "seed_points", "seed_train", "seed_scoring"]


def test_regeneration_matches_the_committed_file(tmp_path):
    """
    Re is drawn log-uniformly, through np.log and np.exp, whose last bit can differ
    between platforms: on Linux, case_32's Re comes out as 44.860206966058605 instead
    of the committed 44.86020696605861. So Re may differ by a few units in the last
    place, and everything else must match byte for byte. Training and scoring read
    the committed file, never a regenerated one.
    """
    output_path = tmp_path / "confirmation_cases.json"
    export_dataset(generate_confirmation_cases(), output_path, newline="\n")
    regenerated = json.loads(output_path.read_text())

    # The sampler writes LF, but git may check the file out with CRLF on Windows (core.autocrlf)
    committed = CASES_PATH.read_bytes().replace(b"\r\n", b"\n")
    for new, old in zip(regenerated["confirmation"], json.loads(committed)["confirmation"], strict=True):
        assert abs(new["Re"] - old["Re"]) <= 4 * math.ulp(old["Re"]), f"[{new['case_id']}] Re differs"
        new["Re"] = old["Re"]

    # With the committed Re values, the regenerated file is the committed one, byte for byte
    export_dataset(regenerated, output_path, newline="\n")
    assert output_path.read_bytes() == committed
