"""
Unit tests for src.detection.frozen_calibration and evaluate_phs.py --calibration (#45).

Verifies that:
  - The five scores are the three old ones plus PHS without bc and PHS without E,
    and the default evaluate_phs.py run keeps the three old ones.
  - fit() uses exactly the 30 clean rows, never a damaged one, and refuses a table
    with any other number of clean fields.
  - apply() and the --calibration path use only the saved numbers: no fitting
    function runs, and each field's scores follow from the frozen normalizers
    and tau alone, whatever the other fields are.
  - A saved calibration reads back exactly, and is written the same way twice.
  - calibration/source_phs_components_raw.csv is the old package's file, byte for byte.
  - The committed calibration/frozen_calibration.json is exactly what fit() gives on
    that file, and applying it to the 30 calibration fields gives back its numbers.
  - PROTOCOL.md records the SHA-256 of both files and the frozen values, with no TBD left.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection import evaluate_phs, frozen_calibration, phs
from src.detection.phs import BASELINE_DEFINITIONS, PHS_COMPONENT_NAMES, SCORE_DEFINITIONS

REPO = Path(__file__).parent.parent
SOURCE_CSV = REPO / "calibration" / "source_phs_components_raw.csv"
FROZEN_JSON = REPO / "calibration" / "frozen_calibration.json"
PROTOCOL_MD = REPO / "PROTOCOL.md"
# The old package's 01_raw_results/phs_components_raw.csv, which a rerun at commit 4437907
# reproduced byte for byte (6 Oct 2026)
SOURCE_SHA256 = "ee3dd7d1e2bfa571d9cb752cc92076640bd4c4d5686ba5e6816af57a089d15bd"

# Rough sizes of the real clean components, so the made-up tables look like the real one
COMPONENT_SCALES = {"mom": 7e-6, "div": 5e-7, "bc": 1.5, "E": 2e-6}


def make_fields(n_cases=30, seed=0):
    """
    A made-up score table shaped like the source CSV: for each case, one clean
    field and two damaged ones with larger components.

    Outputs:
        pd.DataFrame: Columns case_id, label, perturbation_type, epsilon, mom, div, bc, E.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(n_cases):
        for label, epsilon, size in (("clean", 0.0, 1.0), ("hallucinated", 0.01, 5.0), ("hallucinated", 0.05, 50.0)):
            rows.append({
                "case_id": f"case_{n:02d}",
                "label": label,
                "perturbation_type": "none" if label == "clean" else "momentum",
                "epsilon": epsilon,
                **{c: scale * size * rng.lognormal(0.0, 0.5) for c, scale in COMPONENT_SCALES.items()},
            })
    return pd.DataFrame(rows)


def fit_table(tmp_path, df):
    """Writes df to a CSV in tmp_path and returns fit() on it."""
    csv_path = tmp_path / "source.csv"
    df.to_csv(csv_path, index=False)
    return frozen_calibration.fit(csv_path)


def refuse(*args, **kwargs):
    raise AssertionError("a fitting function was called")


def test_five_scores():
    assert SCORE_DEFINITIONS == {
        "Score1_momentum_only": ["mom"],
        "Score2_momentum_divergence": ["mom", "div"],
        "Score3_PHS_full": ["mom", "div", "bc", "E"],
        "Score4_PHS_no_bc": ["mom", "div", "E"],
        "Score5_PHS_no_E": ["mom", "div", "bc"],
    }
    # The default evaluate_phs.py run still writes exactly the three old scores
    assert list(BASELINE_DEFINITIONS) == ["Score1_momentum_only", "Score2_momentum_divergence", "Score3_PHS_full"]


def test_fit_uses_exactly_the_30_clean_rows(tmp_path, monkeypatch):
    seen = {"tau_inputs": []}
    real_normalizers, real_threshold = frozen_calibration.compute_normalizers, frozen_calibration.select_threshold

    def record_normalizers(rows, *args, **kwargs):
        seen["rows"] = rows
        return real_normalizers(rows, *args, **kwargs)

    def record_threshold(scores, *args, **kwargs):
        seen["tau_inputs"].append(len(scores))
        return real_threshold(scores, *args, **kwargs)

    monkeypatch.setattr(frozen_calibration, "compute_normalizers", record_normalizers)
    monkeypatch.setattr(frozen_calibration, "select_threshold", record_threshold)
    calib = fit_table(tmp_path, make_fields())

    rows = seen["rows"]
    assert len(rows) == 30
    assert (rows["label"] == "clean").all()
    assert rows["case_id"].nunique() == 30
    assert seen["tau_inputs"] == [30] * len(SCORE_DEFINITIONS)
    assert calib["fit_case_ids"] == [f"case_{n:02d}" for n in range(30)]


def test_fit_follows_the_formulas(tmp_path):
    df = make_fields()
    calib = fit_table(tmp_path, df)
    clean = df[df["label"] == "clean"]

    for c in PHS_COMPONENT_NAMES:
        assert calib["normalizers"][c] == clean[c].mean()
    bars = {c: clean[c] / (calib["normalizers"][c] + 1e-12) for c in PHS_COMPONENT_NAMES}
    for score_name, components in SCORE_DEFINITIONS.items():
        score = sum(bars[c] for c in components)
        assert calib["thresholds"][score_name] == np.percentile(score, 95, method="linear")
    assert calib["scores"] == SCORE_DEFINITIONS
    assert (calib["percentile"], calib["percentile_method"], calib["normalizer_eps"]) == (95.0, "linear", 1e-12)


def test_fit_ignores_the_damaged_fields(tmp_path):
    df = make_fields()
    changed = df.copy()
    damaged = changed["label"] == "hallucinated"
    changed.loc[damaged, PHS_COMPONENT_NAMES] *= 1000

    calib, calib_changed = fit_table(tmp_path, df), fit_table(tmp_path, changed)

    assert calib_changed["normalizers"] == calib["normalizers"]
    assert calib_changed["thresholds"] == calib["thresholds"]


@pytest.mark.parametrize("change", ["one_missing", "one_twice", "one_more_case"])
def test_fit_refuses_any_other_number_of_clean_fields(tmp_path, change):
    df = make_fields()
    first_clean = df.index[df["label"] == "clean"][0]
    if change == "one_missing":
        df = df.drop(first_clean)
    elif change == "one_twice":
        df = pd.concat([df, df.loc[[first_clean]]], ignore_index=True)
    else:
        df = make_fields(n_cases=31)

    with pytest.raises(ValueError, match="exactly 30 clean fields"):
        fit_table(tmp_path, df)


def test_apply_never_fits(tmp_path, monkeypatch):
    calib = fit_table(tmp_path, make_fields())
    for module, name in [(frozen_calibration, "fit"), (frozen_calibration, "compute_normalizers"),
                         (frozen_calibration, "select_threshold"), (phs, "compute_normalizers"),
                         (phs, "select_threshold"), (np, "percentile")]:
        monkeypatch.setattr(module, name, refuse)

    scored = frozen_calibration.apply(make_fields(seed=1), calib)

    assert all(f"flagged_{name}" in scored for name in SCORE_DEFINITIONS)


def test_apply_uses_only_the_saved_numbers(tmp_path):
    calib = fit_table(tmp_path, make_fields())
    new = make_fields(seed=1)
    # Columns left over from another calibration, as in the old pipeline's table, are replaced
    new["mom_bar"] = -1.0
    new["Score3_PHS_full"] = -1.0

    scored = frozen_calibration.apply(new, calib)

    for c in PHS_COMPONENT_NAMES:
        np.testing.assert_array_equal(scored[f"{c}_bar"], new[c] / (calib["normalizers"][c] + 1e-12))
    for score_name, components in SCORE_DEFINITIONS.items():
        np.testing.assert_array_equal(scored[score_name], sum(scored[f"{c}_bar"] for c in components))
        np.testing.assert_array_equal(scored[f"flagged_{score_name}"],
                                      scored[score_name] > calib["thresholds"][score_name])

    # Each field is scored on its own: a refit would change the scores with the other fields
    pd.testing.assert_frame_equal(frozen_calibration.apply(new.iloc[:7], calib), scored.iloc[:7])
    # Ten times larger components give ten times larger scores; a refit would normalize that away
    tenfold = new.copy()
    tenfold[PHS_COMPONENT_NAMES] *= 10
    np.testing.assert_allclose(frozen_calibration.apply(tenfold, calib)["Score3_PHS_full"],
                               10 * scored["Score3_PHS_full"], rtol=1e-12)


def test_a_field_exactly_at_tau_is_not_flagged(tmp_path):
    calib = fit_table(tmp_path, make_fields())
    new = make_fields(seed=1)
    first_score = float(frozen_calibration.apply(new, calib)["Score3_PHS_full"].iloc[0])
    at_tau = {**calib, "thresholds": {**calib["thresholds"], "Score3_PHS_full": first_score}}

    assert not frozen_calibration.apply(new, at_tau)["flagged_Score3_PHS_full"].iloc[0]


def test_save_and_load_give_back_the_same_calibration(tmp_path):
    calib = fit_table(tmp_path, make_fields())
    json_path = tmp_path / "frozen.json"

    frozen_calibration.save(calib, json_path)

    assert frozen_calibration.load(json_path) == calib
    assert b"\r" not in json_path.read_bytes()  # The same bytes, and SHA-256, on every OS


@pytest.mark.parametrize("edit", ["score_changed", "tau_missing", "other_version"])
def test_load_refuses_a_calibration_for_other_scores(tmp_path, edit):
    calib = fit_table(tmp_path, make_fields())
    if edit == "score_changed":
        calib["scores"]["Score4_PHS_no_bc"] = ["mom", "E"]
    elif edit == "tau_missing":
        del calib["thresholds"]["Score5_PHS_no_E"]
    else:
        calib["format_version"] = 2
    json_path = tmp_path / "frozen.json"
    frozen_calibration.save(calib, json_path)

    with pytest.raises(ValueError):
        frozen_calibration.load(json_path)


def test_script_writes_the_json_and_keeps_an_existing_one(tmp_path):
    source, out = tmp_path / "source.csv", tmp_path / "frozen.json"
    make_fields().to_csv(source, index=False)
    argv = ["--source", str(source), "--out", str(out)]

    frozen_calibration.main(argv)
    first = out.read_bytes()
    assert frozen_calibration.load(out)["thresholds"] == frozen_calibration.fit(source)["thresholds"]

    with pytest.raises(SystemExit):
        frozen_calibration.main(argv)
    assert out.read_bytes() == first

    # The same source and commit write the same bytes: the file has no timestamp
    frozen_calibration.main(argv + ["--overwrite"])
    assert out.read_bytes() == first


def test_source_csv_is_the_old_package_file():
    # A different hash usually means git changed the line endings (see .gitattributes)
    assert hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest() == SOURCE_SHA256
    fields = pd.read_csv(SOURCE_CSV)
    assert len(fields) == 930
    assert (fields["label"] == "clean").sum() == 30


def test_committed_calibration_is_the_fit_of_the_source():
    committed = frozen_calibration.load(FROZEN_JSON)
    fitted = frozen_calibration.fit(SOURCE_CSV)
    provenance = ("git_commit", "git_dirty")  # Those of the commit that wrote the file

    assert ({key: value for key, value in committed.items() if key not in provenance}
            == {key: value for key, value in fitted.items() if key not in provenance})
    assert committed["source_csv_sha256"] == SOURCE_SHA256
    assert committed["git_dirty"] is False
    assert len(committed["git_commit"]) == 40


def test_frozen_values_come_back_on_the_calibration_fields():
    calib = frozen_calibration.load(FROZEN_JSON)
    fields = pd.read_csv(SOURCE_CSV, float_precision="round_trip")
    clean = frozen_calibration.apply(fields, calib).query("label == 'clean'")

    assert sorted(clean["case_id"]) == calib["fit_case_ids"]
    for c in PHS_COMPONENT_NAMES:
        assert clean[c].mean() == calib["normalizers"][c]
    for score_name, tau in calib["thresholds"].items():
        assert np.percentile(clean[score_name], 95, method="linear") == tau
        # tau lies between the 3rd and the 2nd largest of the 30 scores
        assert clean[f"flagged_{score_name}"].sum() == 2


def test_protocol_records_the_frozen_files_and_values():
    protocol = PROTOCOL_MD.read_text(encoding="utf-8")
    calib = frozen_calibration.load(FROZEN_JSON)

    for path in (SOURCE_CSV, FROZEN_JSON):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in protocol
    assert b"\r" not in FROZEN_JSON.read_bytes()  # So its SHA-256 is the same on every OS
    assert calib["git_commit"][:7] in protocol
    for normalizer in calib["normalizers"].values():
        assert (f"{normalizer:.6e}" if normalizer < 1e-3 else f"{normalizer:.6f}") in protocol
    for score_name, tau in calib["thresholds"].items():
        assert f"| `{score_name}` | {tau:.6f} |" in protocol
    assert "TBD" not in protocol


def test_evaluate_phs_calibration_path_never_fits(tmp_path, monkeypatch):
    calib = fit_table(tmp_path, make_fields())
    new = make_fields(seed=1)
    for module, name in [(evaluate_phs, "evaluate_detection"), (evaluate_phs, "compute_normalizers"),
                         (evaluate_phs, "select_threshold"), (frozen_calibration, "fit"),
                         (frozen_calibration, "compute_normalizers"), (frozen_calibration, "select_threshold")]:
        monkeypatch.setattr(module, name, refuse)

    written = evaluate_phs.write_frozen_scores(new, calib, tmp_path)

    pd.testing.assert_frame_equal(written, frozen_calibration.apply(new, calib))
    read_back = pd.read_csv(tmp_path / evaluate_phs.FROZEN_SCORES_FILE, float_precision="round_trip")
    pd.testing.assert_frame_equal(read_back, written)


@pytest.mark.parametrize("argv, expected", [
    ([], None),
    (["--calibration", "calibration/frozen_calibration.json"], "calibration/frozen_calibration.json"),
])
def test_calibration_flag(monkeypatch, argv, expected):
    monkeypatch.setattr(sys, "argv", ["evaluate_phs.py", *argv])
    assert evaluate_phs.parse_args().calibration == expected
