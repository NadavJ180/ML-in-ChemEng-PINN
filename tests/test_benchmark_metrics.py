"""
Unit tests for src.detection.benchmark_metrics and the path flags of the perturbation benchmark (#35).

On a tiny made-up benchmark (6 cases, 2 per k, each with a clean field and 4 damaged
fields; worked out by hand below), verifies that:
  - recall, recall on new detections only, precision, F1 and the false-alarm rate are
    right overall, by k, by damage type and by damage type x strength;
  - a case whose clean field a score already flags is "already flagged" for that score
    only, and is left out of recall_new;
  - every false alarm and every miss is listed, and a score equal to tau is not flagged;
  - the false alarms, and recall at one damage type and strength, get the exact range;
    the other metrics the case bootstrap of grouped_eval.py, with n_boot reported;
  - precision, recall and F1 match sklearn's, NaN where undefined;
  - failed runs stay in every table and are counted, and metrics_passed_only is
    written only when a run failed;
  - every table is written as .csv and .tex, and reads back exactly;
  - scores that don't match the frozen calibration, a broken table, scores that miss or
    add a case of the case file, or a bad sidecar are refused.
and that the path flags of generate_hallucinations.py and evaluate_phs.py default to
today's paths.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.detection import benchmark_metrics as bm
from src.detection import evaluate_phs, frozen_calibration
from src.detection.grouped_eval import case_bootstrap, clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations import generate_hallucinations

N_BOOT = 100  # Enough to check the wiring; the real run uses bm.N_BOOT = 2000

S1, S2, S3, S4, S5 = SCORE_DEFINITIONS
# Score1 flags above 1, Score2 above 2, and the other three flag nothing
THRESHOLDS = {S1: 1.0, S2: 2.0, S3: 100.0, S4: 100.0, S5: 100.0}

DAMAGED = [("velocity_divergence", 0.01), ("velocity_divergence", 0.05), ("pressure", 0.01), ("pressure", 0.05)]
# case_id: (k, clean score, scores of the DAMAGED fields in that order). Every score of a field
# is the same number (see make_inputs), so only tau differs between the scores.
CASES = {
    "case_30": (1, 1.5, [0.8, 5.0, 1.2, 6.0]),   # Clean above 1: already flagged by Score1
    "case_31": (1, 0.5, [0.9, 3.0, 0.7, 4.0]),
    "case_40": (2, 0.5, [1.1, 2.5, 0.6, 7.0]),
    "case_41": (2, 2.5, [3.0, 4.0, 2.6, 5.0]),   # Clean above 1 and 2: already flagged by Score1 and Score2
    "case_50": (3, 0.5, [1.0, 1.5, 0.3, 2.2]),   # 1.0 equals Score1's tau, so it is not flagged
    "case_51": (3, 0.5, [1.3, 8.0, 0.9, 9.0]),
}

# Worked out by hand. Score1 (tau 1) flags 17 of the 24 damaged fields: 3, 2, 3, 4, 2, 3 per case,
# and the clean fields of case_30 and case_41. Leaving those two cases out, it flags 2+3+2+3 = 10 of 16.
# Score2 (tau 2) flags 13 of 24 (2, 2, 2, 4, 1, 2 per case) and the clean field of case_41;
# without case_41, 9 of 20.
OVERALL = {
    S1: dict(recall=17 / 24, recall_new=10 / 16, precision=17 / 19, f1=34 / 43, n_false_alarms=2,
             n_preflagged_cases=2, n_damaged_new=16, n_damaged_new_flagged=10),
    S2: dict(recall=13 / 24, recall_new=9 / 20, precision=13 / 14, f1=26 / 38, n_false_alarms=1,
             n_preflagged_cases=1, n_damaged_new=20, n_damaged_new_flagged=9),
    S3: dict(recall=0.0, recall_new=0.0, precision=np.nan, f1=0.0, n_false_alarms=0,
             n_preflagged_cases=0, n_damaged_new=24, n_damaged_new_flagged=0),
}
# Score1 by k: (recall, recall_new, false alarms, precision, F1)
SCORE1_BY_K = {1: (5 / 8, 2 / 4, 1, 5 / 6, 10 / 14), 2: (7 / 8, 3 / 4, 1, 7 / 8, 14 / 16), 3: (5 / 8, 5 / 8, 0, 1.0, 10 / 13)}
# Score1 by type: (recall, recall_new)
SCORE1_BY_TYPE = {"velocity_divergence": (9 / 12, 6 / 8), "pressure": (8 / 12, 4 / 8)}
# Score1 by type x strength: (flagged, out of, flagged among new, out of)
SCORE1_BY_TYPE_EPS = {("velocity_divergence", 0.01): (3, 6, 2, 4), ("velocity_divergence", 0.05): (6, 6, 4, 4),
                      ("pressure", 0.01): (2, 6, 0, 4), ("pressure", 0.05): (6, 6, 4, 4)}


def make_calibration() -> dict:
    """A frozen calibration whose normalizers are 1 (and eps 0), so each normalized component is its raw value."""
    return {
        "format_version": frozen_calibration.FORMAT_VERSION,
        "description": "Made-up calibration of tests/test_benchmark_metrics.py",
        "normalizers": {component: 1.0 for component in PHS_COMPONENT_NAMES},
        "normalizer_eps": 0.0,
        "scores": {name: list(components) for name, components in SCORE_DEFINITIONS.items()},
        "thresholds": dict(THRESHOLDS),
    }


def make_fields() -> pd.DataFrame:
    """
    The made-up fields as evaluate_phs.py's index rows plus raw components. Only mom is
    nonzero, so every score of a field equals its mom exactly.
    """
    rows = []
    for case_id, (k, clean_value, damaged_values) in CASES.items():
        base = {"case_id": case_id, "split": "confirmation", "Re": 100.0, "U0": 1.0, "k": k, "T": 1.0}
        for (perturbation, epsilon), value in zip(DAMAGED, damaged_values):
            rows.append({**base, "perturbation_type": perturbation, "epsilon": epsilon, "label": "hallucinated",
                         "mom": value, "div": 0.0, "bc": 0.0, "E": 0.0})
        rows.append({**base, "perturbation_type": "none", "epsilon": 0.0, "label": "clean",
                     "mom": clean_value, "div": 0.0, "bc": 0.0, "E": 0.0})
    return pd.DataFrame(rows)


def write_sidecars(models_dir: Path, failed=()):
    """One lbfgs_final.json sidecar per case, PASSED unless the case is in failed."""
    for case_id, (k, _, _) in CASES.items():
        case_dir = models_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        passed = case_id not in failed
        sidecar = {"case_id": case_id, "checkpoint": "lbfgs_final", "status": "PASSED" if passed else "FAILED",
                   "metrics": {"passed": passed}, "k": k}
        (case_dir / "lbfgs_final.json").write_text(json.dumps(sidecar))


def write_case_file(path: Path, cases: dict = CASES):
    """A case file of the made-up cases, laid out like cases/confirmation_cases.json."""
    path.write_text(json.dumps({"confirmation": [{"case_id": case_id, "k": k} for case_id, (k, _, _) in cases.items()]}))


def make_inputs(folder: Path, failed=()) -> SimpleNamespace:
    """Writes the scores (as evaluate_phs.py --calibration does), the calibration, the case file and the sidecars."""
    folder.mkdir(parents=True, exist_ok=True)
    calib = make_calibration()
    calibration = folder / "calibration.json"
    frozen_calibration.save(calib, calibration)
    evaluate_phs.write_frozen_scores(make_fields(), frozen_calibration.load(calibration), folder)
    write_case_file(folder / "cases.json")
    write_sidecars(folder / "models", failed)
    return SimpleNamespace(scores=folder / evaluate_phs.FROZEN_SCORES_FILE, calibration=calibration,
                           metadata=folder / "cases.json", models_dir=folder / "models", output_dir=folder / "out")


def run_benchmark(inputs: SimpleNamespace) -> dict:
    return bm.run(inputs.scores, inputs.calibration, inputs.models_dir, inputs.output_dir, n_boot=N_BOOT,
                  metadata=inputs.metadata)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """The made-up benchmark with every run passed, run once: its inputs and tables."""
    inputs = make_inputs(tmp_path_factory.mktemp("bench"))
    return SimpleNamespace(inputs=inputs, tables=run_benchmark(inputs))


def row_of(table: pd.DataFrame, **keys) -> pd.Series:
    """The single row of table matching keys."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in keys.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, f"{mask.sum()} rows match {keys}"
    return table[mask].iloc[0]


def assert_close(actual, expected):
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected, abs=1e-12)


# --- Metrics worked out by hand ---

@pytest.mark.parametrize("score", [S1, S2, S3])
def test_overall_metrics_by_hand(bench, score):
    row = row_of(bench.tables["metrics_overall"], score=score)
    for name, expected in OVERALL[score].items():
        assert_close(row[name], expected)
    assert (row["n_cases"], row["n_clean"], row["n_damaged"], row["n_failed_runs"]) == (6, 6, 24, 0)
    assert row["tau"] == THRESHOLDS[score]
    assert_close(row["fpr"], OVERALL[score]["n_false_alarms"] / 6)
    assert (row["fpr_low"], row["fpr_high"]) == clopper_pearson(OVERALL[score]["n_false_alarms"], 6)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_metrics_by_k_by_hand(bench, k):
    row = row_of(bench.tables["metrics_by_k"], score=S1, k=k)
    expected_recall, expected_recall_new, false_alarms, expected_precision, expected_f1 = SCORE1_BY_K[k]
    assert_close(row["recall"], expected_recall)
    assert_close(row["recall_new"], expected_recall_new)
    assert_close(row["precision"], expected_precision)
    assert_close(row["f1"], expected_f1)
    assert (row["n_cases"], row["n_clean"], row["n_false_alarms"]) == (2, 2, false_alarms)
    assert (row["fpr_low"], row["fpr_high"]) == clopper_pearson(false_alarms, 2)


@pytest.mark.parametrize("perturbation", ["velocity_divergence", "pressure"])
def test_metrics_by_type_by_hand(bench, perturbation):
    row = row_of(bench.tables["metrics_by_type"], score=S1, perturbation=perturbation)
    assert_close(row["recall"], SCORE1_BY_TYPE[perturbation][0])
    assert_close(row["recall_new"], SCORE1_BY_TYPE[perturbation][1])
    # Against all 6 clean fields, so the false alarms are the overall ones
    assert (row["n_clean"], row["n_damaged"], row["n_false_alarms"]) == (6, 12, 2)


@pytest.mark.parametrize("perturbation, epsilon", list(SCORE1_BY_TYPE_EPS))
def test_metrics_by_type_eps_exact_ranges(bench, perturbation, epsilon):
    table = bench.tables["metrics_by_type_eps"]
    row = row_of(table, score=S1, perturbation=perturbation, epsilon=epsilon)
    flagged, n, flagged_new, n_new = SCORE1_BY_TYPE_EPS[(perturbation, epsilon)]
    assert (row["n_damaged_flagged"], row["n_damaged"]) == (flagged, n)
    assert (row["n_damaged_new_flagged"], row["n_damaged_new"]) == (flagged_new, n_new)
    assert_close(row["recall"], flagged / n)
    assert (row["recall_low"], row["recall_high"]) == clopper_pearson(flagged, n)
    assert_close(row["recall_new"], flagged_new / n_new)
    assert (row["recall_new_low"], row["recall_new_high"]) == clopper_pearson(flagged_new, n_new)
    # One field per case here: recall has the exact range, the other metrics the bootstrap
    assert "recall_n_boot" not in table.columns and "recall_new_n_boot" not in table.columns
    assert {"auc_n_boot", "precision_n_boot", "f1_n_boot"} <= set(table.columns)
    assert len(table) == len(SCORE_DEFINITIONS) * len(DAMAGED)


def test_a_score_equal_to_tau_is_not_flagged(bench):
    fields = bench.tables["confirmation_scores_raw"]
    at_tau = row_of(fields, case_id="case_50", perturbation="velocity_divergence", epsilon=0.01)
    assert at_tau[S1] == THRESHOLDS[S1] and not at_tau[f"flagged_{S1}"]
    row_of(bench.tables["missed_detections"], score=S1, case_id="case_50", perturbation="velocity_divergence",
           epsilon=0.01)


# --- Already-flagged cases, false alarms and misses ---

def test_preflagged_columns(bench):
    fields = bench.tables["confirmation_scores_raw"]
    expected = {S1: {"case_30", "case_41"}, S2: {"case_41"}, S3: set(), S4: set(), S5: set()}
    for score, cases in expected.items():
        assert set(fields.loc[fields[f"preflagged_{score}"], "case_id"]) == cases
        # Every field of an already-flagged case carries the mark, not only its clean field
        assert fields[f"preflagged_{score}"].sum() == 5 * len(cases)


def test_preflagged_cases_table(bench):
    table = bench.tables["preflagged_cases"]
    assert list(zip(table["score"], table["case_id"], table["n_damaged_flagged"])) == [
        (S1, "case_30", 3), (S1, "case_41", 4), (S2, "case_41", 4)]
    assert (table["n_damaged"] == 4).all() and (table["clean_score"] > table["tau"]).all()


def test_every_false_alarm_listed(bench):
    table = bench.tables["false_alarms"]
    assert list(zip(table["score"], table["case_id"])) == [(S1, "case_30"), (S1, "case_41"), (S2, "case_41")]
    assert (table["perturbation"] == "none").all() and (table["score_value"] > table["tau"]).all()


def test_every_miss_listed(bench):
    table = bench.tables["missed_detections"]
    assert table.groupby("score").size().to_dict() == {S1: 7, S2: 11, S3: 24, S4: 24, S5: 24}
    assert (table["score_value"] <= table["tau"]).all() and (table["perturbation"] != "none").all()
    fields = bench.tables["confirmation_scores_raw"]
    for score in SCORE_DEFINITIONS:
        missed = fields[(fields["label"] != "clean") & ~fields[f"flagged_{score}"]]
        listed = table[table["score"] == score]
        assert sorted(zip(missed["case_id"], missed["perturbation"], missed["epsilon"])) == \
            sorted(zip(listed["case_id"], listed["perturbation"], listed["epsilon"]))


def test_clean_false_positives_table(bench):
    table = bench.tables["clean_false_positives"]
    expected = {"all": (2, 6, "case_30;case_41"), "k=1": (1, 2, "case_30"), "k=2": (1, 2, "case_41"), "k=3": (0, 2, "")}
    for subset, (count, n, cases) in expected.items():
        row = row_of(table, score=S1, subset=subset)
        assert (row["n_false_alarms"], row["n_clean"], row["false_alarm_cases"]) == (count, n, cases)
        assert (row["fpr_low"], row["fpr_high"]) == clopper_pearson(count, n)
    assert len(table) == len(SCORE_DEFINITIONS) * 4


# --- Ranges and metric definitions ---

def test_bootstrap_ranges_come_from_grouped_eval(bench):
    fields = bench.tables["confirmation_scores_raw"]
    row = row_of(bench.tables["metrics_overall"], score=S1)
    expected = case_bootstrap(bm.score_view(fields, S1), bm.auc, n_boot=N_BOOT, seed=bm.BOOT_SEED)
    assert (row["auc"], row["auc_low"], row["auc_high"], row["auc_n_boot"]) == tuple(expected)
    assert row["auc_n_boot"] == N_BOOT and row["auc_low"] <= row["auc_high"]
    assert row["auc"] == pytest.approx(roc_auc_score(fields["label"] != "clean", fields[S1]), abs=1e-12)
    # Score3 flags nothing, so precision is undefined on every resample
    assert row_of(bench.tables["metrics_overall"], score=S3)["precision_n_boot"] == 0


@pytest.mark.parametrize("seed", range(6))
def test_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = 40
    view = pd.DataFrame({"damaged": rng.random(n) < 0.7,
                         "flagged": rng.random(n) < [0.0, 0.3, 0.9][seed % 3],
                         "preflagged": False,
                         # Rounded, so that some scores tie
                         "value": np.round(rng.random(n), 1)})
    for ours, theirs in [(bm.precision, precision_score), (bm.recall, recall_score), (bm.f1, f1_score)]:
        assert_close(ours(view), theirs(view["damaged"], view["flagged"], zero_division=np.nan))
    assert bm.auc(view) == pytest.approx(roc_auc_score(view["damaged"], view["value"]), abs=1e-12)
    assert np.isnan(bm.auc(view[view["damaged"]])) and np.isnan(bm.recall(view[~view["damaged"]]))


# --- Failed runs ---

def test_failed_run_is_kept_and_counted(tmp_path, bench):
    inputs = make_inputs(tmp_path, failed=("case_51",))
    tables = run_benchmark(inputs)

    fields = tables["confirmation_scores_raw"]
    assert len(fields) == 30 and not fields.loc[fields["case_id"] == "case_51", "passed_acceptance"].any()
    assert (tables["metrics_overall"]["n_failed_runs"] == 1).all()
    by_k = tables["metrics_by_k"]
    assert (by_k.loc[by_k["k"] == 3, "n_failed_runs"] == 1).all() and (by_k.loc[by_k["k"] != 3, "n_failed_runs"] == 0).all()
    # The failed run's results are the same as when it passed: nothing is dropped
    pd.testing.assert_frame_equal(tables["metrics_overall"].drop(columns="n_failed_runs"),
                                  bench.tables["metrics_overall"].drop(columns="n_failed_runs"))

    # Passed runs only: case_51 (3 of its 4 damaged fields flagged by Score1) is left out
    passed_only = tables["metrics_passed_only"]
    row = row_of(passed_only, score=S1, subset="all")
    assert_close(row["recall"], 14 / 20)
    assert (row["n_cases"], row["n_failed_runs"], row["n_false_alarms"], row["n_clean"]) == (5, 0, 2, 5)
    assert row_of(passed_only, score=S1, subset="k=3")["n_cases"] == 1
    assert list(passed_only["subset"][:4]) == ["all", "k=1", "k=2", "k=3"]
    for suffix in (".csv", ".tex"):
        assert (inputs.output_dir / f"metrics_passed_only{suffix}").exists()

    # Once every run passes, the passed-only table goes away
    write_sidecars(inputs.models_dir)
    assert "metrics_passed_only" not in run_benchmark(inputs)
    assert not list(inputs.output_dir.glob("metrics_passed_only.*"))


# --- Files ---

EXPECTED_TABLES = ["confirmation_scores_raw", "metrics_overall", "metrics_by_k", "metrics_by_type",
                   "metrics_by_type_eps", "clean_false_positives", "preflagged_cases", "false_alarms",
                   "missed_detections"]


def test_every_table_written_as_csv_and_tex(bench):
    assert sorted(bench.tables) == sorted(EXPECTED_TABLES)
    out = bench.inputs.output_dir
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{name}{suffix}" for name in EXPECTED_TABLES
                                                           for suffix in (".csv", ".tex"))
    for name, table in bench.tables.items():
        csv_bytes = (out / f"{name}.csv").read_bytes()
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert b"\r\n" not in csv_bytes and "\r\n" not in tex
        assert "\\begin{tabular}" in tex or "\\begin{longtable}" in tex
        read_back = pd.read_csv(out / f"{name}.csv", float_precision="round_trip", keep_default_na=False,
                                na_values=["NaN"])
        expected = table.copy()
        if "false_alarm_cases" in expected:  # An empty list of cases is an empty string, not NaN
            read_back["false_alarm_cases"] = read_back["false_alarm_cases"].astype(str)
        pd.testing.assert_frame_equal(read_back, expected, check_dtype=False)


def test_raw_table_columns(bench):
    fields = bench.tables["confirmation_scores_raw"]
    scores = list(SCORE_DEFINITIONS)
    assert list(fields.columns) == [
        "case_id", "k", "perturbation", "epsilon", "label", *PHS_COMPONENT_NAMES,
        *(f"{c}_bar" for c in PHS_COMPONENT_NAMES), *scores, *(f"flagged_{s}" for s in scores),
        "passed_acceptance", *(f"preflagged_{s}" for s in scores)]
    # Each case: its clean field first, then the damage types and strengths
    assert list(fields["perturbation"][:5]) == ["none", "velocity_divergence", "velocity_divergence", "pressure", "pressure"]


def test_tex_cells(bench):
    tex = (bench.inputs.output_dir / "metrics_overall.tex").read_text(encoding="utf-8")
    row = row_of(bench.tables["metrics_overall"], score=S1)
    assert f"{row['recall']:.3f} [{row['recall_low']:.3f}, {row['recall_high']:.3f}]" in tex
    assert f"2/6 [{row['fpr_low']:.3f}, {row['fpr_high']:.3f}]" in tex
    assert r"Score1\_momentum\_only" in tex and " -- " in tex  # Score3's undefined precision


# --- Refused inputs ---

def test_refuses_scores_of_another_calibration(tmp_path):
    inputs = make_inputs(tmp_path)
    calib = frozen_calibration.load(inputs.calibration)
    calib["thresholds"][S1] = 1.2  # Would flag different fields than the scored table says
    frozen_calibration.save(calib, inputs.calibration)
    with pytest.raises(ValueError, match="frozen calibration"):
        run_benchmark(inputs)


def test_refuses_a_changed_score(tmp_path):
    inputs = make_inputs(tmp_path)
    scores = pd.read_csv(inputs.scores, float_precision="round_trip")
    scores.loc[0, S3] += 1e-9
    scores.to_csv(inputs.scores, index=False)
    with pytest.raises(ValueError, match=S3):
        run_benchmark(inputs)


@pytest.mark.parametrize("change", ["drop_clean", "drop_damaged", "duplicate"])
def test_refuses_a_broken_layout(tmp_path, change):
    inputs = make_inputs(tmp_path)
    scores = pd.read_csv(inputs.scores, float_precision="round_trip")
    if change == "drop_clean":
        scores = scores[~((scores["case_id"] == "case_31") & (scores["label"] == "clean"))]
    elif change == "drop_damaged":
        scores = scores.drop(index=0)
    else:
        scores = pd.concat([scores, scores.iloc[[0]]])
    scores.to_csv(inputs.scores, index=False)
    with pytest.raises(ValueError):
        run_benchmark(inputs)


@pytest.mark.parametrize("change", ["missing_case", "extra_case", "wrong_k"])
def test_refuses_scores_that_dont_cover_the_case_file(tmp_path, change):
    inputs = make_inputs(tmp_path)
    cases = dict(CASES)
    if change == "missing_case":  # As from a --case_id run: case_51 is in the case file but not scored
        scores = pd.read_csv(inputs.scores, float_precision="round_trip")
        scores[scores["case_id"] != "case_51"].to_csv(inputs.scores, index=False)
    elif change == "extra_case":  # case_51 is scored but not in the case file
        del cases["case_51"]
    else:  # case_51 is scored with k = 3, but the case file says k = 2
        cases["case_51"] = (2, None, None)
    write_case_file(inputs.metadata, cases)
    with pytest.raises(ValueError, match="case file"):
        run_benchmark(inputs)


def test_committed_results_cover_the_case_file():
    repo = Path(__file__).parent.parent
    fields = pd.read_csv(repo / "results" / "benchmark" / "confirmation_scores_raw.csv")
    bm.check_cases(fields, generate_hallucinations.load_case_metadata(bm.DEFAULT_METADATA))
    assert Path(bm.parse_args([]).metadata) == repo.resolve() / "cases" / "confirmation_cases.json"


@pytest.mark.parametrize("sidecar_change", [{"status": None}, {"metrics": {"passed": False}}, {"k": 2}])
def test_refuses_a_bad_sidecar(tmp_path, sidecar_change):
    inputs = make_inputs(tmp_path)
    path = inputs.models_dir / "case_30" / "lbfgs_final.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **sidecar_change}))
    with pytest.raises(ValueError):
        run_benchmark(inputs)


# --- The path flags of the benchmark run (how the scripts read the new cases and models is in
# tests/test_new_case_inputs.py) ---

@pytest.mark.parametrize("module, flags", [
    (generate_hallucinations, ["metadata", "models_dir", "tensors_dir", "output_dir"]),
    (evaluate_phs, ["metadata", "models_dir", "hallucinations_dir", "scores_dir"]),
])
def test_path_flags_default_to_todays_paths(monkeypatch, module, flags):
    monkeypatch.setattr(sys, "argv", ["script.py"])
    defaults = module.parse_args()
    assert all(getattr(defaults, flag) is None for flag in flags)  # None: today's path
    monkeypatch.setattr(sys, "argv", ["script.py", *(f"--{flag}={flag}_value" for flag in flags)])
    given = module.parse_args()
    assert all(getattr(given, flag) == f"{flag}_value" for flag in flags)
