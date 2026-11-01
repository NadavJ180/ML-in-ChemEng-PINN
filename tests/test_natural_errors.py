"""
Unit tests for src.detection.natural_errors (#39).

On a tiny made-up table of checkpoints (4 cases, 2 per k, each at 3 stages; worked
out by hand below), verifies that:
  - the join with the true errors keeps every checkpoint once, in order, and refuses a
    missing, extra or doubled checkpoint, a different k, adam_step or passed_acceptance,
    and an inaccurate column that is not relL2_uv > 0.10 (exactly 0.10 is accurate);
  - the counts by stage x k (inaccurate, flagged, missed, false alarms) and the two
    rates are right, a score equal to tau is not flagged, one stage gets the exact
    range and all stages together the case bootstrap of grouped_eval.py;
  - failed runs stay in and are counted;
  - every missed failure and every false alarm is listed, per score, and an inaccurate
    final model is listed apart;
  - the consistency check with #35 passes within 1e-9 (relative), reports the largest
    difference, and fails on a larger difference, a different flag, a missing case or
    another k;
  - every table is written as .csv and .tex and reads back exactly, and a
    checkpoint_scores.csv that doesn't follow from its raw components is refused;
  - the field plots are chosen by the documented rule.
And, on the real checkpoints and the committed results:
  - the scoring settings are PROTOCOL.md section 4's, read from evaluate_phs.py, and
    each case's scoring seed is evaluate_phs.py's;
  - scoring case_30's lbfgs_final gives #35's clean field, and scoring one of its Adam
    checkpoints gives the committed row;
  - a run folder with a missing checkpoint, a checkpoint of another case or the old
    layout is refused before any scoring;
  - the committed results/natural_errors tables cover the 210 checkpoints and follow
    from their raw components, the frozen calibration and the true errors.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection import frozen_calibration
from src.detection import natural_errors as ne
from src.detection.grouped_eval import case_bootstrap, clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.models.accuracy_audit import read_cases
from src.models.checkpoints import CASE_FIELDS, STAGES, load_checkpoint

REPO = Path(__file__).parent.parent
N_BOOT = 100  # Enough to check the wiring; the real run uses ne.N_BOOT = 2000

S1, S2, S3, S4, S5 = SCORE_DEFINITIONS
# Score1 flags above 1, Score2 above 2, and the other three flag nothing
THRESHOLDS = {S1: 1.0, S2: 2.0, S3: 100.0, S4: 100.0, S5: 100.0}

TINY_STAGES = ("adam_00100", "adam_05000", "lbfgs_final")
ADAM_STEP = {"adam_00100": 100, "adam_05000": 5000, "lbfgs_final": 5000}
K = {"case_30": 1, "case_31": 1, "case_40": 2, "case_41": 2}
# case_id: (mom, relL2_uv) at each of TINY_STAGES. Only mom is nonzero and the normalizers
# are 1, so every score of a checkpoint equals its mom; only tau differs between the scores.
CASES = {
    "case_30": [(5.0, 0.50), (1.5, 0.08), (0.5, 0.003)],
    "case_31": [(0.9, 0.40), (1.0, 0.12), (2.5, 0.004)],  # 1.0 equals Score1's tau, so it is not flagged
    "case_40": [(3.0, 0.60), (0.7, 0.10), (0.2, 0.002)],  # 0.10 is not above 0.10, so it is accurate
    "case_41": [(2.0, 0.30), (1.2, 0.15), (0.4, 0.20)],   # 2.0 equals Score2's tau; an inaccurate final model
}

# Worked out by hand. 7 checkpoints are inaccurate: case_30 and case_31 at adam_00100,
# case_31 at adam_05000, case_40 at adam_00100, and case_41 at all 3 stages.
# Score1 (tau 1) flags 6: case_30 at adam_00100 and adam_05000, case_31 at lbfgs_final,
# case_40 at adam_00100, case_41 at adam_00100 and adam_05000. It misses case_31 at
# adam_00100 and adam_05000 and case_41 at lbfgs_final, and its false alarms are case_30
# at adam_05000 and case_31 at lbfgs_final.
# (stage, k): (n_checkpoints, n_inaccurate, n_flagged, n_missed, n_false_alarms)
SCORE1_BY_STAGE_K = {
    ("adam_00100", "1"): (2, 2, 1, 1, 0),
    ("adam_00100", "2"): (2, 2, 2, 0, 0),
    ("adam_00100", "all"): (4, 4, 3, 1, 0),
    ("adam_05000", "1"): (2, 1, 1, 1, 1),
    ("adam_05000", "2"): (2, 1, 1, 0, 0),
    ("adam_05000", "all"): (4, 2, 2, 1, 1),
    ("lbfgs_final", "1"): (2, 0, 1, 0, 1),
    ("lbfgs_final", "2"): (2, 1, 0, 1, 0),
    ("lbfgs_final", "all"): (4, 1, 1, 1, 1),
    ("all", "1"): (6, 3, 3, 2, 2),
    ("all", "2"): (6, 4, 3, 1, 0),
    ("all", "all"): (12, 7, 6, 3, 2),
}
# All stages and all k: (n_flagged, n_missed, n_false_alarms). Score2 (tau 2) flags case_30 and
# case_40 at adam_00100 and case_31 at lbfgs_final; Score3 to Score5 flag nothing.
POOLED = {S1: (6, 3, 2), S2: (3, 5, 1), S3: (0, 7, 0), S4: (0, 7, 0), S5: (0, 7, 0)}
MISSED = {
    S1: [("case_31", "adam_00100"), ("case_31", "adam_05000"), ("case_41", "lbfgs_final")],
    S2: [("case_31", "adam_00100"), ("case_31", "adam_05000"), ("case_41", "adam_00100"), ("case_41", "adam_05000"),
         ("case_41", "lbfgs_final")],
}
FALSE_ALARMS = {S1: [("case_30", "adam_05000"), ("case_31", "lbfgs_final")], S2: [("case_31", "lbfgs_final")]}


def make_calibration() -> dict:
    """A frozen calibration whose normalizers are 1 (and eps 0), so each normalized component is its raw value."""
    return {
        "format_version": frozen_calibration.FORMAT_VERSION,
        "description": "Made-up calibration of tests/test_natural_errors.py",
        "normalizers": {component: 1.0 for component in PHS_COMPONENT_NAMES},
        "normalizer_eps": 0.0,
        "scores": {name: list(components) for name, components in SCORE_DEFINITIONS.items()},
        "thresholds": dict(THRESHOLDS),
    }


def make_raw(failed=()) -> pd.DataFrame:
    """The made-up checkpoints as score_checkpoints() returns them; the runs of `failed` failed."""
    rows = []
    for case_id, values in CASES.items():
        for stage, (mom, _) in zip(TINY_STAGES, values):
            rows.append({"case_id": case_id, "k": K[case_id], "stage": stage, "adam_step": ADAM_STEP[stage],
                         "passed_acceptance": case_id not in failed, "mom": mom, "div": 0.0, "bc": 0.0, "E": 0.0})
    return pd.DataFrame(rows)


def make_errors(failed=()) -> pd.DataFrame:
    """The made-up true errors, laid out like results/accuracy/reference_errors_confirmation.csv."""
    rows = []
    for case_id, values in CASES.items():
        for stage, (_, error) in zip(TINY_STAGES, values):
            rows.append({"case_id": case_id, "k": K[case_id], "split": "confirmation", "stage": stage,
                         "adam_step": ADAM_STEP[stage], "relL2_uv": error, "relL2_p": 2 * error, "relL2_uv_T": error,
                         "passed_acceptance": case_id not in failed, "inaccurate": error > 0.10})
    return pd.DataFrame(rows)


def make_table(failed=()) -> pd.DataFrame:
    return ne.build_checkpoint_table(make_raw(failed), make_calibration(), make_errors(failed))


def make_benchmark(table: pd.DataFrame) -> pd.DataFrame:
    """#35's table for the made-up cases: each lbfgs_final field as a clean row, plus a damaged row per case."""
    final = table[table["stage"] == "lbfgs_final"]
    columns = ["case_id", "k", *PHS_COMPONENT_NAMES, *ne.BAR_NAMES, *ne.SCORE_NAMES, *ne.FLAG_NAMES]
    clean = final[columns].assign(perturbation="none", epsilon=0.0, label="clean")
    damaged = clean.assign(perturbation="pressure", epsilon=0.05, label="hallucinated", mom=99.0)
    return pd.concat([clean, damaged], ignore_index=True)


@pytest.fixture(scope="module")
def tiny():
    """The made-up checkpoints with every run passed: the checkpoint table and every output table."""
    table = make_table()
    return table, ne.make_tables(table, THRESHOLDS, n_boot=N_BOOT)


def row_of(table: pd.DataFrame, **keys) -> pd.Series:
    """The single row of table matching keys."""
    mask = np.ones(len(table), dtype=bool)
    for column, value in keys.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, f"{mask.sum()} rows match {keys}"
    return table[mask].iloc[0]


def assert_rate(actual, expected):
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == pytest.approx(expected, abs=1e-12)


# --- The join with the true errors ---

def test_join_adds_the_true_errors_in_order(tiny):
    table, _ = tiny
    assert list(table.columns) == list(ne.CHECKPOINT_COLUMNS)
    assert list(zip(table["case_id"], table["stage"])) == [(c, s) for c in CASES for s in TINY_STAGES]
    assert list(table["relL2_uv"]) == [error for values in CASES.values() for _, error in values]
    assert int(table["inaccurate"].sum()) == 7
    # Exactly 0.10 is not above 0.10
    assert not row_of(table, case_id="case_40", stage="adam_05000")["inaccurate"]
    # The flags are the frozen calibration's: score > tau, so a score equal to tau is not flagged
    assert not row_of(table, case_id="case_31", stage="adam_05000")[f"flagged_{S1}"]
    assert not row_of(table, case_id="case_41", stage="adam_00100")[f"flagged_{S2}"]
    assert row_of(table, case_id="case_30", stage="adam_05000")[f"flagged_{S1}"]


def _drop_one(errors):
    return errors.drop(index=3)


def _add_one(errors):
    return pd.concat([errors, errors.iloc[[0]].assign(case_id="case_99")], ignore_index=True)


def _double_one(errors):
    return pd.concat([errors, errors.iloc[[0]]], ignore_index=True)


def _other_k(errors):
    return errors.assign(k=errors["k"].where(errors["case_id"] != "case_30", 3))


def _other_adam_step(errors):
    return errors.assign(adam_step=errors["adam_step"].where(errors["stage"] != "adam_05000", 4000))


def _other_status(errors):
    return errors.assign(passed_acceptance=errors["case_id"] != "case_41")


def _wrong_inaccurate(errors):
    return errors.assign(inaccurate=errors["relL2_uv"] >= 0.10)  # Calls 0.10 inaccurate


@pytest.mark.parametrize("change", [_drop_one, _add_one, _double_one, _other_k, _other_adam_step, _other_status,
                                    _wrong_inaccurate])
def test_join_refuses_true_errors_that_dont_match(change):
    with pytest.raises(ValueError):
        ne.build_checkpoint_table(make_raw(), make_calibration(), change(make_errors()))


def test_join_refuses_a_checkpoint_scored_twice():
    raw = make_raw()
    with pytest.raises(ValueError, match="twice"):
        ne.build_checkpoint_table(pd.concat([raw, raw.iloc[[0]]], ignore_index=True), make_calibration(),
                                  make_errors())


def test_join_refuses_true_errors_without_a_column():
    with pytest.raises(ValueError, match="relL2_uv"):
        ne.join_errors(make_raw(), make_errors().drop(columns="relL2_uv"))


def test_check_coverage():
    cases = [{"case_id": "case_30", "k": 1}, {"case_id": "case_40", "k": 2}]
    table = pd.DataFrame([{"case_id": c["case_id"], "k": c["k"], "stage": s} for c in cases for s in STAGES])
    ne.check_coverage(table, cases)
    with pytest.raises(ValueError, match="missing"):
        ne.check_coverage(table.drop(index=2), cases)
    with pytest.raises(ValueError, match="in order"):
        ne.check_coverage(table.iloc[::-1].reset_index(drop=True), cases)
    with pytest.raises(ValueError, match="k differs"):
        ne.check_coverage(table.assign(k=1), cases)


# --- Counts and rates by stage x k ---

@pytest.mark.parametrize("stage, k", list(SCORE1_BY_STAGE_K))
def test_score1_counts_by_hand(tiny, stage, k):
    row = row_of(tiny[1]["table_stage_by_k"], score=S1, stage=stage, k=k)
    n, n_inaccurate, n_flagged, n_missed, n_false_alarms = SCORE1_BY_STAGE_K[(stage, k)]
    assert (row["n_checkpoints"], row["n_inaccurate"], row["n_flagged"], row["n_missed"], row["n_false_alarms"]) == \
        (n, n_inaccurate, n_flagged, n_missed, n_false_alarms)
    assert row["n_cases"] == (4 if k == "all" else 2)
    assert row["tau"] == THRESHOLDS[S1]
    n_accurate = n - n_inaccurate
    assert_rate(row["recall"], (n_inaccurate - n_missed) / n_inaccurate if n_inaccurate else np.nan)
    assert_rate(row["fpr"], n_false_alarms / n_accurate if n_accurate else np.nan)


@pytest.mark.parametrize("score", list(POOLED))
def test_pooled_counts_of_every_score(tiny, score):
    row = row_of(tiny[1]["table_stage_by_k"], score=score, stage="all", k="all")
    assert (row["n_flagged"], row["n_missed"], row["n_false_alarms"]) == POOLED[score]
    assert row["n_inaccurate"] == 7 and row["n_checkpoints"] == 12


def test_table_layout(tiny):
    table = tiny[1]["table_stage_by_k"]
    assert list(table.columns) == list(ne.STAGE_BY_K_COLUMNS)
    # Every score x (3 stages + all) x (2 k + all), scores in order, stages in training order
    assert len(table) == 5 * 4 * 3
    assert list(table["score"].unique()) == list(ne.SCORE_NAMES)
    assert list(table["stage"].unique()) == [*TINY_STAGES, "all"]
    assert list(table["k"].unique()) == ["1", "2", "all"]


def test_one_stage_gets_the_exact_range(tiny):
    table = tiny[1]["table_stage_by_k"]
    for stage, k in SCORE1_BY_STAGE_K:
        if stage == "all":
            continue
        row = row_of(table, score=S1, stage=stage, k=k)
        n, n_inaccurate, _, n_missed, n_false_alarms = SCORE1_BY_STAGE_K[(stage, k)]
        assert row["range_method"] == "clopper_pearson"
        assert pd.isna(row["recall_n_boot"]) and pd.isna(row["fpr_n_boot"])
        for rate, count, total in (("recall", n_inaccurate - n_missed, n_inaccurate),
                                   ("fpr", n_false_alarms, n - n_inaccurate)):
            if total:
                assert (row[f"{rate}_low"], row[f"{rate}_high"]) == clopper_pearson(count, total)
            else:  # Nothing to count: NaN, never 0
                assert np.isnan(row[rate]) and np.isnan(row[f"{rate}_low"]) and np.isnan(row[f"{rate}_high"])


@pytest.mark.parametrize("k", ["1", "2", "all"])
def test_all_stages_get_the_case_bootstrap(tiny, k):
    table, tables = tiny
    row = row_of(tables["table_stage_by_k"], score=S1, stage="all", k=k)
    group = table if k == "all" else table[table["k"] == int(k)]
    assert row["range_method"] == "case_bootstrap"
    for rate, function in (("recall", ne.recall_rate), ("fpr", ne.false_alarm_rate)):
        estimate, low, high, used = case_bootstrap(ne._view(group, S1), function, n_boot=N_BOOT, seed=ne.BOOT_SEED)
        assert row[rate] == estimate
        assert (row[f"{rate}_low"], row[f"{rate}_high"], row[f"{rate}_n_boot"]) == (low, high, used)
    # The resamples keep whole cases within each k: 10 of 10 would be 2 of 2 here
    assert row["recall_low"] <= row["recall"] <= row["recall_high"]


def test_failed_run_is_kept_and_counted():
    table = make_table(failed=("case_41",))
    tables = ne.make_tables(table, THRESHOLDS, n_boot=N_BOOT)
    by_stage = tables["table_stage_by_k"]
    assert row_of(by_stage, score=S1, stage="all", k="all")["n_failed_runs"] == 1
    assert row_of(by_stage, score=S1, stage="lbfgs_final", k="2")["n_failed_runs"] == 1
    assert row_of(by_stage, score=S1, stage="lbfgs_final", k="1")["n_failed_runs"] == 0
    # Its checkpoints are still counted and listed, marked as a failed run
    assert row_of(by_stage, score=S1, stage="all", k="all")["n_checkpoints"] == 12
    missed = tables["missed_failures"]
    assert not missed.loc[missed["case_id"] == "case_41", "passed_acceptance"].any()
    assert missed.loc[missed["case_id"] != "case_41", "passed_acceptance"].all()
    assert not tables["inaccurate_final_models"]["passed_acceptance"].any()


# --- The lists ---

@pytest.mark.parametrize("score", [S1, S2, S3])
def test_every_missed_failure_listed(tiny, score):
    table, tables = tiny
    listed = tables["missed_failures"]
    listed = listed[listed["score"] == score]
    expected = MISSED.get(score, [(c, s) for c, s, bad in zip(table["case_id"], table["stage"], table["inaccurate"])
                                  if bad])
    assert list(zip(listed["case_id"], listed["stage"])) == expected
    assert (listed["relL2_uv"] > 0.10).all() and (listed["score_value"] <= THRESHOLDS[score]).all()
    assert list(listed["score_over_tau"]) == list(listed["score_value"] / THRESHOLDS[score])


@pytest.mark.parametrize("score", [S1, S2, S3])
def test_every_false_alarm_listed(tiny, score):
    listed = tiny[1]["false_alarms"]
    listed = listed[listed["score"] == score]
    assert list(zip(listed["case_id"], listed["stage"])) == FALSE_ALARMS.get(score, [])
    assert (listed["relL2_uv"] <= 0.10).all() and (listed["score_value"] > THRESHOLDS[score]).all()


def test_list_columns(tiny):
    tables = tiny[1]
    for name in ("missed_failures", "false_alarms"):
        assert list(tables[name].columns) == list(ne.LIST_COLUMNS)
    assert len(tables["missed_failures"]) == 3 + 5 + 3 * 7
    assert len(tables["false_alarms"]) == 2 + 1


def test_inaccurate_final_models_listed_apart(tiny):
    final = tiny[1]["inaccurate_final_models"]
    assert list(final.columns) == list(ne.FINAL_COLUMNS)
    assert list(zip(final["case_id"], final["stage"])) == [("case_41", "lbfgs_final")]
    assert final["relL2_uv"].iloc[0] == 0.20
    assert not final[list(ne.FLAG_NAMES)].iloc[0].any()  # Missed by every score


def test_empty_lists_keep_their_columns():
    # No inaccurate checkpoint at all
    errors = make_errors().assign(relL2_uv=0.01, inaccurate=False)
    table = ne.build_checkpoint_table(make_raw(), make_calibration(), errors)
    tables = ne.make_tables(table, THRESHOLDS, n_boot=N_BOOT)
    assert tables["missed_failures"].empty and list(tables["missed_failures"].columns) == list(ne.LIST_COLUMNS)
    assert tables["inaccurate_final_models"].empty
    assert "There is none" in ne.to_latex("inaccurate_final_models", tables["inaccurate_final_models"])
    assert np.isnan(row_of(tables["table_stage_by_k"], score=S1, stage="all", k="all")["recall"])


# --- The consistency check with #35 ---

def test_consistency_check_passes_on_the_same_fields(tiny):
    table, _ = tiny
    result = ne.consistency_check(table, make_benchmark(table))
    assert result["n_fields"] == 4
    assert result["max_rel_diff"] == 0.0


def test_consistency_check_allows_last_digit_differences(tiny):
    table, _ = tiny
    benchmark = make_benchmark(table)
    # 2**-42 (about 2.3e-13, the size of the Linux-Windows differences) is exact in floating point
    benchmark.loc[benchmark["case_id"] == "case_31", "mom"] *= 1 + 2.0 ** -42
    result = ne.consistency_check(table, benchmark)
    assert result["max_rel_diff"] == pytest.approx(2.0 ** -42, rel=1e-6)
    assert (result["column"], result["case_id"]) == ("mom", "case_31")


def _bigger_difference(benchmark):
    benchmark.loc[benchmark["case_id"] == "case_31", "E"] = 1e-8  # Ours is 0
    benchmark.loc[benchmark["case_id"] == "case_30", "mom"] *= 1 + 1e-8
    return benchmark


def _other_flag(benchmark):
    flag = f"flagged_{S2}"
    benchmark.loc[benchmark["case_id"] == "case_30", flag] = ~benchmark.loc[benchmark["case_id"] == "case_30", flag]
    return benchmark


def _missing_case(benchmark):
    return benchmark[~((benchmark["case_id"] == "case_40") & (benchmark["label"] == "clean"))]


def _benchmark_other_k(benchmark):
    benchmark.loc[benchmark["case_id"] == "case_40", "k"] = 3
    return benchmark


def _missing_value(benchmark):
    benchmark.loc[benchmark["case_id"] == "case_40", "bc"] = np.nan
    return benchmark


@pytest.mark.parametrize("change", [_bigger_difference, _other_flag, _missing_case, _benchmark_other_k,
                                    _missing_value])
def test_consistency_check_fails(tiny, change):
    table, _ = tiny
    with pytest.raises(ValueError):
        ne.consistency_check(table, change(make_benchmark(table)))


# --- Writing and reading the tables ---

def test_every_table_written_as_csv_and_tex(tmp_path, tiny):
    table, tables = tiny
    written = ne.write_tables(tables, tmp_path)
    names = [ne.SCORES_TABLE, "table_stage_by_k", "missed_failures", "false_alarms", "inaccurate_final_models"]
    assert set(written) == {tmp_path / f"{name}.{suffix}" for name in names for suffix in ("csv", "tex")}
    for name in names:
        tex = (tmp_path / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\begin{" in tex and f"tab:natural_errors_{name}" in tex
        assert b"\r\n" not in (tmp_path / f"{name}.csv").read_bytes()
    back = pd.read_csv(tmp_path / f"{ne.SCORES_TABLE}.csv", float_precision="round_trip")
    for column in ne.CHECKPOINT_COLUMNS:
        assert np.array_equal(back[column].to_numpy(), table[column].to_numpy()), column


def test_tex_cells(tiny):
    tex = ne.to_latex("table_stage_by_k", tiny[1]["table_stage_by_k"])
    low, high = clopper_pearson(1, 2)  # Score1 at adam_00100, k = 1: 1 of 2 inaccurate flagged
    assert f"0.500 [{low:.3f}, {high:.3f}]" in tex
    assert "--" in tex  # The false-alarm rate where every checkpoint is inaccurate
    scores_tex = ne.to_latex(ne.SCORES_TABLE, tiny[0])
    assert "0.00300" in scores_tex  # relL2_uv in significant digits, not 0.003 -> "0.003"


def test_significant_digits():
    assert ne._significant(0.0015387, 3) == "0.00154"
    assert ne._significant(0.4624, 3) == "0.462"
    assert ne._significant(12345.678, 4) == "12346"
    assert ne._significant(1.5858374, 4) == "1.586"
    assert ne._significant(2.5e-7, 3) == "2.50e-07"
    assert ne._significant(np.nan, 3) == "--"


def test_reads_back_scores_that_follow_from_their_raw_components(tmp_path, tiny):
    table, tables = tiny
    ne.write_tables(tables, tmp_path)
    back = ne.read_checkpoint_scores(tmp_path / f"{ne.SCORES_TABLE}.csv", make_calibration(), make_errors())
    pd.testing.assert_frame_equal(back, table)


@pytest.mark.parametrize("column, value", [(S3, 50.0), (f"flagged_{S1}", False), ("relL2_uv", 0.09),
                                           ("inaccurate", False)])
def test_refuses_scores_that_dont_follow_from_their_raw_components(tmp_path, tiny, column, value):
    table, tables = tiny
    ne.write_tables(tables, tmp_path)
    path = tmp_path / f"{ne.SCORES_TABLE}.csv"
    edited = pd.read_csv(path, float_precision="round_trip")
    edited.loc[0, column] = value  # Row 0: case_30 at adam_00100
    edited.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ne.read_checkpoint_scores(path, make_calibration(), make_errors())


def test_refuses_scores_of_another_calibration(tmp_path, tiny):
    _, tables = tiny
    ne.write_tables(tables, tmp_path)
    other = make_calibration()
    other["thresholds"][S1] = 0.5
    with pytest.raises(ValueError, match="flagged"):
        ne.read_checkpoint_scores(tmp_path / f"{ne.SCORES_TABLE}.csv", other, make_errors())


def test_scatter_plot_written(tmp_path):
    table = make_table(failed=("case_41",))
    written = ne.plot_score_vs_error(table, THRESHOLDS, tmp_path)
    assert written == [tmp_path / "fig_score_vs_error.pdf", tmp_path / "fig_score_vs_error.png"]
    assert all(path.stat().st_size > 0 for path in written)


# --- Choosing the field plots ---

def test_field_plots_worst_miss_of_each_score_when_many(tiny):
    # Score3 to Score5 miss all 7 inaccurate checkpoints, more than MAX_MISSED_FIELD_PLOTS (6)
    chosen = ne.choose_field_checkpoints(tiny[0])
    assert chosen == [
        (("case_31", "adam_00100"), ["worst miss of Score1", "worst miss of Score2"]),
        (("case_40", "adam_00100"), ["worst miss of Score3", "worst miss of Score4", "worst miss of Score5"]),
        # Final errors 0.002, 0.003, 0.004, 0.20: the lower middle one
        (("case_30", "lbfgs_final"), ["typical final model"]),
    ]


def test_field_plots_every_miss_when_few(tiny):
    chosen = dict(ne.choose_field_checkpoints(tiny[0], max_missed=7))
    assert chosen[("case_30", "adam_00100")] == ["missed by Score3, Score4, Score5"]
    assert chosen[("case_31", "adam_00100")] == ["missed by Score1, Score2, Score3, Score4, Score5"]
    assert len([key for key, reasons in chosen.items() if reasons[0].startswith("missed by")]) == 7


def test_field_plots_catches_and_false_alarms_of_phs(tiny):
    table = tiny[0].copy()
    table[f"flagged_{S3}"] = table[S3] > 1.0  # As if PHS's tau were 1
    chosen = dict(ne.choose_field_checkpoints(table))
    # PHS catches case_41 at adam_05000 (error 0.15, score 1.2), case_41 at adam_00100 (0.30, 2.0),
    # case_30 at adam_00100 (0.50, 5.0) and case_40 at adam_00100 (0.60, 3.0)
    assert chosen[("case_41", "adam_00100")] == ["typical failure caught by PHS"]
    assert chosen[("case_41", "adam_05000")] == ["failure closest to being missed by PHS"]
    # Its false alarms are case_30 at adam_05000 and case_31 at lbfgs_final
    assert chosen[("case_30", "adam_05000")] == ["typical false alarm of PHS on an Adam checkpoint"]
    assert chosen[("case_31", "lbfgs_final")] == ["highest false alarm of PHS on a final model"]


# --- The real checkpoints ---

CONFIRMATION_CASES = REPO / "cases" / "confirmation_cases.json"
CONFIRMATION_MODELS = REPO / "models" / "confirmation"
OUTPUT_DIR = REPO / "results" / "natural_errors"


def test_scoring_settings_are_the_protocols():
    argv = list(sys.argv)
    settings = ne.scoring_settings("cpu")
    assert sys.argv == argv
    # PROTOCOL.md section 4
    assert (settings.n_interior, settings.n_bc, settings.n_time, settings.energy_res, settings.chunk_size,
            settings.bc_local_band_width_fraction, settings.bc_local_n_points) == (20000, 1000, 20, 32, 8000, 0.05,
                                                                                   20000)
    assert not settings.no_seed_points and settings.device == "cpu"


def test_scoring_seed_is_evaluate_phs_seed():
    for case in read_cases(CONFIRMATION_CASES):
        assert ne.scoring_seed(case) == 1000 + int(case["case_id"].split("_")[1]) == case["seed_scoring"]
    with pytest.raises(ValueError, match="seed"):
        ne.scoring_seed({"case_id": "case_30", "seed_scoring": 1031})


def _case(case_id):
    return next(case for case in read_cases(CONFIRMATION_CASES) if case["case_id"] == case_id)


def test_final_checkpoint_scores_as_in_the_benchmark():
    case = _case("case_30")
    model, _ = load_checkpoint(CONFIRMATION_MODELS / "case_30" / "lbfgs_final.pth", k=case["k"])
    components = ne.score_checkpoint(model, case, ne.scoring_settings("cpu"))
    benchmark = ne.read_benchmark(ne.DEFAULT_BENCHMARK_CSV)
    clean = row_of(benchmark, case_id="case_30", label="clean")
    for name in PHS_COMPONENT_NAMES:  # Bit for bit on the same machine; last digits elsewhere
        assert components[name] == pytest.approx(clean[name], rel=1e-9, abs=0), name


def test_committed_scores_follow_from_the_models():
    case = _case("case_30")
    scores = pd.read_csv(OUTPUT_DIR / f"{ne.SCORES_TABLE}.csv", float_precision="round_trip")
    row = row_of(scores, case_id="case_30", stage="adam_00100")
    model, _ = load_checkpoint(CONFIRMATION_MODELS / "case_30" / "adam_00100.pth", k=case["k"])
    components = ne.score_checkpoint(model, case, ne.scoring_settings("cpu"))
    for name in PHS_COMPONENT_NAMES:
        assert components[name] == pytest.approx(row[name], rel=1e-9, abs=0), name


def test_field_data_gives_the_true_error(tmp_path):
    case = _case("case_30")
    model, _ = load_checkpoint(CONFIRMATION_MODELS / "case_30" / "adam_00100.pth", k=case["k"])
    data = ne.field_data(model, case)
    errors = ne.read_true_errors(ne.DEFAULT_ERRORS_CSV)
    assert data["relL2_uv"] == pytest.approx(row_of(errors, case_id="case_30", stage="adam_00100")["relL2_uv"],
                                             rel=1e-9, abs=0)
    assert len(data["t"]) == 11 and data["fields"]["u"][0].shape == (64, 64)
    table = pd.read_csv(OUTPUT_DIR / f"{ne.SCORES_TABLE}.csv", float_precision="round_trip")
    checkpoint = row_of(table, case_id="case_30", stage="adam_00100")
    calib = frozen_calibration.load(ne.DEFAULT_CALIBRATION_JSON)
    written = ne.plot_fields(checkpoint, ["a test"], data, calib["thresholds"], tmp_path)
    assert written == [tmp_path / "fig_fields_case_30_adam_00100.pdf", tmp_path / "fig_fields_case_30_adam_00100.png"]


# --- Refusing a run folder before scoring ---

CASE = {"case_id": "case_30", "split": "confirmation", "k": 1, "Re": 100.0, "U0": 1.0, "phi_x": 0.5, "phi_y": 1.5,
        "seed_points": 3030, "seed_train": 4030, "seed_scoring": 1030}


def write_run_folder(models_dir: Path, case: dict = CASE) -> Path:
    """Empty .pth files and minimal sidecars for the 7 stages of a case: enough for run_checkpoints()."""
    case_dir = models_dir / case["case_id"]
    case_dir.mkdir(parents=True)
    for stage in STAGES:
        (case_dir / f"{stage}.pth").write_bytes(b"")
        sidecar = {"checkpoint": stage, **{field: case[field] for field in CASE_FIELDS}}
        (case_dir / f"{stage}.json").write_text(json.dumps(sidecar))
    return case_dir


def test_run_checkpoints_reads_a_run_folder(tmp_path):
    case_dir = write_run_folder(tmp_path)
    [(case, checkpoints)] = ne.run_checkpoints([CASE], tmp_path)
    assert case == CASE
    assert checkpoints == [(stage, case_dir / f"{stage}.pth") for stage in STAGES]


def test_run_checkpoints_refuses_a_missing_checkpoint(tmp_path):
    case_dir = write_run_folder(tmp_path)
    (case_dir / "adam_02500.pth").unlink()
    with pytest.raises(FileNotFoundError, match="adam_02500"):
        ne.run_checkpoints([CASE], tmp_path)


def test_run_checkpoints_refuses_a_checkpoint_of_another_case(tmp_path):
    case_dir = write_run_folder(tmp_path)
    path = case_dir / "adam_04000.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "k": 2}))
    with pytest.raises(ValueError, match="does not match"):
        ne.run_checkpoints([CASE], tmp_path)


def test_run_checkpoints_refuses_the_old_layout(tmp_path):
    (tmp_path / "case_30.pth").write_bytes(b"")
    with pytest.raises(ValueError, match="staged checkpoints"):
        ne.run_checkpoints([CASE], tmp_path)


def test_run_checkpoints_refuses_another_scoring_seed(tmp_path):
    case = {**CASE, "seed_scoring": 1031}
    write_run_folder(tmp_path, case)
    with pytest.raises(ValueError, match="seed"):
        ne.run_checkpoints([case], tmp_path)


# --- The committed results ---

def test_committed_results_cover_every_checkpoint():
    calib = frozen_calibration.load(ne.DEFAULT_CALIBRATION_JSON)
    errors = ne.read_true_errors(ne.DEFAULT_ERRORS_CSV)
    # Rebuilding the table from its raw components, the frozen calibration and the true errors gives it back exactly
    table = ne.read_checkpoint_scores(OUTPUT_DIR / f"{ne.SCORES_TABLE}.csv", calib, errors)
    assert len(table) == 210
    ne.check_coverage(table, read_cases(CONFIRMATION_CASES))
    result = ne.consistency_check(table, ne.read_benchmark(ne.DEFAULT_BENCHMARK_CSV))
    assert result["n_fields"] == 30


def test_committed_tables_follow_from_the_scores():
    calib = frozen_calibration.load(ne.DEFAULT_CALIBRATION_JSON)
    table = ne.read_checkpoint_scores(OUTPUT_DIR / f"{ne.SCORES_TABLE}.csv", calib,
                                      ne.read_true_errors(ne.DEFAULT_ERRORS_CSV))
    thresholds = calib["thresholds"]

    def committed(name):
        return pd.read_csv(OUTPUT_DIR / f"{name}.csv", float_precision="round_trip")

    for name, recomputed in (("missed_failures", ne.missed_failures(table, thresholds)),
                             ("false_alarms", ne.false_alarms(table, thresholds)),
                             ("inaccurate_final_models", ne.inaccurate_final_models(table))):
        pd.testing.assert_frame_equal(committed(name), recomputed, check_dtype=False, obj=name)

    # The counts and the exact ranges don't depend on the number of resamples; the bootstrap
    # ranges do, so only their estimates are compared here
    stage_table = committed("table_stage_by_k")
    recomputed = ne.stage_by_k_table(table, thresholds, n_boot=10)
    counts = ["score", "tau", "stage", "k", "n_cases", "n_failed_runs", "n_checkpoints", "n_inaccurate", "n_flagged",
              "n_missed", "n_false_alarms", "recall", "fpr", "range_method"]
    pd.testing.assert_frame_equal(stage_table[counts], recomputed[counts], check_dtype=False)
    exact = (stage_table["range_method"] == "clopper_pearson").to_numpy()
    ranges = ["recall_low", "recall_high", "fpr_low", "fpr_high"]
    pd.testing.assert_frame_equal(stage_table.loc[exact, ranges], recomputed.loc[exact, ranges], check_dtype=False)
    assert (stage_table.loc[~exact, "recall_n_boot"] > 0).all()
    assert set(stage_table.loc[~exact, "stage"]) == {"all"}


def test_committed_figures_exist():
    for stem in ["fig_score_vs_error", *(p.stem for p in OUTPUT_DIR.glob("fig_fields_*.png"))]:
        assert (OUTPUT_DIR / f"{stem}.pdf").exists() and (OUTPUT_DIR / f"{stem}.png").exists(), stem
    assert list(OUTPUT_DIR.glob("fig_fields_*.png"))


def test_path_flags_default_to_the_repo_paths():
    args = ne.parse_args([])
    assert Path(args.metadata) == REPO.resolve() / "cases" / "confirmation_cases.json"
    assert Path(args.models_dir) == REPO.resolve() / "models" / "confirmation"
    assert Path(args.errors) == REPO.resolve() / "results" / "accuracy" / "reference_errors_confirmation.csv"
    assert Path(args.benchmark) == REPO.resolve() / "results" / "benchmark" / "confirmation_scores_raw.csv"
    assert Path(args.output_dir) == REPO.resolve() / "results" / "natural_errors"
    assert not args.from_scores
