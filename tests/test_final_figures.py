"""
Unit tests for src.detection.final_figures (#36).

On a tiny made-up benchmark (6 cases, 2 per k, each with its clean field and 2 damage
types at 2 strengths; worked out by hand below), verifies that:
  - recall by score x k x type x strength counts the flags, over all cases and over the
    cases whose clean field the score doesn't flag, a score equal to tau is not flagged,
    and each row gets the exact range of its count;
  - the case table keeps the case-file order and refuses inputs that disagree: a case
    missing or listed twice, a pass/fail that differs between loss_summary.csv, the true
    errors and the benchmark, or that doesn't follow from the acceptance metrics;
  - the caption helpers find the strengths of full recall, move labels only up, and
    refuse a zero-width range whose cases don't share one count.
And, on the committed results:
  - the inputs agree (the benchmark follows from the frozen calibration, #35's metrics
    match the flags and the scores), and a changed flag, count or AUC is refused;
  - the captions give the numbers #36 asks them to explain, and are plain LaTeX;
  - run() writes every figure and table, a second run gives the same bytes, and the
    committed results/figures tables and captions are what the script writes.
"""

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection import benchmark_metrics as bm
from src.detection import final_figures as ff
from src.detection import frozen_calibration
from src.detection.grouped_eval import clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations.perturbations import PERTURBATION_NAMES
from src.models.accuracy_audit import read_cases

REPO = Path(__file__).parent.parent
OUTPUT_DIR = REPO / "results" / "figures"

S1, S2, S3, S4, S5 = SCORE_DEFINITIONS
# Only mom is nonzero and the normalizers are 1, so every score of a field equals its mom
THRESHOLDS = {S1: 1.0, S2: 2.0, S3: 3.0, S4: 3.0, S5: 100.0}
K = {"case_30": 1, "case_31": 1, "case_40": 2, "case_41": 2, "case_50": 3, "case_51": 3}
FIELDS = (("none", 0.0), ("momentum", 0.0001), ("momentum", 0.01), ("pressure", 0.0001), ("pressure", 0.01))
VALUES = {  # mom of each field of FIELDS
    "case_30": (0.5, 0.6, 5.0, 0.4, 2.5),
    "case_31": (1.5, 1.6, 4.0, 1.4, 3.5),   # Clean above Score1's tau: already flagged by Score1
    "case_40": (0.2, 0.3, 2.5, 0.2, 0.9),
    "case_41": (0.1, 1.0, 1.2, 0.1, 6.0),   # 1.0 equals Score1's tau, so it is not flagged
    "case_50": (0.3, 0.3, 0.8, 0.3, 1.1),
    "case_51": (2.5, 2.6, 9.0, 2.4, 3.0),   # Already flagged by Score1 and Score2
}
# Worked out by hand: (k, type, strength): (flagged, out of; new flagged, out of)
BY_HAND = {
    S1: {(1, "momentum", 0.0001): (1, 2, 0, 1), (1, "momentum", 0.01): (2, 2, 1, 1),
         (1, "pressure", 0.0001): (1, 2, 0, 1), (1, "pressure", 0.01): (2, 2, 1, 1),
         (2, "momentum", 0.0001): (0, 2, 0, 2), (2, "momentum", 0.01): (2, 2, 2, 2),
         (2, "pressure", 0.0001): (0, 2, 0, 2), (2, "pressure", 0.01): (1, 2, 1, 2),
         (3, "momentum", 0.0001): (1, 2, 0, 1), (3, "momentum", 0.01): (1, 2, 0, 1),
         (3, "pressure", 0.0001): (1, 2, 0, 1), (3, "pressure", 0.01): (2, 2, 1, 1)},
    S2: {(1, "momentum", 0.0001): (0, 2, 0, 2), (1, "momentum", 0.01): (2, 2, 2, 2),
         (1, "pressure", 0.0001): (0, 2, 0, 2), (1, "pressure", 0.01): (2, 2, 2, 2),
         (2, "momentum", 0.0001): (0, 2, 0, 2), (2, "momentum", 0.01): (1, 2, 1, 2),
         (2, "pressure", 0.0001): (0, 2, 0, 2), (2, "pressure", 0.01): (1, 2, 1, 2),
         (3, "momentum", 0.0001): (1, 2, 0, 1), (3, "momentum", 0.01): (1, 2, 0, 1),
         (3, "pressure", 0.0001): (1, 2, 0, 1), (3, "pressure", 0.01): (1, 2, 0, 1)},
}


def make_calibration() -> dict:
    return {
        "format_version": frozen_calibration.FORMAT_VERSION,
        "description": "Made-up calibration of tests/test_final_figures.py",
        "normalizers": {component: 1.0 for component in PHS_COMPONENT_NAMES},
        "normalizer_eps": 0.0,
        "scores": {name: list(components) for name, components in SCORE_DEFINITIONS.items()},
        "thresholds": dict(THRESHOLDS),
    }


def make_fields() -> pd.DataFrame:
    """The made-up benchmark, laid out like results/benchmark/confirmation_scores_raw.csv."""
    rows = []
    for case_id, values in VALUES.items():
        for (perturbation, epsilon), mom in zip(FIELDS, values):
            rows.append({"case_id": case_id, "k": K[case_id], "perturbation": perturbation, "epsilon": epsilon,
                         "label": "clean" if perturbation == "none" else "hallucinated",
                         "mom": mom, "div": 0.0, "bc": 0.0, "E": 0.0})
    scored = frozen_calibration.apply(pd.DataFrame(rows), make_calibration())
    return bm.build_field_table(scored, {case_id: True for case_id in K})


def make_case_inputs(passed=None):
    """A case list, true errors, a loss summary and a benchmark for the made-up cases; passed: {case_id: bool}."""
    passed = passed or {}
    cases = [{"case_id": case_id, "k": k, "U0": 1.0 + i / 10, "Re": 50.0 + i, "phi_x": 0.1 * i, "phi_y": 0.2 * i,
              "seed_points": 3000 + int(case_id[5:]), "seed_train": 4000 + int(case_id[5:]),
              "seed_scoring": 1000 + int(case_id[5:]), "split": "confirmation"}
             for i, (case_id, k) in enumerate(K.items())]
    errors = pd.DataFrame([{"case_id": case["case_id"], "k": case["k"], "stage": stage,
                            "relL2_uv": 0.002 * (j + 1) if stage == "lbfgs_final" else 0.3,
                            "passed_acceptance": passed.get(case["case_id"], True)}
                           for j, case in enumerate(cases) for stage in ("adam_05000", "lbfgs_final")])
    loss = pd.DataFrame([{"case_id": case["case_id"], "rel_l2": 0.002 * (j + 1) if passed.get(case["case_id"], True)
                          else 0.2, "mse_rc": 1e-6, "mse_rm": 1e-5, "passed": passed.get(case["case_id"], True)}
                         for j, case in enumerate(cases)])
    fields = make_fields()
    fields["passed_acceptance"] = fields["case_id"].map(lambda c: passed.get(c, True))
    return cases, errors, loss, fields


@pytest.fixture(scope="module")
def committed():
    """Every input of the committed results, read and checked (no file written)."""
    return ff.build()


# --- Recall by score x k x type x strength ---

@pytest.mark.parametrize("score", [S1, S2])
def test_recall_by_k_counts_by_hand(score):
    table = ff.recall_by_type_eps_k(make_fields())
    assert list(table.columns) == list(ff.RECALL_BY_K_COLUMNS)
    rows = table[table["score"] == score]
    assert len(rows) == len(BY_HAND[score])
    for (k, perturbation, epsilon), (flagged, n, new_flagged, n_new) in BY_HAND[score].items():
        row = rows[(rows["k"] == k) & (rows["perturbation"] == perturbation) & (rows["epsilon"] == epsilon)].iloc[0]
        assert (row["n_damaged_flagged"], row["n_damaged"]) == (flagged, n), (k, perturbation, epsilon)
        assert (row["n_damaged_new_flagged"], row["n_damaged_new"]) == (new_flagged, n_new), (k, perturbation, epsilon)
        assert row["recall"] == flagged / n
        assert (row["recall_low"], row["recall_high"]) == clopper_pearson(flagged, n)
        assert (row["recall_new_low"], row["recall_new_high"]) == clopper_pearson(new_flagged, n_new)
        assert row["n_cases"] == 2


def test_recall_by_k_covers_every_score_k_type_and_strength():
    table = ff.recall_by_type_eps_k(make_fields())
    assert len(table) == len(SCORE_DEFINITIONS) * 3 * 2 * 2
    assert not table.duplicated(["score", "k", "perturbation", "epsilon"]).any()
    # Score5's tau is above every field, so it flags nothing
    assert (table.loc[table["score"] == S5, "n_damaged_flagged"] == 0).all()


def test_recall_by_k_refuses_two_fields_of_a_case_at_one_strength():
    fields = make_fields()
    doubled = pd.concat([fields, fields[(fields["case_id"] == "case_30") & (fields["perturbation"] == "momentum")]])
    with pytest.raises(ValueError, match="two damaged fields"):
        ff.recall_by_type_eps_k(doubled)


# --- The case table ---

def test_case_table_in_case_file_order():
    cases, errors, loss, fields = make_case_inputs()
    table = ff.case_table(cases, errors, loss, fields)
    assert list(table.columns) == list(ff.CASE_COLUMNS)
    assert table["case_id"].tolist() == list(K)
    assert table["passed_acceptance"].all()
    # relL2_uv is the final model's (lbfgs_final), not an Adam checkpoint's
    assert table["relL2_uv"].tolist() == pytest.approx([0.002 * (j + 1) for j in range(len(K))])
    assert table["seed_points"].tolist() == [3000 + int(c[5:]) for c in K]


def test_case_table_keeps_a_failed_run():
    cases, errors, loss, fields = make_case_inputs(passed={"case_40": False})
    table = ff.case_table(cases, errors, loss, fields).set_index("case_id")
    assert not table.loc["case_40", "passed_acceptance"]
    assert table.drop(index="case_40")["passed_acceptance"].all()
    assert "5 of 6 runs passed (failed runs stay in every table)" in ff.caption_cases(table.reset_index())


@pytest.mark.parametrize("change", ["missing_in_loss", "twice_in_errors", "passed_differs", "not_following",
                                    "benchmark_differs", "k_differs"])
def test_case_table_refuses_inputs_that_disagree(change):
    cases, errors, loss, fields = make_case_inputs()
    if change == "missing_in_loss":
        loss = loss[loss["case_id"] != "case_41"]
    elif change == "twice_in_errors":
        errors = pd.concat([errors, errors[(errors["case_id"] == "case_41") & (errors["stage"] == "lbfgs_final")]])
    elif change == "passed_differs":
        errors.loc[errors["case_id"] == "case_41", "passed_acceptance"] = False
    elif change == "not_following":
        loss.loc[loss["case_id"] == "case_41", "mse_rm"] = 2e-3   # Above 1e-3, yet marked passed
    elif change == "benchmark_differs":
        fields.loc[fields["case_id"] == "case_41", "passed_acceptance"] = False
    elif change == "k_differs":
        errors.loc[errors["case_id"] == "case_41", "k"] = 3
    with pytest.raises(ValueError):
        ff.case_table(cases, errors, loss, fields)


def test_seed_rule():
    cases, errors, loss, fields = make_case_inputs()
    table = ff.case_table(cases, errors, loss, fields)
    assert ff._seed_rule(table, "seed_points") == " ($3000 + n$)"
    table.loc[0, "seed_points"] = 1
    assert ff._seed_rule(table, "seed_points") == ""


# --- Caption helpers ---

def test_full_recall_epsilon():
    table = pd.DataFrame({"score": S1, "perturbation": "momentum", "epsilon": [0.0001, 0.002, 0.01, 0.02],
                          "recall": [0.0, 1.0, 0.9, 1.0]})
    # 1.0 at 0.002 does not count: recall drops again at 0.01
    assert ff.full_recall_epsilon(table, "recall") == {(S1, "momentum"): 0.02}
    table["recall"] = [0.0, 0.5, 0.9, 0.9]
    assert np.isnan(ff.full_recall_epsilon(table, "recall")[(S1, "momentum")])


def test_recall_sentence_names_the_exceptions():
    # Most scores flag everything from 0.01 on; Score3 already from 0.002; Score1 only from 0.02
    counts = {S1: (1, 15, 30), S2: (10, 30, 30), S3: (30, 30, 30), S4: (12, 30, 30), S5: (9, 30, 30)}
    rows = [{"score": score, "perturbation": "boundary", "epsilon": epsilon, "recall": n / 30,
             "n_damaged_flagged": n, "n_damaged": 30}
            for score, values in counts.items() for epsilon, n in zip((0.002, 0.01, 0.02), values)]
    sentence = ff.recall_sentence(pd.DataFrame(rows), "recall", "n_damaged_flagged", "n_damaged")
    assert sentence.startswith("Every score flags every damaged field from $\\varepsilon = 0.01$ on")
    assert "(already from 0.002 for Score3 with boundary damage)" in sentence
    assert "except Score1 for boundary damage (15/30 at 0.01)" in sentence


def test_label_positions_move_only_up():
    heights = ff._label_positions([4.49, 4.28], min_ratio=1.16, floor=4.6)
    assert heights[1] == pytest.approx(4.6)             # The lower point's label, raised to the floor
    assert heights[0] == pytest.approx(4.6 * 1.16)      # The higher one, kept above it
    assert ff._label_positions([38.9, 23.4], min_ratio=1.16) == [38.9, 23.4]   # Far apart: unmoved


def test_relative_energy_error():
    clean = pd.DataFrame({"U0": [2.0], "E": [(0.01 * 2.0 ** 2 / 4) ** 2]})
    assert ff.relative_energy_error(clean).iloc[0] == pytest.approx(0.01)


def test_zero_width_recall_needs_one_shared_count():
    fields = make_fields()
    # Pretend the bootstrap gave Score1 no width for k = 2, whose cases flag 1 and 2 fields
    row = {"score": S1, "k": 2, "recall": 0.625, "recall_low": 0.625, "recall_high": 0.625}
    with pytest.raises(ValueError, match="same count"):
        ff.zero_width_recalls(pd.DataFrame([row]), fields)
    # Score2 for k = 1 is flagged twice in each case, so a zero width is consistent there
    row = {"score": S2, "k": 1, "recall": 0.5, "recall_low": 0.5, "recall_high": 0.5}
    (found_row, count, per_case), = ff.zero_width_recalls(pd.DataFrame([row]), fields)
    assert (count, per_case) == (2, 4)


def test_within_case_concordance():
    fields = make_fields()
    within, across = ff.within_case_concordance(fields, S1)
    # Damaged fields above their own clean field, ties counting one half: case_30 3, case_31 3,
    # case_40 3.5 (0.2 ties its clean 0.2), case_41 3.5 (0.1 ties), case_50 3 (two ties), case_51 3; of 24
    assert within == pytest.approx(19 / 24)
    assert 0 < across < 1


def test_names_cover_every_score_and_damage_type():
    assert set(ff.SCORE_LABELS) == set(ff.SCORE_COLOURS) == set(ff.SCORE_MARKERS) == set(SCORE_DEFINITIONS)
    assert set(ff.DAMAGE_LABELS) == set(PERTURBATION_NAMES)
    assert len(set(ff.SCORE_COLOURS.values())) == len(SCORE_DEFINITIONS)
    assert len(set(ff.SCORE_MARKERS.values())) == len(SCORE_DEFINITIONS)


# --- The committed results ---

def test_committed_inputs_agree(committed):
    fields = committed["fields"]
    assert len(fields) == 930 and fields["case_id"].nunique() == 30
    assert len(committed["old"]) == 30
    assert committed["tables"]["table_cases"]["case_id"].tolist() == [c["case_id"] for c in read_cases(ff.DEFAULT_METADATA)]
    assert len(committed["tables"]["table_recall_by_type_eps_k"]) == 5 * 3 * 5 * 6


def test_recall_by_k_sums_to_the_benchmark(committed):
    """Summed over k, the type x strength x k counts are #35's type x strength counts."""
    by_k = committed["tables"]["table_recall_by_type_eps_k"]
    summed = by_k.groupby(["score", "perturbation", "epsilon"])[["n_damaged_flagged", "n_damaged",
                                                                 "n_damaged_new_flagged", "n_damaged_new"]].sum()
    type_eps = committed["metrics"]["by_type_eps"].set_index(["score", "perturbation", "epsilon"])
    for column in summed.columns:
        assert (summed[column] == type_eps.loc[summed.index, column]).all(), column


def copy_benchmark(tmp_path) -> Path:
    folder = tmp_path / "benchmark"
    shutil.copytree(ff.DEFAULT_BENCHMARK_DIR, folder)
    return folder


def test_read_fields_refuses_a_changed_flag(tmp_path):
    folder = copy_benchmark(tmp_path)
    path = folder / ff.FIELDS_CSV
    fields = pd.read_csv(path, float_precision="round_trip")
    fields.loc[5, f"flagged_{S3}"] = not fields.loc[5, f"flagged_{S3}"]
    fields.to_csv(path, index=False, lineterminator="\n")
    calib = frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON)
    with pytest.raises(ValueError, match="frozen calibration"):
        ff.read_fields(folder, calib, ff.DEFAULT_METADATA)


def test_read_fields_refuses_a_changed_preflag(tmp_path):
    folder = copy_benchmark(tmp_path)
    path = folder / ff.FIELDS_CSV
    fields = pd.read_csv(path, float_precision="round_trip")
    fields.loc[5, f"preflagged_{S1}"] = not fields.loc[5, f"preflagged_{S1}"]
    fields.to_csv(path, index=False, lineterminator="\n")
    calib = frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON)
    with pytest.raises(ValueError, match="preflagged"):
        ff.read_fields(folder, calib, ff.DEFAULT_METADATA)


@pytest.mark.parametrize("name, column, value", [("by_type_eps", "n_damaged_flagged", 1),
                                                 ("by_k", "n_false_alarms", 1), ("overall", "auc", 1e-9)])
def test_check_metrics_refuses_a_table_that_disagrees(committed, name, column, value):
    metrics = {key: table.copy() for key, table in committed["metrics"].items()}
    metrics[name].loc[3, column] += value
    with pytest.raises(ValueError, match=f"metrics_{name}"):
        ff.check_metrics(committed["fields"], metrics)


def test_read_old_clean_fields_refuses_another_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(Path(ff.DEFAULT_SOURCE_CSV).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        ff.read_old_clean_fields(source, frozen_calibration.load(ff.DEFAULT_CALIBRATION_JSON))


def test_old_clean_fields_give_the_calibration_back(committed):
    """tau is the 95th percentile of the old clean scores (PROTOCOL.md section 5), each with 2 of 30 above it."""
    thresholds = committed["calib"]["thresholds"]
    for score, tau in thresholds.items():
        assert np.percentile(committed["old"][score], 95) == pytest.approx(tau, rel=1e-12)
        assert (committed["old"][score] > tau).sum() == 2


def test_captions_explain_what_36_asks(committed):
    captions = committed["captions"]
    # An AUC range above its own estimate, and why
    assert "Score4 for $k = 1$" in captions["fig_roc"] and "0.927 [0.928, 0.941]" in captions["fig_roc"]
    assert "draws a case twice" in captions["fig_roc"]
    # A range with no width, and why
    assert "0.733 [0.733, 0.733]" in captions["fig_recall_vs_eps_by_k"]
    assert "22 of its 30 damaged fields" in captions["fig_recall_vs_eps_by_k"]
    # The false alarms by k, and the energy term
    text = captions["fig_clean_scores_vs_tau"]
    for case_id in ("case\\_46", "case\\_48", "case\\_53", "case\\_34", "case\\_45", "case\\_36"):
        assert case_id in text
    assert "$U_0^4$" in text and "$-0.80$" in text


def test_captions_are_plain_latex(committed):
    """
    No '#', and no '_' or '%' outside math unless escaped (they break LaTeX); no Greek
    letters outside math; no repository files or issue numbers, since the captions go into the paper.
    """
    for name, caption in committed["captions"].items():
        assert caption.count("$") % 2 == 0, name
        text = "".join(caption.split("$")[0::2])   # The caption outside math
        text = re.sub(r"\\ref\{[^}]*\}", "", text)  # Labels may hold underscores
        for bad in ("#", "PROTOCOL", ".md", "τ", "ε"):
            assert bad not in caption, (name, bad)
        assert "_" not in text.replace("\\_", ""), name
        assert "%" not in text.replace("\\%", ""), name


def test_run_writes_everything_and_reruns_identically(tmp_path):
    _, written = ff.run(output_dir=tmp_path / "a")
    names = sorted(path.name for path in written)
    expected = sorted([f"{stem}.{ext}" for stem in ff.FIGURES for ext in ("pdf", "png", "tex")]
                      + [f"{stem}.{ext}" for stem in ff.TABLES for ext in ("csv", "tex")])
    assert names == expected
    ff.run(output_dir=tmp_path / "b")
    for name in names:
        first, second = (tmp_path / "a" / name).read_bytes(), (tmp_path / "b" / name).read_bytes()
        assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest(), name


def test_committed_tables_and_captions_are_the_scripts(committed):
    """The committed .csv read back as the tables built now; the committed .tex are what the script writes."""
    for stem in ff.TABLES:
        saved = pd.read_csv(OUTPUT_DIR / f"{stem}.csv", float_precision="round_trip")
        # Clopper-Pearson ends can differ in the last digit between operating systems (scipy)
        pd.testing.assert_frame_equal(saved, committed["tables"][stem], check_dtype=False, rtol=1e-12, obj=stem)
    tables = committed["tables"]
    expected = {
        "table_cases.tex": ff.case_table_latex(tables["table_cases"], committed["captions"]["table_cases"]),
        "table_recall_by_type_eps_k.tex": ff.recall_by_k_latex(tables["table_recall_by_type_eps_k"],
                                                               committed["captions"]["table_recall_by_type_eps_k"]),
        **{f"{stem}.tex": ff.figure_tex(stem, committed["captions"][stem], wide=stem != "fig_roc")
           for stem in ff.FIGURES},
    }
    for name, text in expected.items():
        assert (OUTPUT_DIR / name).read_text(encoding="utf-8") == text, name


def test_committed_figures_exist():
    for stem in ff.FIGURES:
        for ext in ("pdf", "png"):
            assert (OUTPUT_DIR / f"{stem}.{ext}").stat().st_size > 10_000, f"{stem}.{ext}"


def test_path_flags_default_to_the_repo_paths():
    args = ff.parse_args([])
    assert Path(args.benchmark_dir) == REPO.resolve() / "results" / "benchmark"
    assert Path(args.errors) == REPO.resolve() / "results" / "accuracy" / "reference_errors_confirmation.csv"
    assert Path(args.metadata) == REPO.resolve() / "cases" / "confirmation_cases.json"
    assert Path(args.loss_summary) == REPO.resolve() / "models" / "confirmation" / "loss_summary.csv"
    assert Path(args.calibration) == REPO.resolve() / "calibration" / "frozen_calibration.json"
    assert Path(args.source) == REPO.resolve() / "calibration" / "source_phs_components_raw.csv"
    assert Path(args.output_dir) == REPO.resolve() / "results" / "figures"
