"""
Natural errors of the final iteration (#39): checkpoint scores vs analytical velocity error.

Barak: "Report checkpoint scores against analytical velocity error, including
inaccurate outputs that remain unflagged ... Keep unsuccessful runs visible in
the results."

The perturbation benchmark (#35) damages the fields of the final models on
purpose. Here nothing is damaged. Each of the 30 new cases was saved at 7
stages of its training (#40): after 100, 500, 1000, 2500, 4000 and 5000 Adam
steps, and after L-BFGS (lbfgs_final, the final model). The early checkpoints
are wrong on their own, because they are only partly trained. This script
scores the clean field of all 210 checkpoints and holds the scores against each
checkpoint's true error, measured against the exact Taylor-Green solution by
the accuracy audit (#37):

  1. Score (the only step that needs the models: about 3 s per checkpoint on a
     CPU). Every checkpoint is found with accuracy_audit.case_checkpoints() and
     checked against the case file (checkpoints.check_sidecar()) before any
     scoring, so a missing or wrong checkpoint stops the run instead of being
     skipped. Each is loaded with checkpoints.load_checkpoint(), and its clean
     field gets the 4 raw components from phs.compute_phs_components(), the
     function evaluate_phs.py calls for every field, called the way
     evaluate_phs.py calls it: with evaluate_phs.py's own default settings
     (read from its parser, PROTOCOL.md section 4) and the case's seed_scoring
     (1000 + n, the seed evaluate_phs.py derives from the case number).
     frozen_calibration.apply() (#45) then adds the normalized components, the
     5 scores and the 5 flags (score > tau). Nothing is fitted.
  2. Join with the true errors (results/accuracy/reference_errors_confirmation.csv)
     on (case_id, stage). A checkpoint is inaccurate if relL2_uv > 0.10.
  3. Consistency check. The lbfgs_final fields are the clean fields of #35, so
     they must match the clean rows of results/benchmark/confirmation_scores_raw.csv:
     every flag the same, and every component and score within 1e-9
     (relative). On the same machine they are equal bit for bit; on another
     machine or OS the components differ in the last digits (up to about 2e-13,
     relative). The run stops if the check fails, and prints the largest
     difference.
  4. Count, list and plot (below).

Counts, for each score, by stage x k (each stage and all stages together, each
k and all k together):
    n_inaccurate     checkpoints with relL2_uv > 0.10
    n_flagged        checkpoints with score > tau
    n_missed         inaccurate and not flagged: a missed failure
    n_false_alarms   accurate (relL2_uv <= 0.10) and flagged
and two rates with their 95% ranges (PROTOCOL.md section 7, grouped_eval.py, #41):
    recall   flagged inaccurate checkpoints / inaccurate checkpoints
             (1 - recall is the miss rate)
    fpr      false alarms / accurate checkpoints: the false-alarm rate
A rate with nothing to count (recall where no checkpoint is inaccurate) is NaN.
Within one stage each case gives one checkpoint, so those rows get the exact
Clopper-Pearson range of their counts. All stages together give 7 checkpoints
per case, so those rows get the case bootstrap: 2000 resamples of whole cases
within each k, seed 0 (<rate>_n_boot says how many resamples were used).
range_method names the method of each row. Where every (or no) checkpoint
counted by a rate is flagged, every resample gives the same rate and the
bootstrap range has zero width (as in #35's tables); the exact ranges of the
single stages then show the uncertainty.

"Accurate" means the 10% rule only. An Adam checkpoint with a 5% error is
accurate by it, though far less accurate than any final model (0.15% to 0.73%),
so every list gives the stage, and the by-stage rows keep the false alarms on
the final models (the clean false alarms of #35) apart from those on Adam
checkpoints.

Failed runs stay in. passed_acceptance is the status of the run's lbfgs_final
sidecar (PASSED or FAILED, PROTOCOL.md section 2), the same on all 7
checkpoints of a run. It is in every table, n_failed_runs counts the failed
runs of each group, and the scatter plot rings their checkpoints.

Outputs, in results/natural_errors/, each table as .csv and .tex (the .tex
files need the booktabs and longtable packages, and are written with
benchmark_metrics.py's LaTeX helpers), each figure as .pdf and .png:
    checkpoint_scores         every checkpoint: case_id, k, stage, adam_step, the raw
                              and normalized components, the 5 scores, the 5 flags,
                              relL2_uv, inaccurate and passed_acceptance (210 rows)
    table_stage_by_k          the four counts and the two rates of each score, by
                              stage x k
    missed_failures           every inaccurate checkpoint a score does not flag, one
                              row per checkpoint x score
    false_alarms              every accurate checkpoint a score flags, one row per
                              checkpoint x score
    inaccurate_final_models   every inaccurate final model (lbfgs_final), with its
                              scores and flags, kept apart from the Adam stages
    fig_score_vs_error        each score against relL2_uv, one panel per score
                              (log-log), with tau and the 10% line; colour = stage,
                              marker = k, failed runs ringed
    fig_fields_<case_id>_<stage>   the fields of the checkpoints chosen by
                              choose_field_checkpoints(): the missed failures, a
                              typical caught failure and the one closest to being
                              missed, a typical final model, and a typical and the
                              highest false alarm of PHS

Usage:
    # Score the 210 checkpoints and write everything (about 15 min on a CPU)
    python src/detection/natural_errors.py
    # Rebuild the tables and figures from the saved checkpoint_scores.csv, without rescoring
    python src/detection/natural_errors.py --from_scores
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detection import benchmark_metrics as bm
from src.detection import evaluate_phs, frozen_calibration
from src.detection.grouped_eval import case_bootstrap
from src.detection.phs import PHS_COMPONENT_NAMES, compute_phs_components
from src.models.accuracy_audit import INACCURATE_ABOVE, accuracy_grid, case_checkpoints, read_cases, relative_l2
from src.models.checkpoints import FINAL_STAGE, STAGES, check_sidecar, load_checkpoint
from src.models.scaling import ResidualScaler
from src.physics.taylor_green import compute_nu, compute_T, generate_tgv
from src.utils.console import enable_utf8_output

DEFAULT_METADATA = PROJECT_ROOT / "cases" / "confirmation_cases.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "confirmation"
DEFAULT_CALIBRATION_JSON = PROJECT_ROOT / "calibration" / "frozen_calibration.json"
DEFAULT_ERRORS_CSV = PROJECT_ROOT / "results" / "accuracy" / "reference_errors_confirmation.csv"
DEFAULT_BENCHMARK_CSV = PROJECT_ROOT / "results" / "benchmark" / "confirmation_scores_raw.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "natural_errors"

N_BOOT = bm.N_BOOT          # Resamples of case_bootstrap (PROTOCOL.md section 7)
BOOT_SEED = bm.BOOT_SEED
CONSISTENCY_RTOL = 1e-9     # Largest relative difference allowed against #35's clean fields
CLEAN = bm.CLEAN            # "none": the clean field's perturbation, as in the hallucination index
ALL = "all"                 # The stage or k of the rows that pool all stages or all k

SCORE_NAMES = bm.SCORE_NAMES
BAR_NAMES = bm.BAR_NAMES
FLAG_NAMES = tuple(f"flagged_{name}" for name in SCORE_NAMES)
PHS = "Score3_PHS_full"     # The official score; used to choose the typical field plots

SCORES_TABLE = "checkpoint_scores"
# What step 1 gives: one row per checkpoint, before the calibration and the join
RAW_COLUMNS = ("case_id", "k", "stage", "adam_step", "passed_acceptance", *PHS_COMPONENT_NAMES)
CHECKPOINT_COLUMNS = ("case_id", "k", "stage", "adam_step", *PHS_COMPONENT_NAMES, *BAR_NAMES, *SCORE_NAMES,
                      *FLAG_NAMES, "relL2_uv", "inaccurate", "passed_acceptance")
# The columns read from the accuracy audit's CSV
ERROR_COLUMNS = ("case_id", "k", "stage", "adam_step", "relL2_uv", "passed_acceptance", "inaccurate")
_CHECKED_IN_JOIN = ("k", "adam_step", "passed_acceptance")
STAGE_BY_K_COLUMNS = ("score", "tau", "stage", "k", "n_cases", "n_failed_runs", "n_checkpoints", "n_inaccurate",
                      "n_flagged", "n_missed", "n_false_alarms", "recall", "recall_low", "recall_high",
                      "recall_n_boot", "fpr", "fpr_low", "fpr_high", "fpr_n_boot", "range_method")
LIST_COLUMNS = ("score", "tau", "case_id", "k", "stage", "adam_step", "relL2_uv", "score_value", "score_over_tau",
                "passed_acceptance")
FINAL_COLUMNS = ("case_id", "k", "stage", "adam_step", "relL2_uv", "passed_acceptance", *SCORE_NAMES, *FLAG_NAMES)

# Field plots: every missed failure is plotted when there are at most this many of them;
# otherwise the worst miss of each score (see choose_field_checkpoints)
MAX_MISSED_FIELD_PLOTS = 6


# --- Step 1: scoring the checkpoints ---

def scoring_settings(device: str):
    """
    evaluate_phs.py's scoring settings (PROTOCOL.md section 4): the defaults of its
    parser, read by running the parser with no flag but --device, so they are
    evaluate_phs.py's own values and not a copy of them.

    Inputs:
        device (str): "cpu" or "cuda" (the protocol allows both, float64 either way).

    Outputs:
        argparse.Namespace: evaluate_phs.py's arguments (n_interior, n_bc, n_time,
                            energy_res, chunk_size, bc_local_band_width_fraction,
                            bc_local_n_points, device, ...).
    """
    with mock.patch.object(sys, "argv", ["evaluate_phs.py", "--device", device]):
        return evaluate_phs.parse_args()


def scoring_seed(case: dict) -> int:
    """
    The seed of a case's scoring points: its seed_scoring in the case file, which must
    be the seed evaluate_phs.py derives from the case number (1000 + n, PROTOCOL.md
    section 4), so that each lbfgs_final field is scored at the same points as in #35.
    """
    derived = 1000 + int(case["case_id"].split("_")[1])
    if case.get("seed_scoring") != derived:
        raise ValueError(f"{case['case_id']} has seed_scoring {case.get('seed_scoring')}, but evaluate_phs.py "
                         f"scores it with seed {derived}")
    return derived


def score_checkpoint(model, case: dict, settings) -> dict:
    """
    The 4 raw components of a model's clean field: phs.compute_phs_components(),
    called as evaluate_phs.py's score_all_fields() calls it for a clean row
    (perturbation "none", epsilon 0), with its settings and the case's scoring seed.

    Inputs:
        model (nn.Module): The checkpoint, in eval mode, float64, on settings.device.
        case (dict): The case's entry in the case file.
        settings (argparse.Namespace): Output of scoring_settings().

    Outputs:
        dict: {"mom", "div", "bc", "E"}, floats.
    """
    U0, Re, k = case["U0"], case["Re"], case["k"]
    return compute_phs_components(
        model, case, compute_nu(U0, Re, k), compute_T(U0, Re, k), ResidualScaler(U0, k),
        CLEAN, 0.0,
        n_interior=settings.n_interior, n_bc_per_axis=settings.n_bc,
        n_time=settings.n_time, energy_res=settings.energy_res,
        chunk_size=settings.chunk_size, device=settings.device,
        bc_local_band_width_fraction=settings.bc_local_band_width_fraction,
        bc_local_n_points=settings.bc_local_n_points,
        seed=scoring_seed(case),
    )


def run_checkpoints(cases: list, models_dir) -> list:
    """
    The 7 checkpoints of every case, all checked before any scoring: each case needs
    its run folder with every stage of STAGES (accuracy_audit.case_checkpoints()),
    each sidecar must be that checkpoint of that case (checkpoints.check_sidecar()),
    and the case's scoring seed must be evaluate_phs.py's. A missing or wrong one
    stops the run instead of being skipped (PROTOCOL.md section 1: no case is dropped).

    Inputs:
        cases (list[dict]): The cases, from accuracy_audit.read_cases().
        models_dir (str | Path): Folder with one folder of checkpoints per case.

    Outputs:
        list of (case, [(stage, path), ...]), in case order, stages in training order.
    """
    plan = []
    for case in cases:
        checkpoints = case_checkpoints(case["case_id"], models_dir)
        if [stage for stage, _ in checkpoints] != list(STAGES):
            raise ValueError(f"{case['case_id']} has no folder of the {len(STAGES)} staged checkpoints in "
                             f"{models_dir}; the natural errors need the checkpoints of a run of #40")
        for stage, path in checkpoints:
            with open(Path(path).with_suffix(".json"), "r", encoding="utf-8") as f:
                check_sidecar(json.load(f), case, stage, path)
        scoring_seed(case)
        plan.append((case, checkpoints))
    return plan


def score_checkpoints(metadata_path, models_dir, settings) -> pd.DataFrame:
    """
    Step 1: the raw components of the clean field of every checkpoint of every case.

    Inputs:
        metadata_path (str | Path): The case file (cases/confirmation_cases.json).
        models_dir (str | Path): Folder with one folder of checkpoints per case (models/confirmation).
        settings (argparse.Namespace): Output of scoring_settings().

    Outputs:
        pd.DataFrame: One row per checkpoint, with RAW_COLUMNS, cases in case-file order
                      and stages in training order. passed_acceptance is the run's: its
                      lbfgs_final sidecar says PASSED.
    """
    plan = run_checkpoints(read_cases(metadata_path), models_dir)
    rows = []
    for case, checkpoints in plan:
        print(f"[{case['case_id']}] scoring {len(checkpoints)} checkpoints (k={case['k']})...")
        case_rows, status = [], None
        for stage, path in checkpoints:
            model, info = load_checkpoint(path, device=settings.device, k=case["k"], metadata_paths=[metadata_path])
            check_sidecar(info, case, stage, path)
            if stage == FINAL_STAGE:
                status = info["status"]
            case_rows.append({"case_id": case["case_id"], "k": case["k"], "stage": stage,
                              "adam_step": info["adam_steps"], **score_checkpoint(model, case, settings)})
            del model
        if status not in ("PASSED", "FAILED"):
            raise ValueError(f"The lbfgs_final sidecar of {case['case_id']} has status {status!r}, "
                             "expected PASSED or FAILED")
        rows.extend({**row, "passed_acceptance": status == "PASSED"} for row in case_rows)
        if settings.device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)[list(RAW_COLUMNS)]


# --- Step 2: the calibration and the join with the true errors ---

def read_true_errors(path) -> pd.DataFrame:
    """Reads the accuracy audit's CSV exactly (float_precision="round_trip")."""
    errors = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in ERROR_COLUMNS if column not in errors.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}; write it with src/models/accuracy_audit.py")
    return errors


def join_errors(scores: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the checkpoint scores with their true errors (#37) on (case_id, stage).

    Refuses a join that would drop or double a checkpoint silently: each side must list
    each (case_id, stage) once, both sides the same checkpoints, with the same k,
    adam_step and passed_acceptance. inaccurate is recomputed as relL2_uv > 0.10 and
    must match the column of the true errors.

    Inputs:
        scores (pd.DataFrame): One row per checkpoint, with case_id, k, stage,
            adam_step and passed_acceptance.
        errors (pd.DataFrame): The true errors, from read_true_errors().

    Outputs:
        pd.DataFrame: scores, in its row order, with relL2_uv and inaccurate added.
    """
    missing = [column for column in ERROR_COLUMNS if column not in errors.columns]
    if missing:
        raise ValueError(f"The true errors have no column(s) {missing}")
    key = ["case_id", "stage"]
    for name, table in (("scores", scores), ("true errors", errors)):
        doubled = table.duplicated(key)
        if doubled.any():
            raise ValueError(f"Checkpoints listed twice in the {name}: {table.loc[doubled, key].values.tolist()}")
    scored = set(zip(scores["case_id"], scores["stage"]))
    measured = set(zip(errors["case_id"], errors["stage"]))
    if scored != measured:
        raise ValueError(f"The scores and the true errors must cover the same checkpoints: no true error for "
                         f"{sorted(scored - measured)}, not scored {sorted(measured - scored)}")

    joined = scores.drop(columns=["relL2_uv", "inaccurate"], errors="ignore").merge(
        errors[list(ERROR_COLUMNS)], on=key, how="left", validate="one_to_one", suffixes=("", "_errors"))
    for column in _CHECKED_IN_JOIN:
        differ = ~joined[column].eq(joined[f"{column}_errors"])
        if differ.any():
            raise ValueError(f"{column} differs between the scores and the true errors for "
                             f"{joined.loc[differ, key].values.tolist()}")
    if joined["relL2_uv"].isna().any():
        raise ValueError(f"No relL2_uv for {joined.loc[joined['relL2_uv'].isna(), key].values.tolist()}")
    inaccurate = joined["relL2_uv"] > INACCURATE_ABOVE
    differ = ~inaccurate.eq(joined["inaccurate"])
    if differ.any():
        raise ValueError(f"The true errors' inaccurate column is not relL2_uv > {INACCURATE_ABOVE} for "
                         f"{joined.loc[differ, key].values.tolist()}")
    joined["inaccurate"] = inaccurate
    return joined.drop(columns=[f"{column}_errors" for column in _CHECKED_IN_JOIN])


def build_checkpoint_table(raw: pd.DataFrame, calib: dict, errors: pd.DataFrame) -> pd.DataFrame:
    """
    The checkpoint table from the raw components: the frozen calibration
    (frozen_calibration.apply(): normalized components, scores and flags, nothing
    fitted) and the join with the true errors (join_errors()).

    Inputs:
        raw (pd.DataFrame): One row per checkpoint, with RAW_COLUMNS (score_checkpoints()).
        calib (dict): The frozen calibration, from frozen_calibration.load().
        errors (pd.DataFrame): The true errors, from read_true_errors().

    Outputs:
        pd.DataFrame: One row per checkpoint, with CHECKPOINT_COLUMNS, in raw's row order.
    """
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"The raw scores have no column(s) {missing}")
    scored = frozen_calibration.apply(raw[list(RAW_COLUMNS)], calib)
    return join_errors(scored, errors)[list(CHECKPOINT_COLUMNS)].reset_index(drop=True)


def read_checkpoint_scores(path, calib: dict, errors: pd.DataFrame) -> pd.DataFrame:
    """
    Reads a checkpoint_scores.csv back exactly (float_precision="round_trip"), for
    --from_scores, and refuses it unless it follows from its own raw components:
    rebuilding it from them with the frozen calibration and the true errors
    (build_checkpoint_table()) must give every column back bit for bit. So the tables
    can't come from checkpoints scored with another calibration, joined with other
    true errors, or edited by hand.

    Outputs:
        pd.DataFrame: The checkpoint table, with CHECKPOINT_COLUMNS.
    """
    table = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in CHECKPOINT_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}")
    rebuilt = build_checkpoint_table(table[list(RAW_COLUMNS)], calib, errors)
    for column in CHECKPOINT_COLUMNS:
        if not np.array_equal(rebuilt[column].to_numpy(), table[column].to_numpy()):
            raise ValueError(f"Column {column} of {path} doesn't follow from its raw components, the frozen "
                             "calibration and the true errors; rescore without --from_scores")
    return rebuilt


def check_coverage(table: pd.DataFrame, cases: list) -> None:
    """
    Refuses a table that is not the 7 checkpoints of every case of the case file, each
    once, in case-file and training order, each with its case's k. No case or stage
    is dropped or added (PROTOCOL.md sections 1 and 2).
    """
    expected = [(case["case_id"], stage) for case in cases for stage in STAGES]
    found = list(zip(table["case_id"], table["stage"]))
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise ValueError(f"The table must hold the {len(STAGES)} checkpoints of every case of the case file, "
                         f"each once and in order: missing {missing}, not in the case file {extra}")
    k_of_case = {case["case_id"]: case["k"] for case in cases}
    wrong_k = sorted({case_id for case_id, k in zip(table["case_id"], table["k"]) if k != k_of_case[case_id]})
    if wrong_k:
        raise ValueError(f"Cases whose k differs from the case file: {wrong_k}")


# --- Step 3: the consistency check with #35 ---

def read_benchmark(path) -> pd.DataFrame:
    """Reads #35's per-field table (confirmation_scores_raw.csv) exactly (float_precision="round_trip")."""
    return pd.read_csv(path, float_precision="round_trip")


def consistency_check(table: pd.DataFrame, benchmark: pd.DataFrame, rtol: float = CONSISTENCY_RTOL) -> dict:
    """
    Step 3: the lbfgs_final fields must be #35's clean fields. Compares them with the
    clean rows of #35's confirmation_scores_raw (label "clean", perturbation "none"):
    the same cases, each once, with the same k, every flag the same, and every raw and
    normalized component and every score within rtol, relative.

    Inputs:
        table (pd.DataFrame): The checkpoint table.
        benchmark (pd.DataFrame): #35's confirmation_scores_raw, from read_benchmark().
        rtol (float): The largest relative difference allowed.

    Outputs:
        dict: n_fields; max_rel_diff, the largest relative difference of any component or
              score, with the column and case_id where it is; and max_rel_diff_by_column.
              Raises ValueError if the check fails.
    """
    clean = benchmark[(benchmark["label"] == "clean") & (benchmark["perturbation"] == CLEAN)]
    final = table[table["stage"] == FINAL_STAGE]
    for name, rows in (("#35's clean fields", clean), ("the lbfgs_final checkpoints", final)):
        doubled = rows["case_id"][rows["case_id"].duplicated()].tolist()
        if doubled:
            raise ValueError(f"Cases listed twice among {name}: {doubled}")
    if set(clean["case_id"]) != set(final["case_id"]):
        raise ValueError(f"The lbfgs_final checkpoints and #35's clean fields must be the same cases: only here "
                         f"{sorted(set(final['case_id']) - set(clean['case_id']))}, only in #35 "
                         f"{sorted(set(clean['case_id']) - set(final['case_id']))}")
    merged = final.merge(clean, on="case_id", how="left", suffixes=("", "_benchmark"), validate="one_to_one")
    wrong_k = merged.loc[merged["k"] != merged["k_benchmark"], "case_id"].tolist()
    if wrong_k:
        raise ValueError(f"k differs from #35's for {wrong_k}")

    flags_differ = [(case_id, name) for name in SCORE_NAMES
                    for case_id in merged.loc[merged[f"flagged_{name}"] != merged[f"flagged_{name}_benchmark"],
                                              "case_id"]]
    by_column, where = {}, {}
    for column in (*PHS_COMPONENT_NAMES, *BAR_NAMES, *SCORE_NAMES):
        ours = merged[column].to_numpy(dtype=float)
        theirs = merged[f"{column}_benchmark"].to_numpy(dtype=float)
        difference = np.abs(ours - theirs)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(difference == 0, 0.0, difference / np.abs(theirs))
        relative = np.where(np.isnan(relative), np.inf, relative)  # A missing value never passes
        i = int(np.argmax(relative))
        by_column[column], where[column] = float(relative[i]), merged["case_id"].iloc[i]
    worst = max(by_column, key=by_column.get)
    result = {"n_fields": len(merged), "max_rel_diff": by_column[worst], "column": worst, "case_id": where[worst],
              "max_rel_diff_by_column": by_column}
    if flags_differ or result["max_rel_diff"] > rtol:
        raise ValueError(f"The lbfgs_final checkpoints don't match #35's clean fields: flags differ for "
                         f"{flags_differ}; largest relative difference {result['max_rel_diff']:.3g} "
                         f"({worst}, {where[worst]}), allowed {rtol:g}")
    return result


# --- Step 4: counts, lists and figures ---

def _view(group: pd.DataFrame, score: str) -> pd.DataFrame:
    """The columns the rates of one score need, under fixed names: case_id, k, inaccurate, flagged."""
    return pd.DataFrame({
        "case_id": group["case_id"].to_numpy(),
        "k": group["k"].to_numpy(),
        "inaccurate": group["inaccurate"].to_numpy(dtype=bool),
        "flagged": group[f"flagged_{score}"].to_numpy(dtype=bool),
    })


def recall_rate(view: pd.DataFrame) -> float:
    """Flagged inaccurate checkpoints / inaccurate checkpoints; NaN without an inaccurate one."""
    inaccurate = view["inaccurate"].to_numpy(dtype=bool)
    n = int(inaccurate.sum())
    return np.nan if n == 0 else float((view["flagged"].to_numpy(dtype=bool) & inaccurate).sum() / n)


def false_alarm_rate(view: pd.DataFrame) -> float:
    """False alarms / accurate checkpoints; NaN without an accurate one."""
    accurate = ~view["inaccurate"].to_numpy(dtype=bool)
    n = int(accurate.sum())
    return np.nan if n == 0 else float((view["flagged"].to_numpy(dtype=bool) & accurate).sum() / n)


RATES = {"recall": recall_rate, "fpr": false_alarm_rate}


def stage_by_k_row(group: pd.DataFrame, score: str, tau: float, n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> dict:
    """
    The counts and rates of one score on one group of checkpoints. A group with one
    checkpoint per case (one stage) gets the exact range of each rate's count; a group
    with several checkpoints per case (all stages) gets the case bootstrap, with the
    same resamples for both rates.

    Inputs:
        group (pd.DataFrame): The group's rows of the checkpoint table.
        score (str): One of SCORE_NAMES.
        tau (float): The score's frozen alarm level (reported only; the flags are in group).
        n_boot, seed: Of case_bootstrap.

    Outputs:
        dict: The row, without score, stage and k.
    """
    view = _view(group, score)
    inaccurate, flagged = view["inaccurate"].to_numpy(), view["flagged"].to_numpy()
    passed_by_case = group.groupby("case_id")["passed_acceptance"].first().astype(bool)
    row = {"tau": tau, "n_cases": int(view["case_id"].nunique()), "n_failed_runs": int((~passed_by_case).sum()),
           "n_checkpoints": len(view), "n_inaccurate": int(inaccurate.sum()), "n_flagged": int(flagged.sum()),
           "n_missed": int((inaccurate & ~flagged).sum()), "n_false_alarms": int((~inaccurate & flagged).sum())}
    if not view["case_id"].duplicated().any():
        row["range_method"] = "clopper_pearson"
        bm._exact(row, "recall", row["n_inaccurate"] - row["n_missed"], row["n_inaccurate"])
        bm._exact(row, "fpr", row["n_false_alarms"], row["n_checkpoints"] - row["n_inaccurate"])
        row.update(recall_n_boot=pd.NA, fpr_n_boot=pd.NA)
    else:
        row["range_method"] = "case_bootstrap"
        for name, rate in RATES.items():
            estimate, low, high, used = case_bootstrap(view, rate, n_boot=n_boot, seed=seed)
            row.update({name: estimate, f"{name}_low": low, f"{name}_high": high, f"{name}_n_boot": used})
    return row


def stage_by_k_table(table: pd.DataFrame, thresholds: dict, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    The counts and rates of every score by stage x k: each stage (in training order)
    and then all stages together, each with every k and then all k together.

    Inputs:
        table (pd.DataFrame): The checkpoint table.
        thresholds (dict): {score: tau} of the frozen calibration.
        n_boot, seed: Of case_bootstrap.

    Outputs:
        pd.DataFrame: One row per score x stage x k, with STAGE_BY_K_COLUMNS. stage and k
                      are text, "all" for the pooled rows.
    """
    unknown = sorted(set(table["stage"]) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages {unknown}")
    stages = [stage for stage in STAGES if stage in set(table["stage"])]
    ks = sorted(table["k"].unique())
    rows = []
    for score in SCORE_NAMES:
        for stage in (*stages, ALL):
            of_stage = table if stage == ALL else table[table["stage"] == stage]
            for k in (*ks, ALL):
                group = of_stage if k == ALL else of_stage[of_stage["k"] == k]
                row = stage_by_k_row(group, score, thresholds[score], n_boot, seed)
                rows.append({"score": score, "stage": stage, "k": str(k), **row})
    out = pd.DataFrame(rows)[list(STAGE_BY_K_COLUMNS)]
    return out.astype({"recall_n_boot": "Int64", "fpr_n_boot": "Int64"})


def _score_rows(table: pd.DataFrame, thresholds: dict, select) -> pd.DataFrame:
    """One row per (checkpoint, score) that select(table, score) keeps, with the score's value and tau."""
    frames = []
    for score in SCORE_NAMES:
        chosen = table[select(table, score)]
        if chosen.empty:
            continue
        tau = thresholds[score]
        frames.append(pd.DataFrame({
            "score": score, "tau": tau,
            "case_id": chosen["case_id"].to_numpy(), "k": chosen["k"].to_numpy(),
            "stage": chosen["stage"].to_numpy(), "adam_step": chosen["adam_step"].to_numpy(),
            "relL2_uv": chosen["relL2_uv"].to_numpy(), "score_value": chosen[score].to_numpy(),
            "score_over_tau": chosen[score].to_numpy() / tau,
            "passed_acceptance": chosen["passed_acceptance"].to_numpy(dtype=bool),
        }))
    if not frames:
        return pd.DataFrame(columns=list(LIST_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(LIST_COLUMNS)]


def missed_failures(table: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Every inaccurate checkpoint a score does not flag (score <= tau), one row per checkpoint x score."""
    return _score_rows(table, thresholds, lambda t, score: t["inaccurate"] & ~t[f"flagged_{score}"])


def false_alarms(table: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Every accurate checkpoint a score flags (score > tau), one row per checkpoint x score."""
    return _score_rows(table, thresholds, lambda t, score: ~t["inaccurate"] & t[f"flagged_{score}"])


def inaccurate_final_models(table: pd.DataFrame) -> pd.DataFrame:
    """Every inaccurate final model (lbfgs_final), with its scores and flags, apart from the Adam stages."""
    final = table[(table["stage"] == FINAL_STAGE) & table["inaccurate"]]
    return final[list(FINAL_COLUMNS)].reset_index(drop=True)


def make_tables(table: pd.DataFrame, thresholds: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """
    Builds every output table from the checkpoint table.

    Outputs:
        dict: {table name: DataFrame}.
    """
    return {
        SCORES_TABLE: table,
        "table_stage_by_k": stage_by_k_table(table, thresholds, n_boot, seed),
        "missed_failures": missed_failures(table, thresholds),
        "false_alarms": false_alarms(table, thresholds),
        "inaccurate_final_models": inaccurate_final_models(table),
    }


# --- Writing the tables ---

def _significant(value, digits: int) -> str:
    """value with `digits` significant digits, in plain notation between 1e-3 and 1e6; "--" if NaN."""
    if pd.isna(value):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(value))))
    if -3 <= magnitude < 6:
        return f"{value:.{max(digits - 1 - magnitude, 0)}f}"
    return f"{value:.{digits - 1}e}"


# Columns the .tex files give with significant digits instead of list_latex's 3 decimals,
# since the errors span 0.0015 to 0.85 and the scores 1 to about 1e4
_TEX_DIGITS = {"relL2_uv": 3, "tau": 4, "score_value": 4, "score_over_tau": 4,
               **{column: 4 for column in (*BAR_NAMES, *SCORE_NAMES)}}

CAPTIONS = {
    SCORES_TABLE: "The clean field of every checkpoint of the 30 new cases (7 stages each): raw and normalized "
                  "components, the five scores and their flags (score above $\\tau$, frozen calibration), the "
                  "velocity error relL2\\_uv against the exact solution, inaccurate (relL2\\_uv above 0.10) and "
                  "passed\\_acceptance of the run.",
    "table_stage_by_k": "Natural errors, by stage and $k$: checkpoints, inaccurate (relL2\\_uv above 0.10), flagged "
                        "(score above $\\tau$), missed (inaccurate, not flagged) and false alarms (accurate, flagged). "
                        "Recall: flagged inaccurate checkpoints / inaccurate checkpoints (one minus the miss rate). "
                        "False-alarm rate: false alarms / accurate checkpoints. Each cell: estimate [95\\% range]. "
                        "One stage (one checkpoint per case): exact Clopper--Pearson range. All stages: case "
                        "bootstrap, 2000 resamples of whole cases within each $k$; where every (or no) checkpoint "
                        "counted is flagged, no resample can change the rate and the range has zero width, so the "
                        "exact ranges of the single stages show the uncertainty. -- : nothing to count.",
    "missed_failures": "Every inaccurate checkpoint (relL2\\_uv above 0.10) that a score does not flag (score at or "
                       "below $\\tau$), one row per checkpoint and score.",
    "false_alarms": "Every accurate checkpoint (relL2\\_uv at most 0.10) that a score flags (score above $\\tau$), "
                    "one row per checkpoint and score. An accurate Adam checkpoint can still be far less accurate "
                    "than the final models.",
    "inaccurate_final_models": "Every final model (lbfgs\\_final) that is inaccurate (relL2\\_uv above 0.10), with its "
                               "five scores and flags, kept apart from the half-trained Adam stages.",
}


def list_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """A per-checkpoint table as a LaTeX longtable: benchmark_metrics.list_latex, with the errors and scores
    in significant digits."""
    formatted = table.copy()
    for column, digits in _TEX_DIGITS.items():
        if column in formatted.columns:
            formatted[column] = [_significant(value, digits) for value in formatted[column]]
    return bm.list_latex(formatted, caption, label)


def stage_by_k_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """The stage x k table as LaTeX: the counts, and each rate as 'estimate [range]'."""
    rows = []
    for _, row in table.iterrows():
        rows.append({
            "Score": bm._tex_escape(row["score"]), "Stage": bm._tex_escape(row["stage"]), "$k$": row["k"],
            "Failed runs": str(int(row["n_failed_runs"])), "Checkpoints": str(int(row["n_checkpoints"])),
            "Inaccurate": str(int(row["n_inaccurate"])), "Flagged": str(int(row["n_flagged"])),
            "Missed": str(int(row["n_missed"])), "False alarms": str(int(row["n_false_alarms"])),
            "Recall": bm._with_range(row, "recall"), "False-alarm rate": bm._with_range(row, "fpr"),
        })
    return pd.DataFrame(rows).to_latex(index=False, escape=False, longtable=len(rows) > 40, caption=caption,
                                       label=label)


def to_latex(name: str, table: pd.DataFrame) -> str:
    """The LaTeX of one output table."""
    label = f"tab:natural_errors_{name}"
    caption = CAPTIONS[name]
    if name == "table_stage_by_k":
        return stage_by_k_latex(table, caption, label)
    if name == "inaccurate_final_models" and table.empty:
        caption += " There is none: every final model is accurate."
    return list_latex(table, caption, label)


def write_tables(tables: dict, output_dir) -> list:
    """
    Writes each table as <name>.csv (full precision, so it reads back exactly; "NaN"
    where undefined) and <name>.tex, with "\\n" line endings on every OS.

    Outputs:
        list[Path]: The files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        table.to_csv(csv_path, index=False, na_rep="NaN", lineterminator="\n")
        tex_path = output_dir / f"{name}.tex"
        with open(tex_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(to_latex(name, table))
        written += [csv_path, tex_path]
    return written


# --- Figures ---

# One-hue ordinal ramp for the 7 stages (light = early, dark = final), checked with the
# dataviz validator: monotone lightness, adjacent steps >= 0.06 apart in OKLCH L, the
# lightest at 2.09:1 against white
STAGE_COLOURS = dict(zip(STAGES, ("#7cb4fc", "#599bee", "#4283d4", "#2a6cba", "#0e55a1", "#04407f", "#032d5c")))
K_MARKERS = {1: "o", 2: "s", 3: "^"}
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_COLOUR, AXIS_COLOUR, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
FAILED_COLOUR = "#d03b3b"   # Status "critical", only for the ring of a failed run
SERIES_COLOUR = "#2a78d6"
FIGURE_RC = {
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "axes.edgecolor": AXIS_COLOUR, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,  # Embed TrueType fonts, as journals ask
}


def stage_label(stage: str) -> str:
    """'Adam, 100 steps', ..., 'L-BFGS (final)'."""
    if stage == FINAL_STAGE:
        return "L-BFGS (final)"
    return f"Adam, {int(stage.split('_')[1])} steps"


def short_score(score: str) -> str:
    """'Score3' for 'Score3_PHS_full'."""
    return score.split("_")[0]


def _save(fig, output_dir: Path, stem: str) -> list:
    """Saves a figure as <stem>.pdf and <stem>.png; the PDF without a creation date, so a rerun gives the same file."""
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=200)
    return [pdf_path, png_path]


def plot_score_vs_error(table: pd.DataFrame, thresholds: dict, output_dir) -> list:
    """
    fig_score_vs_error: each score against relL2_uv, one log-log panel per score, with
    tau (horizontal) and the 10% line (vertical). The shaded corners hold the missed
    failures (inaccurate, at or below tau) and the false alarms (accurate, above tau),
    with their counts. Colour = stage, marker = k; the checkpoints of failed runs are ringed.

    Outputs:
        list[Path]: The .pdf and .png written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x = table["relL2_uv"].to_numpy(dtype=float)
    failed = ~table["passed_acceptance"].to_numpy(dtype=bool)
    xlim = (x.min() / 1.6, x.max() * 1.6)
    with plt.rc_context(FIGURE_RC):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharex=True)
        for ax, score in zip(axes.flat, SCORE_NAMES):
            y = table[score].to_numpy(dtype=float)
            tau = thresholds[score]
            ylim = (min(y.min(), tau) / 2.5, max(y.max(), tau) * 2.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.fill_between([INACCURATE_ABOVE, xlim[1]], ylim[0], tau, color=WASH, linewidth=0, zorder=0)
            ax.fill_between([xlim[0], INACCURATE_ABOVE], tau, ylim[1], color=WASH, linewidth=0, zorder=0)
            ax.grid(True, which="major", color=GRID_COLOUR, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.axhline(tau, color=INK_SECONDARY, linewidth=1, zorder=1)
            ax.axvline(INACCURATE_ABOVE, color=INK_SECONDARY, linewidth=1, zorder=1)
            for stage in STAGES:
                for k, marker in K_MARKERS.items():
                    chosen = ((table["stage"] == stage) & (table["k"] == k)).to_numpy()
                    ax.scatter(x[chosen], y[chosen], s=26, marker=marker, color=STAGE_COLOURS[stage],
                               edgecolors="white", linewidths=0.5, zorder=3)
            if failed.any():
                ax.scatter(x[failed], y[failed], s=110, facecolors="none", edgecolors=FAILED_COLOUR,
                           linewidths=1.3, zorder=4)
            flagged = table[f"flagged_{score}"].to_numpy(dtype=bool)
            inaccurate = table["inaccurate"].to_numpy(dtype=bool)
            ax.text(xlim[1] / 1.12, ylim[0] * 1.4, f"missed: {int((inaccurate & ~flagged).sum())}",
                    ha="right", va="bottom", color=INK_SECONDARY, fontsize=8.5)
            ax.text(xlim[0] * 1.12, ylim[1] / 1.4, f"false alarms: {int((~inaccurate & flagged).sum())}",
                    ha="left", va="top", color=INK_SECONDARY, fontsize=8.5)
            ax.text(xlim[0] * 1.12, tau * 1.15, f"τ = {tau:.3g}", ha="left", va="bottom",
                    color=INK_SECONDARY, fontsize=8)
            ax.set_title(score, color=INK)
            ax.set_ylabel("score")
        # The top-right panel has the legend below it, so it gets its own x labels
        axes[0, 2].xaxis.set_tick_params(labelbottom=True)
        for ax in (*axes[-1, :2], axes[0, 2]):
            ax.set_xlabel("relL2_uv (velocity error against the exact solution)")

        legend_ax = axes.flat[len(SCORE_NAMES)]
        legend_ax.axis("off")
        stage_handles = [Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor=STAGE_COLOURS[stage],
                                markeredgecolor="white", label=stage_label(stage)) for stage in STAGES]
        other_handles = [Line2D([], [], linestyle="", marker=marker, markersize=7, markerfacecolor=INK_MUTED,
                                markeredgecolor="white", label=f"k = {k}") for k, marker in K_MARKERS.items()]
        other_handles += [Line2D([], [], color=INK_SECONDARY, linewidth=1, label="τ, and 10% error")]
        if failed.any():
            other_handles.append(Line2D([], [], linestyle="", marker="o", markersize=10, markerfacecolor="none",
                                        markeredgecolor=FAILED_COLOUR, markeredgewidth=1.3, label="failed run"))
        stage_legend = legend_ax.legend(handles=stage_handles, title="Stage", loc="upper left",
                                        bbox_to_anchor=(0.0, 1.0), frameon=False, alignment="left")
        legend_ax.add_artist(stage_legend)
        legend_ax.legend(handles=other_handles, title="Marker, lines", loc="upper left", bbox_to_anchor=(0.56, 1.0),
                         frameon=False, alignment="left")
        n_runs = table["case_id"].nunique()
        n_failed = int(table.loc[failed, "case_id"].nunique())
        note = (f"{n_failed} of {n_runs} runs failed (ringed)." if n_failed
                else f"All {n_runs} runs passed acceptance, so no\ncheckpoint is ringed as a failed run.")
        legend_ax.text(0.0, 0.0, f"{len(table)} checkpoints: {n_runs} cases × {len(STAGES)} stages.\n"
                                 f"Shaded: missed failures (right of 10%, at or below τ)\n"
                                 f"and false alarms (left of 10%, above τ).\n{note}",
                       transform=legend_ax.transAxes, ha="left", va="bottom", color=INK_SECONDARY, fontsize=8.5)
        fig.suptitle("Checkpoint scores against the true velocity error (frozen calibration, flagged above τ)",
                     color=INK)
        fig.tight_layout()
        paths = _save(fig, output_dir, "fig_score_vs_error")
        plt.close(fig)
    return paths


def _median_row(rows: pd.DataFrame) -> pd.Series:
    """
    The row at the median of relL2_uv: the middle one when sorted by relL2_uv, the lower
    of the two middle ones for an even count (so no tie is left to rounding).
    """
    return rows.sort_values("relL2_uv", kind="stable").iloc[(len(rows) - 1) // 2]


def choose_field_checkpoints(table: pd.DataFrame, max_missed: int = MAX_MISSED_FIELD_PLOTS) -> list:
    """
    Chooses the checkpoints whose fields are plotted, by a fixed rule:
      1. The missed failures: every checkpoint that some score misses, if there are at
         most max_missed of them; otherwise the worst miss of each score (its missed
         checkpoint with the largest relL2_uv).
      2. A typical caught failure: the inaccurate checkpoint PHS flags at the median of
         their relL2_uv (_median_row: the lower middle one for an even count).
      3. The failure closest to being missed: the inaccurate checkpoint PHS flags with
         the lowest PHS score.
      4. A typical final model: the lbfgs_final model at the median of their relL2_uv.
      5. A typical false alarm on a half-trained model: the accurate Adam checkpoint PHS
         flags at the median of their relL2_uv.
      6. The highest false alarm on a final model: the accurate lbfgs_final model PHS
         flags with the highest PHS score.
    A rule with nothing to choose from is skipped, and a checkpoint chosen twice is
    plotted once, with both reasons. (3 and 5 were added after the first full run
    showed no missed failure, so that the plots still show the cases nearest to tau.)

    Outputs:
        list of ((case_id, stage), [reason, ...]), in the order above.
    """
    chosen = {}

    def add(row, reason):
        chosen.setdefault((row["case_id"], row["stage"]), []).append(reason)

    inaccurate = table["inaccurate"].to_numpy(dtype=bool)
    missed_by = {score: inaccurate & ~table[f"flagged_{score}"].to_numpy(dtype=bool) for score in SCORE_NAMES}
    missed_by_any = np.logical_or.reduce(list(missed_by.values()))
    if missed_by_any.sum() <= max_missed:
        for i in np.flatnonzero(missed_by_any):
            row = table.iloc[i]
            add(row, "missed by " + ", ".join(short_score(s) for s in SCORE_NAMES if missed_by[s][i]))
    else:
        for score in SCORE_NAMES:
            missed = table[missed_by[score]]
            if not missed.empty:
                add(missed.loc[missed["relL2_uv"].idxmax()], f"worst miss of {short_score(score)}")

    flagged_by_phs = table[f"flagged_{PHS}"].to_numpy(dtype=bool)
    caught = table[inaccurate & flagged_by_phs]
    if not caught.empty:
        add(_median_row(caught), "typical failure caught by PHS")
        add(caught.loc[caught[PHS].idxmin()], "failure closest to being missed by PHS")
    final = table[table["stage"] == FINAL_STAGE]
    if not final.empty:
        add(_median_row(final), "typical final model")
    alarms = table[~inaccurate & flagged_by_phs]
    on_adam = alarms[alarms["stage"] != FINAL_STAGE]
    if not on_adam.empty:
        add(_median_row(on_adam), "typical false alarm of PHS on an Adam checkpoint")
    on_final = alarms[alarms["stage"] == FINAL_STAGE]
    if not on_final.empty:
        add(on_final.loc[on_final[PHS].idxmax()], "highest false alarm of PHS on a final model")
    return list(chosen.items())


def field_data(model, case: dict) -> dict:
    """
    The model and the exact solution on the accuracy audit's grid (PROTOCOL.md section
    6: 64 x 64 points at 11 times from 0 to T), the pressures without their mean at each time.

    Outputs:
        dict: T; t (the 11 times); relL2_uv over the grid (as the audit computes it);
              relL2_uv_t, the velocity error at each time; E and E_exact, the kinetic
              energy at each time; fields, {"u", "v", "p": (model, exact)} at t = T, as
              numpy arrays indexed [x, y].
    """
    U0, Re, k = case["U0"], case["Re"], case["k"]
    T = compute_T(U0, Re, k)
    t, x, y = accuracy_grid(T)
    coords = torch.stack([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(coords).to(device="cpu", dtype=torch.float64).reshape(*t.shape, 3)
    u, v, p = pred[..., 0], pred[..., 1], pred[..., 2]
    u_exact, v_exact, p_exact = generate_tgv(x, y, t, U0, k, case["phi_x"], case["phi_y"], compute_nu(U0, Re, k))
    p = p - p.mean(dim=(1, 2), keepdim=True)
    p_exact = p_exact - p_exact.mean(dim=(1, 2), keepdim=True)
    return {
        "T": T,
        "t": t[:, 0, 0].numpy(),
        "relL2_uv": relative_l2((u, v), (u_exact, v_exact)),
        "relL2_uv_t": np.array([relative_l2((u[i], v[i]), (u_exact[i], v_exact[i])) for i in range(t.shape[0])]),
        "E": (0.5 * (u ** 2 + v ** 2).mean(dim=(1, 2))).numpy(),
        "E_exact": (0.5 * (u_exact ** 2 + v_exact ** 2).mean(dim=(1, 2))).numpy(),
        "fields": {name: (model_values[-1].numpy(), exact_values[-1].numpy())
                   for name, model_values, exact_values in (("u", u, u_exact), ("v", v, v_exact), ("p", p, p_exact))},
    }


def plot_fields(checkpoint: pd.Series, reasons: list, data: dict, thresholds: dict, output_dir) -> list:
    """
    fig_fields_<case_id>_<stage>: u, v and p of the checkpoint and of the exact solution
    at t = T, and their difference; the kinetic energy and the velocity error at each
    time; and each score over its tau.

    Inputs:
        checkpoint (pd.Series): The checkpoint's row of the checkpoint table.
        reasons (list[str]): Why it was chosen (choose_field_checkpoints).
        data (dict): Output of field_data().
        thresholds (dict): {score: tau}.
        output_dir (str | Path): Where to write.

    Outputs:
        list[Path]: The .pdf and .png written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    ticks, tick_labels = [0, np.pi, 2 * np.pi], ["0", "π", "2π"]
    t_over_T = data["t"] / data["T"]
    with plt.rc_context(FIGURE_RC):
        fig = plt.figure(figsize=(13.5, 8.6))
        grid = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.62, hspace=0.55)
        for i, name in enumerate(("u", "v", "p")):
            model_values, exact_values = data["fields"][name]
            difference = model_values - exact_values
            bound = float(max(np.abs(model_values).max(), np.abs(exact_values).max())) or 1.0
            difference_bound = float(np.abs(difference).max()) or 1.0
            panels = ((model_values, f"{name}, model", bound), (exact_values, f"{name}, exact", bound),
                      (difference, f"{name}, model − exact", difference_bound))
            for j, (values, title, limit) in enumerate(panels):
                ax = fig.add_subplot(grid[i, j])
                image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                  interpolation="nearest")
                ax.set_title(title)
                ax.set_xticks(ticks, tick_labels)
                ax.set_yticks(ticks, tick_labels)
                if i == 2:
                    ax.set_xlabel("x")
                if j == 0:
                    ax.set_ylabel("y")
                colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                colorbar.formatter.set_powerlimits((-2, 3))  # 0.002 -> 2 with a "1e-3" offset, so labels stay short
                colorbar.update_ticks()

        ax = fig.add_subplot(grid[0, 3])
        ax.plot(t_over_T, data["E_exact"], color=INK_MUTED, linewidth=1.5, label="exact")
        ax.plot(t_over_T, data["E"], color=SERIES_COLOUR, linewidth=2, marker="o", markersize=3.5, label="model")
        ax.set_title("Kinetic energy E(t)")
        ax.set_xlabel("t / T")
        ax.grid(True, color=GRID_COLOUR, linewidth=0.5)
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[1, 3])
        ax.plot(t_over_T, 100 * data["relL2_uv_t"], color=SERIES_COLOUR, linewidth=2, marker="o", markersize=3.5)
        ax.axhline(100 * INACCURATE_ABOVE, color=INK_SECONDARY, linewidth=1)
        # Labelled just outside the right edge, where no point can hide it
        ax.text(1.02, 100 * INACCURATE_ABOVE, "10%", transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=INK_SECONDARY, fontsize=8)
        ax.set_ylim(0, max(100 * data["relL2_uv_t"].max(), 100 * INACCURATE_ABOVE) * 1.15)
        ax.set_title("Velocity error at each time (%)")
        ax.set_xlabel("t / T")
        ax.grid(True, color=GRID_COLOUR, linewidth=0.5)

        ax = fig.add_subplot(grid[2, 3])
        ratios = np.array([checkpoint[score] / thresholds[score] for score in SCORE_NAMES])
        flagged = [bool(checkpoint[f"flagged_{score}"]) for score in SCORE_NAMES]
        positions = np.arange(len(SCORE_NAMES))[::-1]
        # At least 0.1 to 10, so the axis always has whole-decade ticks and no crowded minor labels
        low, high = min(ratios.min() / 3, 0.1), max(ratios.max() * 3, 10.0)
        ax.set_xscale("log")
        ax.set_xlim(low, high)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvline(1.0, color=INK_SECONDARY, linewidth=1)
        ax.hlines(positions, low, ratios, color=GRID_COLOUR, linewidth=1.5)
        ax.scatter(ratios, positions, s=42, color=[SERIES_COLOUR if f else "white" for f in flagged],
                   edgecolors=SERIES_COLOUR, linewidths=1.5, zorder=3)
        # The values in a column just outside the right edge, so they never cover a dot
        for position, ratio in zip(positions, ratios):
            ax.text(1.03, position, _significant(ratio, 3), transform=ax.get_yaxis_transform(), va="center",
                    ha="left", color=INK, fontsize=8)
        ax.set_yticks(positions, [short_score(score) for score in SCORE_NAMES])
        ax.set_ylim(-0.7, len(SCORE_NAMES) - 0.3)
        ax.set_title("Score / τ (filled: flagged, above 1)")
        ax.grid(True, axis="x", color=GRID_COLOUR, linewidth=0.5)

        status = "inaccurate" if checkpoint["inaccurate"] else "accurate"
        fig.suptitle(f"{checkpoint['case_id']}, {stage_label(checkpoint['stage'])} (k = {checkpoint['k']}): velocity "
                     f"error {100 * checkpoint['relL2_uv']:.3g}% ({status}); fields at t = T\n"
                     f"Chosen as: {'; '.join(reasons)}", color=INK)
        paths = _save(fig, output_dir, f"fig_fields_{checkpoint['case_id']}_{checkpoint['stage']}")
        plt.close(fig)
    return paths


def plot_field_checkpoints(table: pd.DataFrame, thresholds: dict, metadata_path, models_dir, output_dir) -> list:
    """
    Plots the fields of the checkpoints choose_field_checkpoints() picks, after removing
    the field plots of an earlier run. Each model is checked against its sidecar, and
    its velocity error on the plotted grid must be the true error of the table, so a
    plot never shows another model than the one the table describes.

    Outputs:
        list[Path]: The files written.
    """
    output_dir = Path(output_dir)
    for stale in list(output_dir.glob("fig_fields_*.pdf")) + list(output_dir.glob("fig_fields_*.png")):
        stale.unlink()
    case_by_id = {case["case_id"]: case for case in read_cases(metadata_path)}
    written = []
    for (case_id, stage), reasons in choose_field_checkpoints(table):
        case = case_by_id[case_id]
        path = Path(models_dir) / case_id / f"{stage}.pth"
        model, info = load_checkpoint(path, k=case["k"], metadata_paths=[metadata_path])
        check_sidecar(info, case, stage, path)
        data = field_data(model, case)
        checkpoint = table[(table["case_id"] == case_id) & (table["stage"] == stage)].iloc[0]
        if not np.isclose(data["relL2_uv"], checkpoint["relL2_uv"], rtol=1e-9, atol=0.0):
            raise ValueError(f"{path}: velocity error {data['relL2_uv']} on the plotted grid, but the table says "
                             f"{checkpoint['relL2_uv']}")
        print(f"  fields of {case_id} {stage}: {'; '.join(reasons)}")
        written += plot_fields(checkpoint, reasons, data, thresholds, output_dir)
    return written


# --- Running it ---

def run(metadata=DEFAULT_METADATA, models_dir=DEFAULT_MODELS_DIR, calibration_json=DEFAULT_CALIBRATION_JSON,
        errors_csv=DEFAULT_ERRORS_CSV, benchmark_csv=DEFAULT_BENCHMARK_CSV, output_dir=DEFAULT_OUTPUT_DIR,
        device: str = "cpu", from_scores: bool = False, figures: bool = True, n_boot: int = N_BOOT,
        seed: int = BOOT_SEED):
    """
    Scores the checkpoints (or, with from_scores, reads <output_dir>/checkpoint_scores.csv),
    runs the consistency check, and writes every table and figure. See the module docstring.

    Outputs:
        (tables, consistency, written): {table name: DataFrame} as written, the output of
            consistency_check(), and the list of files written.
    """
    calib = frozen_calibration.load(calibration_json)
    errors = read_true_errors(errors_csv)
    cases = read_cases(metadata)
    output_dir = Path(output_dir)
    if from_scores:
        table = read_checkpoint_scores(output_dir / f"{SCORES_TABLE}.csv", calib, errors)
    else:
        raw = score_checkpoints(metadata, models_dir, scoring_settings(device))
        table = build_checkpoint_table(raw, calib, errors)
    check_coverage(table, cases)
    consistency = consistency_check(table, read_benchmark(benchmark_csv))

    tables = make_tables(table, calib["thresholds"], n_boot, seed)
    written = write_tables(tables, output_dir)
    if figures:
        written += plot_score_vs_error(table, calib["thresholds"], output_dir)
        written += plot_field_checkpoints(table, calib["thresholds"], metadata, models_dir, output_dir)
    return tables, consistency, written


def print_summary(tables: dict, consistency: dict) -> None:
    """Prints the consistency check and the headline numbers of each score."""
    print(f"Consistency with #35: {consistency['n_fields']} lbfgs_final fields, every flag the same; largest "
          f"relative difference {consistency['max_rel_diff']:.3g} ({consistency['column']}, "
          f"{consistency['case_id']}), allowed {CONSISTENCY_RTOL:g}")
    table = tables[SCORES_TABLE]
    n_failed = int((~table.groupby("case_id")["passed_acceptance"].first().astype(bool)).sum())
    n_final = int((table["stage"] == FINAL_STAGE).sum())
    print(f"{len(table)} checkpoints ({table['case_id'].nunique()} cases x {len(STAGES)} stages): "
          f"{int(table['inaccurate'].sum())} inaccurate (relL2_uv > {INACCURATE_ABOVE}), "
          f"{len(tables['inaccurate_final_models'])} of them among the {n_final} final models; "
          f"{n_failed} failed run(s), kept in every table")
    by_stage = tables["table_stage_by_k"]
    for score in SCORE_NAMES:
        pooled = by_stage[(by_stage["score"] == score) & (by_stage["stage"] == ALL) & (by_stage["k"] == ALL)].iloc[0]
        final = by_stage[(by_stage["score"] == score) & (by_stage["stage"] == FINAL_STAGE)
                         & (by_stage["k"] == ALL)].iloc[0]
        n_accurate = pooled["n_checkpoints"] - pooled["n_inaccurate"]
        print(f"  {score:28s} missed {pooled['n_missed']}/{pooled['n_inaccurate']} inaccurate "
              f"(recall {bm._with_range(pooled, 'recall')}); false alarms {pooled['n_false_alarms']}/{n_accurate} "
              f"accurate ({final['n_false_alarms']} on final models)")


def parse_args(argv=None):
    """
    Parses the command line.

    Outputs:
        argparse.Namespace: metadata, models_dir, calibration, errors, benchmark,
                            output_dir, device (str) and from_scores (bool).
    """
    parser = argparse.ArgumentParser(description="Score the clean field of every checkpoint of the 30 new cases "
                                                 "and hold the scores against the true velocity error (#39).")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="The case file; every case in it is scored (default: cases/confirmation_cases.json).")
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Folder with one folder of 7 checkpoints per case (default: models/confirmation).")
    parser.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_JSON),
                        help="The frozen calibration (default: calibration/frozen_calibration.json).")
    parser.add_argument("--errors", type=str, default=str(DEFAULT_ERRORS_CSV),
                        help="The true errors of #37 (default: results/accuracy/reference_errors_confirmation.csv).")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK_CSV),
                        help="#35's per-field table, for the consistency check "
                             "(default: results/benchmark/confirmation_scores_raw.csv).")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Folder to write the tables and figures into (default: results/natural_errors).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Where to score the checkpoints (float64 on either; PROTOCOL.md section 4).")
    parser.add_argument("--from_scores", action="store_true",
                        help=f"Don't score: read {{output_dir}}/{SCORES_TABLE}.csv, check that it follows from its "
                             "raw components, and rebuild the other tables and the figures from it.")
    return parser.parse_args(argv)


def main(argv=None):
    """Scores the checkpoints (unless --from_scores), writes every table and figure, and prints a summary."""
    args = parse_args(argv)
    tables, consistency, written = run(args.metadata, args.models_dir, args.calibration, args.errors, args.benchmark,
                                       args.output_dir, device=args.device, from_scores=args.from_scores)
    print(f"Wrote {len(written)} files to {args.output_dir}")
    print_summary(tables, consistency)


if __name__ == "__main__":
    enable_utf8_output()
    main()
