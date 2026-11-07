"""
The final figures and the case table of the final iteration (#36).

Barak asks for "figures in PDF/PNG, tables in CSV/LaTeX". This script turns the
saved results of the perturbation benchmark (#35) and of the natural errors (#39)
on the 30 new cases into the paper's figures, puts the two experiments side by
side in the same terms, and writes the table of the 30 cases. It reads only saved
tables, never a model, and fits nothing:

    results/benchmark/confirmation_scores_raw.csv   every field: its scores and flags (#35)
    results/benchmark/metrics_overall.csv           the AUC of each score and its range (#35)
    results/benchmark/metrics_by_k.csv              the same by k, for the captions (#35)
    results/benchmark/metrics_by_type.csv           the same by damage type, for the captions (#35)
    results/benchmark/metrics_by_type_eps.csv       recall by damage type x strength, exact ranges (#35)
    results/natural_errors/checkpoint_scores.csv    every checkpoint: its scores, flags and true error (#39)
    results/natural_errors/table_stage_by_k.csv     #39's counts, recall and false-alarm rate, with ranges
    results/accuracy/reference_errors_confirmation.csv   the true velocity error of each model (#37)
    cases/confirmation_cases.json                   the cases and their seeds (#44)
    models/confirmation/loss_summary.csv            the training acceptance test of each run (#40)
    calibration/frozen_calibration.json             the alarm level tau of each score (#45)
    calibration/source_phs_components_raw.csv       the 30 old clean fields tau was fitted on (#45)

Before drawing anything it checks that these tables agree: rescoring the raw
components with the frozen calibration must give back every normalized
component, score and flag of the benchmark and of the checkpoints bit for bit;
the benchmark must hold every case of the case file, each with its 31 fields,
and the checkpoints the 7 of every case; the final models must be the
benchmark's clean fields; #35's metrics tables must match the flags (counts) and
the scores (AUC), and #39's counts and rates must follow from its flags; and
the pass/fail of each run must be the same in loss_summary.csv, the benchmark and
the accuracy audit, and must follow from its acceptance metrics. A mismatch stops
the run, since a figure built from tables that disagree would be wrong without
anyone noticing.

To put both experiments on one axis, it sizes each synthetic damage as a velocity
error: the benchmark's own damage code (perturbations.apply_perturbation) applied
to the exact solution on the accuracy audit's test grid. The same code, with
autograd residuals, draws what each damage does to the flow. That needs the case
file and the damage formulas, no model.

Outputs, in results/figures/. Each figure as .pdf and .png (300 dpi), sized for
a two-column page (7.16 in wide, one column 3.5 in), plus a .tex file holding
its figure environment and caption. The PDF, for the paper, has no title, since
its caption is the title; the PNG has a one-line title (PNG_TITLES), so that it
can be read on its own. Each table as .csv (full precision) and .tex:
    fig_recall_vs_eps                  recall vs damage strength: one panel per damage type,
                                       one line per score, exact 95% ranges as bands
    fig_recall_vs_eps_new_detections   the same, over the cases whose clean field the score
                                       doesn't already flag
    fig_recall_vs_eps_by_k             the same as fig_recall_vs_eps, split by k (10 fields a point)
    fig_roc                            the ROC curves of the five scores, each AUC with its
                                       range, and the point of each frozen tau
    fig_clean_scores_vs_tau            the clean scores of the 30 new cases against tau, by k
                                       and for each score, every false alarm named, with the
                                       30 old clean fields tau was fitted on for reference
    fig_natural_score_vs_error         the natural errors (#39's figure, in this style): the
                                       score / tau of each checkpoint against its true velocity
                                       error, by training stage and k, with the 10% rule
    fig_score_vs_velocity_error        both experiments on one axis: what a deviation adds to
                                       each score against its size, for each damaged field (the
                                       velocity change of its damage, its rise over its clean
                                       field) and each checkpoint (its own error, its score)
    fig_damage_fields                  what each damage does to the exact solution of one case:
                                       the change, and the momentum and continuity residuals
    fig_energy_term_vs_u0              the energy term of the 60 clean fields against U0, and
                                       the relative energy error, with the energy false alarms
    table_recall_by_type_eps_k         the numbers behind fig_recall_vs_eps_by_k, computed from
                                       the flags of confirmation_scores_raw.csv (#35's tables
                                       have no type x strength x k breakdown)
    table_cases                        one row per new case: k, U0, Re, nu, decay time, T, phases,
                                       the three seeds, acceptance pass/fail, and the final
                                       model's velocity errors (overall, at T) and pressure error
    table_synthetic_vs_natural         both experiments in the same terms, overall and by k: AUC,
                                       recall and false alarms at tau, with their ranges
    table_damage_size                  the size of each synthetic damage as a velocity error, by
                                       damage type and strength (over the 30 cases)
    table_term_auc                     the AUC of each score term on its own, by damage type and
                                       on the natural errors, with confidence intervals
The captions are written from the numbers, so they can't drift from the figures.
A paper includes a figure with \\input{fig_roc.tex}, with results/figures/ in its
\\graphicspath. Two runs on the same machine give the same files, byte for byte.

Usage:
    python src/detection/final_figures.py
    python src/detection/final_figures.py --benchmark_dir results/benchmark --output_dir results/figures
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detection import benchmark_metrics as bm
from src.detection import frozen_calibration
from src.detection import natural_errors as ne
from src.detection.grouped_eval import case_bootstrap, clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.hallucinations.perturbations import PERTURBATION_NAMES, apply_perturbation
from src.models.accuracy_audit import INACCURATE_ABOVE, accuracy_grid, read_cases, relative_l2
from src.models.checkpoints import FINAL_STAGE
from src.models.scaling import ResidualScaler
from src.physics.navier_stokes import compute_residuals
from src.physics.taylor_green import compute_decay_timescale, compute_nu, compute_T, generate_tgv
from src.utils.console import enable_utf8_output

DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmark"
DEFAULT_NATURAL_DIR = PROJECT_ROOT / "results" / "natural_errors"
DEFAULT_ERRORS_CSV = PROJECT_ROOT / "results" / "accuracy" / "reference_errors_confirmation.csv"
DEFAULT_METADATA = PROJECT_ROOT / "cases" / "confirmation_cases.json"
DEFAULT_LOSS_SUMMARY = PROJECT_ROOT / "models" / "confirmation" / "loss_summary.csv"
DEFAULT_CALIBRATION_JSON = frozen_calibration.DEFAULT_CALIBRATION_JSON
DEFAULT_SOURCE_CSV = frozen_calibration.DEFAULT_SOURCE_CSV
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

FIELDS_CSV = "confirmation_scores_raw.csv"
NATURAL_SCORES_CSV = f"{ne.SCORES_TABLE}.csv"
NATURAL_STAGE_CSV = "table_stage_by_k.csv"
SCORE_NAMES = bm.SCORE_NAMES
BAR_NAMES = bm.BAR_NAMES
FLAG_NAMES = tuple(f"flagged_{name}" for name in SCORE_NAMES)
CLEAN = bm.CLEAN
PHS = "Score3_PHS_full"
ALL = ne.ALL                       # The k of the rows that pool every k
N_BOOT, BOOT_SEED = bm.N_BOOT, bm.BOOT_SEED
SYNTHETIC, NATURAL = "synthetic", "natural"   # The two experiments, in table_synthetic_vs_natural

# The training acceptance test of a final model (PROTOCOL.md section 2, train_model.evaluate_model)
ACCEPT_REL_L2_BELOW = 0.10
ACCEPT_MSE_RC_BELOW = 1e-4
ACCEPT_MSE_RM_BELOW = 1e-3

# --- Names, colours and sizes ---

SCORE_LABELS = {
    "Score1_momentum_only": "Score1: mom",
    "Score2_momentum_divergence": "Score2: mom + div",
    "Score3_PHS_full": "Score3: PHS",
    "Score4_PHS_no_bc": "Score4: PHS \u2212 bc",
    "Score5_PHS_no_E": "Score5: PHS \u2212 E",
}
# The first five categorical slots of the dataviz reference palette, in its fixed order. Checked with its
# validator for lines on white: lightness band, chroma floor, CVD separation (worst adjacent pair 9.1) and
# the normal-vision floor (19.6) pass. Aqua, yellow and magenta are below 3:1 against white, so each score
# also has its own marker shape, and every number is in a table.
SCORE_COLOURS = dict(zip(SCORE_NAMES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
SCORE_MARKERS = dict(zip(SCORE_NAMES, ("o", "s", "^", "D", "v")))
# Defines the short names of the score labels, for every caption that shows the scores
SCORE_KEY = ("Score terms: mom, the momentum residual; div, the continuity residual; bc, the boundary-localization "
             "term; E, the energy-decay term; PHS, the sum of all four, each divided by its mean over the calibration "
             "fields.")
DAMAGE_LABELS = {name: name.replace("_", " ") for name in PERTURBATION_NAMES}
K_COLOUR = SCORE_COLOURS[SCORE_NAMES[0]]   # The new clean fields (one series: the first slot)
OLD_COLOUR = ne.INK_MUTED                  # The old calibration fields, a grey reference
# Where both experiments share an axis: the checkpoints keep #39's stage ramp (ne.STAGE_COLOURS, light =
# early, dark = final) and its k markers, and the synthetic damage gets the reference palette's second slot.
# Against every step of the ramp the validator's CVD (worst dE 22.8) and normal-vision (29.6) checks pass;
# the damage types also differ by marker.
SYNTHETIC_COLOUR = "#eb6834"
DAMAGE_MARKERS = {"velocity_divergence": "s", "momentum": "D", "pressure": "P", "temporal_mismatch": "^",
                  "boundary": "v"}
NATURAL_MARKER = "o"

TEXT_WIDTH = 7.16    # in: a two-column page, as publication_figures.py uses
COLUMN_WIDTH = 3.5   # in: one column
PNG_DPI = 300
# natural_errors.py's style (#39: font, ink, grid and axis colours), at print size
PRINT_RC = {
    **ne.FIGURE_RC,
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    # No text below 7 pt at print size, the smallest many journals accept
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.titlesize": 7.5, "figure.labelsize": 7.5,
    "axes.linewidth": 0.6, "axes.titlepad": 4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
}
LINE_WIDTH = 1.1
MARKER_SIZE = 3.6
BAND_ALPHA = 0.09            # A score's own range: its colour, as a light wash
SHARED_BAND_COLOUR = ne.INK_MUTED
SHARED_BAND_ALPHA = 0.16     # A range several scores share, drawn once

FIGURES = ("fig_recall_vs_eps", "fig_recall_vs_eps_new_detections", "fig_recall_vs_eps_by_k", "fig_roc",
           "fig_clean_scores_vs_tau", "fig_natural_score_vs_error", "fig_score_vs_velocity_error", "fig_damage_fields",
           "fig_energy_term_vs_u0")
NARROW_FIGURES = ("fig_roc",)   # One column wide; the others span the page
TABLES = ("table_recall_by_type_eps_k", "table_cases", "table_synthetic_vs_natural", "table_damage_size",
          "table_term_auc")
# The title of each PNG, so it says what it shows when read on its own (a README, slides). The PDF, for
# the paper, has none: there the caption is the title.
PNG_TITLES = {
    "fig_recall_vs_eps": "Synthetic damage: recall at the frozen τ against damage strength",
    "fig_recall_vs_eps_new_detections": "Synthetic damage: recall of new detections (cases whose clean field "
                                        "is not flagged)",
    "fig_recall_vs_eps_by_k": "Synthetic damage: recall at the frozen τ against damage strength, by k",
    "fig_roc": "Synthetic damage: ROC curves, damaged vs clean fields",
    "fig_clean_scores_vs_tau": "The clean fields of the new final models against the frozen τ",
    "fig_natural_score_vs_error": "Natural errors: the score of each checkpoint against its true velocity error",
    "fig_score_vs_velocity_error": "Synthetic damage and natural errors: what a deviation adds to the score, "
                                   "by its size",
    "fig_damage_fields": "What each synthetic damage changes, and the residuals it creates",
    "fig_energy_term_vs_u0": "The energy term of the clean fields against the velocity scale U₀",
}
PNG_TITLE_BAND = 0.26   # in: the band above a PNG that holds its title
PNG_TITLE_SIZE = 8.5    # pt
SUMMARY_COLUMNS = ("experiment", "score", "k", "tau", "n_cases", "n_positive", "n_negative", "n_positive_flagged",
                   "n_missed", "n_false_alarms", "auc", "auc_low", "auc_high", "auc_n_boot", "recall", "recall_low",
                   "recall_high", "recall_n_boot", "fpr", "fpr_low", "fpr_high", "fpr_n_boot", "fpr_range_method")
DAMAGE_SIZE_COLUMNS = ("perturbation", "epsilon", "n_cases", "velocity_change_min", "velocity_change_median",
                       "velocity_change_max")
TERM_AUC_COLUMNS = ("term", "experiment", "group", "n_cases", "n_positive", "n_negative", "auc", "auc_low",
                    "auc_high", "auc_n_boot", "unchanged")
TERM_LABELS = dict(zip(BAR_NAMES, ("mom: momentum residual", "div: continuity residual",
                                   "bc: boundary localization", "E: energy decay")))
FIELD_EPSILON = 0.01   # The strength fig_damage_fields shows: from it on every score flags (almost) every field
FIELD_GRID = 128       # Points per side of its maps

RECALL_BY_K_COLUMNS = ("score", "k", "perturbation", "epsilon", "n_cases", "n_damaged", "n_damaged_flagged",
                       "recall", "recall_low", "recall_high", "n_preflagged_cases", "n_damaged_new",
                       "n_damaged_new_flagged", "recall_new", "recall_new_low", "recall_new_high")
CASE_COLUMNS = ("case_id", "k", "U0", "Re", "nu", "tau_decay", "T", "phi_x", "phi_y", "seed_points", "seed_train",
                "seed_scoring", "passed_acceptance", "rel_l2_eval_grid", "relL2_uv", "relL2_uv_T", "relL2_p")
CASE_ERROR_COLUMNS = ("relL2_uv", "relL2_uv_T", "relL2_p")   # Of the final model, from the accuracy audit (#37)


# --- Reading and checking the inputs ---

def read_fields(benchmark_dir, calib: dict, metadata) -> pd.DataFrame:
    """
    Reads #35's per-field table exactly (float_precision="round_trip") and refuses it
    unless it follows from its own raw components and the frozen calibration (every
    normalized component, score and flag bit for bit, and each preflagged_<score> the
    flag of its case's clean field), holds every case of the case file with its k, and
    has the benchmark's layout (one clean field and the same damaged fields per case).

    Inputs:
        benchmark_dir (str | Path): Folder with confirmation_scores_raw.csv (results/benchmark).
        calib (dict): The frozen calibration, from frozen_calibration.load().
        metadata (str | Path): The case file (cases/confirmation_cases.json).

    Outputs:
        pd.DataFrame: The table, one row per field.
    """
    path = Path(benchmark_dir) / FIELDS_CSV
    fields = pd.read_csv(path, float_precision="round_trip")
    required = ["case_id", "k", "perturbation", "epsilon", "label", "passed_acceptance", *PHS_COMPONENT_NAMES,
                *BAR_NAMES, *SCORE_NAMES, *FLAG_NAMES, *(f"preflagged_{name}" for name in SCORE_NAMES)]
    missing = [column for column in required if column not in fields.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}; write it with src/detection/benchmark_metrics.py")
    rescored = frozen_calibration.apply(fields[["case_id", *PHS_COMPONENT_NAMES]], calib)
    for column in (*BAR_NAMES, *SCORE_NAMES, *FLAG_NAMES):
        if not np.array_equal(rescored[column].to_numpy(), fields[column].to_numpy()):
            raise ValueError(f"Column {column} of {path} doesn't follow from its raw components and the frozen "
                             "calibration: it was scored with another calibration, or the file was changed")
    bm.check_layout(fields)
    bm.check_cases(fields, load_case_metadata(Path(metadata)))
    clean = fields[fields["label"] == "clean"].set_index("case_id")
    for name in SCORE_NAMES:
        if not fields["case_id"].map(clean[f"flagged_{name}"]).astype(bool).equals(
                fields[f"preflagged_{name}"].astype(bool)):
            raise ValueError(f"preflagged_{name} of {path} is not the flag of each case's clean field")
    return fields


def read_metrics(benchmark_dir, name: str) -> pd.DataFrame:
    """Reads one of #35's metrics tables (metrics_<name>.csv) exactly."""
    path = Path(benchmark_dir) / f"metrics_{name}.csv"
    return pd.read_csv(path, float_precision="round_trip")


def check_metrics(fields: pd.DataFrame, metrics: dict) -> None:
    """
    Refuses #35's metrics tables unless they agree with the per-field table: the
    counts of every row (damaged fields, flagged ones, new detections, false alarms)
    must be the flags' counts, and every AUC estimate must be the AUC of the scores
    (benchmark_metrics.auc, to 1e-12). So the figures, which draw lines from both,
    can't mix two runs of the benchmark.

    Inputs:
        fields (pd.DataFrame): Output of read_fields().
        metrics (dict): {"overall", "by_k", "by_type", "by_type_eps": DataFrame}, from read_metrics().
    """
    breakdowns = {"overall": ("overall", []), "by_k": ("k", ["k"]), "by_type": ("type", ["perturbation"]),
                  "by_type_eps": ("type_eps", ["perturbation", "epsilon"])}
    for name, (by, keys) in breakdowns.items():
        table = metrics[name]
        expected_rows = len(SCORE_NAMES) * len(list(bm._groups(fields, by)))
        if len(table) != expected_rows:
            raise ValueError(f"metrics_{name} has {len(table)} rows, the benchmark gives {expected_rows}")
        for score in SCORE_NAMES:
            for group_keys, group in bm._groups(fields, by):
                selected = table["score"] == score
                for key in keys:
                    selected &= table[key] == group_keys[key]
                if selected.sum() != 1:
                    raise ValueError(f"metrics_{name} has {int(selected.sum())} rows for {score} {group_keys}")
                row = table[selected].iloc[0]
                view = bm.score_view(group, score)
                damaged, flagged, preflagged = view["damaged"], view["flagged"], view["preflagged"]
                counts = {"n_clean": int((~damaged).sum()), "n_damaged": int(damaged.sum()),
                          "n_damaged_flagged": int((damaged & flagged).sum()),
                          "n_damaged_new": int((damaged & ~preflagged).sum()),
                          "n_damaged_new_flagged": int((damaged & ~preflagged & flagged).sum()),
                          "n_false_alarms": int((~damaged & flagged).sum())}
                differ = {column: (int(row[column]), count) for column, count in counts.items()
                          if int(row[column]) != count}
                if differ:
                    raise ValueError(f"metrics_{name}, {score} {group_keys}: the counts {differ} (table, flags) "
                                     "differ; rerun src/detection/benchmark_metrics.py")
                if not np.isclose(row["auc"], bm.auc(view), rtol=0, atol=1e-12):
                    raise ValueError(f"metrics_{name}, {score} {group_keys}: AUC {row['auc']} in the table, "
                                     f"{bm.auc(view)} from the scores")


def read_old_clean_fields(source_csv, calib: dict) -> pd.DataFrame:
    """
    The 30 old clean fields the calibration was fitted on (PROTOCOL.md section 5),
    scored with it: frozen_calibration.apply() on the clean rows of
    calibration/source_phs_components_raw.csv. Refuses a source that isn't the one
    the calibration names (its SHA-256), or that doesn't give 30 clean fields.
    """
    if frozen_calibration._sha256(source_csv) != calib["source_csv_sha256"]:
        raise ValueError(f"{source_csv} is not the source the calibration was fitted on (SHA-256 differs)")
    source = pd.read_csv(source_csv, float_precision="round_trip")
    clean = source[source["label"] == "clean"]
    if len(clean) != frozen_calibration.N_CLEAN_FIELDS or clean["case_id"].duplicated().any():
        raise ValueError(f"{source_csv} must have {frozen_calibration.N_CLEAN_FIELDS} clean fields, one per case")
    scored = frozen_calibration.apply(clean[["case_id", *PHS_COMPONENT_NAMES]], calib)
    return pd.concat([clean[["case_id", "k", "U0", "Re"]].reset_index(drop=True),
                      scored.drop(columns="case_id").reset_index(drop=True)], axis=1)


def read_case_inputs(metadata, errors_csv, loss_summary_csv) -> tuple:
    """Reads the case file (in file order), the true errors (#37) and the training summary (#40)."""
    cases = read_cases(metadata)
    errors = ne.read_true_errors(errors_csv)
    loss = pd.read_csv(loss_summary_csv, float_precision="round_trip")
    missing = [column for column in ("case_id", "rel_l2", "mse_rc", "mse_rm", "passed") if column not in loss.columns]
    if missing:
        raise ValueError(f"{loss_summary_csv} has no column(s) {missing}; it is written by train_model.py")
    return cases, errors, loss


def read_natural(natural_dir, calib: dict, errors: pd.DataFrame, cases: list, fields: pd.DataFrame) -> pd.DataFrame:
    """
    #39's table of the checkpoints (checkpoint_scores.csv), refused unless it follows
    from its own raw components, the frozen calibration and the true errors bit for bit
    (natural_errors.read_checkpoint_scores), holds the 7 checkpoints of every case of the
    case file in order (check_coverage), and its final models are the benchmark's clean
    fields (consistency_check). So both experiments are scored with the same calibration,
    and the final models are the same fields in both.

    Inputs:
        natural_dir (str | Path): Folder with checkpoint_scores.csv (results/natural_errors).
        calib (dict): The frozen calibration.
        errors (pd.DataFrame): The true errors (#37), from natural_errors.read_true_errors().
        cases (list[dict]): The cases, from accuracy_audit.read_cases().
        fields (pd.DataFrame): The benchmark, from read_fields().

    Outputs:
        pd.DataFrame: One row per checkpoint, with natural_errors.CHECKPOINT_COLUMNS.
    """
    table = ne.read_checkpoint_scores(Path(natural_dir) / NATURAL_SCORES_CSV, calib, errors)
    ne.check_coverage(table, cases)
    ne.consistency_check(table, fields)
    return table


def read_natural_stage_table(natural_dir) -> pd.DataFrame:
    """Reads #39's table_stage_by_k.csv exactly, its stage and k as text ("all" for the pooled rows)."""
    return pd.read_csv(Path(natural_dir) / NATURAL_STAGE_CSV, float_precision="round_trip",
                       dtype={"stage": str, "k": str})


# --- The tables ---

def recall_by_type_eps_k(fields: pd.DataFrame) -> pd.DataFrame:
    """
    Recall by score x k x damage type x strength, from the flags: flagged damaged
    fields / damaged fields, over all cases (recall) and over the cases whose clean
    field the score doesn't flag (recall_new), each with the exact Clopper-Pearson
    range of its count. Each case gives one field per type and strength, so the
    fields of a row are independent (grouped_eval.py).

    Inputs:
        fields (pd.DataFrame): Output of read_fields().

    Outputs:
        pd.DataFrame: One row per score x k x type x strength, with RECALL_BY_K_COLUMNS.
    """
    damaged = fields[fields["label"] != "clean"]
    perturbations = sorted(damaged["perturbation"].unique(), key=bm._perturbation_order)
    rows = []
    for score in SCORE_NAMES:
        for k in sorted(damaged["k"].unique()):
            for perturbation in perturbations:
                of_type = damaged[(damaged["k"] == k) & (damaged["perturbation"] == perturbation)]
                for epsilon in sorted(of_type["epsilon"].unique()):
                    group = of_type[of_type["epsilon"] == epsilon]
                    if group["case_id"].duplicated().any():
                        raise ValueError(f"A case has two damaged fields at {perturbation}, epsilon {epsilon}")
                    flagged = group[f"flagged_{score}"].to_numpy(dtype=bool)
                    new = ~group[f"preflagged_{score}"].to_numpy(dtype=bool)
                    row = {"score": score, "k": int(k), "perturbation": perturbation, "epsilon": float(epsilon),
                           "n_cases": int(group["case_id"].nunique()), "n_damaged": len(group),
                           "n_damaged_flagged": int(flagged.sum())}
                    bm._exact(row, "recall", row["n_damaged_flagged"], row["n_damaged"])
                    row.update({"n_preflagged_cases": int((~new).sum()), "n_damaged_new": int(new.sum()),
                                "n_damaged_new_flagged": int((flagged & new).sum())})
                    bm._exact(row, "recall_new", row["n_damaged_new_flagged"], row["n_damaged_new"])
                    rows.append(row)
    return pd.DataFrame(rows)[list(RECALL_BY_K_COLUMNS)]


def case_table(cases: list, errors: pd.DataFrame, loss: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    """
    The table of the new cases: one row per case, in case-file order, with its
    settings and the quantities that follow from them (viscosity nu = U0 / (Re k),
    velocity decay time tau_decay = 1 / (2 nu k^2), time window T = min(2, tau_decay);
    src/physics/taylor_green.py), its seeds, whether its run passed the training
    acceptance test, and the final model's errors: the relative L2 velocity error on the
    acceptance test's 64 x 64 x 20 evaluation grid (rel_l2_eval_grid, from
    loss_summary.csv), and on the independent 64 x 64 x 11 grid of the accuracy audit
    (#37; PROTOCOL.md section 6) over all times (relL2_uv, the main number) and at t = T
    (relL2_uv_T), and the pressure error there (relL2_p, mean-free at each time).

    Refuses inputs that disagree: every case must be in loss_summary.csv and have an
    lbfgs_final row in the true errors, each once; its pass/fail must be the same there
    and in the benchmark, and must follow from loss_summary.csv's acceptance metrics
    (rel_l2 < 0.10, mse_rc < 1e-4 and mse_rm < 1e-3).

    Inputs:
        cases (list[dict]): The cases, from accuracy_audit.read_cases().
        errors (pd.DataFrame): The true errors (results/accuracy/reference_errors_confirmation.csv).
        loss (pd.DataFrame): models/confirmation/loss_summary.csv.
        fields (pd.DataFrame): Output of read_fields(), for its passed_acceptance.

    Outputs:
        pd.DataFrame: One row per case, with CASE_COLUMNS.
    """
    case_ids = [case["case_id"] for case in cases]
    missing = [column for column in CASE_ERROR_COLUMNS if column not in errors.columns]
    if missing:
        raise ValueError(f"The true errors have no column(s) {missing}; write them with src/models/accuracy_audit.py")
    final = errors[errors["stage"] == FINAL_STAGE]
    for name, table in (("the true errors' lbfgs_final rows", final), ("loss_summary.csv", loss)):
        listed = table["case_id"].tolist()
        doubled = sorted({case_id for case_id in listed if listed.count(case_id) > 1})
        missing = sorted(set(case_ids) - set(listed))
        extra = sorted(set(listed) - set(case_ids))
        if doubled or missing or extra:
            raise ValueError(f"{name} must list every case of the case file once: twice {doubled}, missing "
                             f"{missing}, not in the case file {extra}")
    final, loss = final.set_index("case_id"), loss.set_index("case_id")
    benchmark_passed = fields.groupby("case_id")["passed_acceptance"].first()

    rows = []
    for case in cases:
        case_id = case["case_id"]
        summary = loss.loc[case_id]
        passed = bool(summary["passed"])
        follows = bool(summary["rel_l2"] < ACCEPT_REL_L2_BELOW and summary["mse_rc"] < ACCEPT_MSE_RC_BELOW
                       and summary["mse_rm"] < ACCEPT_MSE_RM_BELOW)
        if passed != follows:
            raise ValueError(f"{case_id}: loss_summary.csv says passed={passed}, but its acceptance metrics give "
                             f"{follows}")
        for name, other in (("the true errors", final.loc[case_id, "passed_acceptance"]),
                            ("the benchmark", benchmark_passed[case_id])):
            if bool(other) != passed:
                raise ValueError(f"{case_id}: passed_acceptance is {bool(other)} in {name}, {passed} in "
                                 "loss_summary.csv")
        if int(final.loc[case_id, "k"]) != case["k"]:
            raise ValueError(f"{case_id}: k differs between the case file and the true errors")
        nu = compute_nu(case["U0"], case["Re"], case["k"])
        rows.append({"case_id": case_id, "k": case["k"], "U0": case["U0"], "Re": case["Re"], "nu": nu,
                     "tau_decay": compute_decay_timescale(nu, case["k"]),
                     "T": compute_T(case["U0"], case["Re"], case["k"]),
                     "phi_x": case["phi_x"], "phi_y": case["phi_y"], "seed_points": case["seed_points"],
                     "seed_train": case["seed_train"], "seed_scoring": case["seed_scoring"],
                     "passed_acceptance": passed, "rel_l2_eval_grid": float(summary["rel_l2"]),
                     **{column: float(final.loc[case_id, column]) for column in CASE_ERROR_COLUMNS}})
    return pd.DataFrame(rows)[list(CASE_COLUMNS)]


def _synthetic_row(row, score: str, k, tau: float) -> dict:
    """
    One row of table_synthetic_vs_natural from a row of #35's metrics (damaged fields are
    the positives), its numbers unchanged. Refuses a row scored with another tau, or whose
    false-alarm range is not the exact range of its count (one clean field per case).
    """
    if row["tau"] != tau:
        raise ValueError(f"#35's metrics give {score} tau {row['tau']}, the frozen calibration {tau}")
    low, high = clopper_pearson(int(row["n_false_alarms"]), int(row["n_clean"]))
    if not np.allclose([low, high], [row["fpr_low"], row["fpr_high"]], rtol=1e-12, atol=0):
        raise ValueError(f"#35's false-alarm range of {score} (k {k}) is not the exact range of its count")
    n_positive, n_flagged = int(row["n_damaged"]), int(row["n_damaged_flagged"])
    return {"experiment": SYNTHETIC, "score": score, "k": str(k), "tau": tau, "n_cases": int(row["n_cases"]),
            "n_positive": n_positive, "n_negative": int(row["n_clean"]), "n_positive_flagged": n_flagged,
            "n_missed": n_positive - n_flagged, "n_false_alarms": int(row["n_false_alarms"]),
            **{column: row[column] for column in ("auc", "auc_low", "auc_high", "auc_n_boot", "recall", "recall_low",
                                                  "recall_high", "recall_n_boot", "fpr", "fpr_low", "fpr_high")},
            "fpr_n_boot": pd.NA, "fpr_range_method": "clopper_pearson"}


def _natural_row(group: pd.DataFrame, theirs, score: str, tau: float, k, n_boot: int, seed: int) -> dict:
    """
    One row of table_synthetic_vs_natural from a group of checkpoints (inaccurate ones are
    the positives). The counts, recall and false-alarm rate with their ranges are #39's
    (theirs: its table_stage_by_k row over all stages for this score and k), refused
    unless its tau, counts and rates follow from the flags of the checkpoints; the AUC,
    inaccurate against accurate, is new, with #39's case bootstrap (several checkpoints per case).
    """
    inaccurate, flagged = group["inaccurate"].to_numpy(dtype=bool), _flags(group, score)
    n_inaccurate, n_accurate = int(inaccurate.sum()), int((~inaccurate).sum())
    counts = {"n_cases": int(group["case_id"].nunique()), "n_checkpoints": len(group), "n_inaccurate": n_inaccurate,
              "n_missed": int((inaccurate & ~flagged).sum()), "n_false_alarms": int((~inaccurate & flagged).sum())}
    differ = {column: (int(theirs[column]), value) for column, value in counts.items() if int(theirs[column]) != value}
    rates = {"tau": tau,
             "recall": (n_inaccurate - counts["n_missed"]) / n_inaccurate if n_inaccurate else np.nan,
             "fpr": counts["n_false_alarms"] / n_accurate if n_accurate else np.nan}
    differ.update({column: (theirs[column], value) for column, value in rates.items()
                   if not np.isclose(float(theirs[column]), value, rtol=1e-12, atol=0, equal_nan=True)})
    if differ:
        raise ValueError(f"#39's table_stage_by_k ({score}, all stages, k {k}) doesn't follow from the checkpoints: "
                         f"{differ} (#39, here); rerun src/detection/natural_errors.py")
    view = pd.DataFrame({"case_id": group["case_id"].to_numpy(), "k": group["k"].to_numpy(), "damaged": inaccurate,
                         "value": group[score].to_numpy(dtype=float)})
    auc, auc_low, auc_high, auc_n_boot = case_bootstrap(view, bm.auc, n_boot=n_boot, seed=seed)
    return {"experiment": NATURAL, "score": score, "k": str(k), "tau": tau, "n_cases": counts["n_cases"],
            "n_positive": n_inaccurate, "n_negative": n_accurate, "n_positive_flagged": n_inaccurate - counts["n_missed"],
            "n_missed": counts["n_missed"], "n_false_alarms": counts["n_false_alarms"], "auc": auc, "auc_low": auc_low,
            "auc_high": auc_high, "auc_n_boot": auc_n_boot,
            **{column: theirs[column] for column in ("recall", "recall_low", "recall_high", "recall_n_boot", "fpr",
                                                     "fpr_low", "fpr_high", "fpr_n_boot")},
            "fpr_range_method": theirs["range_method"]}


def summary_table(metrics: dict, natural: pd.DataFrame, stage_table: pd.DataFrame, thresholds: dict,
                  n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    Both experiments in the same terms, for each score, over every k and for each k. The
    positives are the damaged fields (synthetic) or the inaccurate checkpoints (natural,
    relL2_uv above 0.10), the negatives the clean fields or the accurate checkpoints.
    AUC (positives against negatives) and recall have case-bootstrap ranges; the
    false-alarm rate has the exact range of its count for the clean fields (one per case)
    and a case-bootstrap range for the checkpoints (several per case). The synthetic rows
    are #35's numbers (metrics_overall, metrics_by_k) and the natural ones #39's
    (table_stage_by_k, over all stages), each checked against its flags; only the AUC of
    the checkpoints is computed here.

    Inputs:
        metrics (dict): #35's metrics tables, from read_metrics() ("overall" and "by_k").
        natural (pd.DataFrame): The checkpoints, from read_natural().
        stage_table (pd.DataFrame): #39's table_stage_by_k, from read_natural_stage_table().
        thresholds (dict): {score: tau} of the frozen calibration.
        n_boot, seed: Of the case bootstrap of the AUC (PROTOCOL.md section 7).

    Outputs:
        pd.DataFrame: One row per experiment x score x k, with SUMMARY_COLUMNS; k is text,
                      "all" for the rows over every k.
    """
    ks = sorted(int(k) for k in natural["k"].unique())
    pooled = stage_table[stage_table["stage"] == ALL]
    rows = []
    for score in SCORE_NAMES:
        for k in (ALL, *ks):
            table = metrics["overall"] if k == ALL else metrics["by_k"][metrics["by_k"]["k"] == k]
            selected = table[table["score"] == score]
            if len(selected) != 1:
                raise ValueError(f"#35's metrics have {len(selected)} rows for {score}, k {k}")
            rows.append(_synthetic_row(selected.iloc[0], score, k, thresholds[score]))
    for score in SCORE_NAMES:
        for k in (ALL, *ks):
            theirs = pooled[(pooled["score"] == score) & (pooled["k"] == str(k))]
            if len(theirs) != 1:
                raise ValueError(f"#39's table_stage_by_k has {len(theirs)} rows for {score}, all stages, k {k}")
            group = natural if k == ALL else natural[natural["k"] == k]
            rows.append(_natural_row(group, theirs.iloc[0], score, thresholds[score], k, n_boot, seed))
    out = pd.DataFrame(rows)[list(SUMMARY_COLUMNS)]
    return out.astype({"auc_n_boot": "Int64", "recall_n_boot": "Int64", "fpr_n_boot": "Int64"})


class _ExactFlow:
    """
    The exact Taylor-Green solution of a case, called like a model: an (N, 3) tensor of
    (x, y, t) in, an (N, 3) tensor of (u, v, p) out. It stands in for the model when a
    damage is sized, since the time-shift damage asks its model for the field at a
    shifted time.
    """

    def __init__(self, case: dict, nu: float):
        self.case, self.nu = case, nu

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        case = self.case
        u, v, p = generate_tgv(coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], case["U0"], case["k"],
                               case["phi_x"], case["phi_y"], self.nu)
        return torch.cat([u, v, p], dim=1)


def damage_sizes(cases: list, epsilons) -> pd.DataFrame:
    """
    The size of each synthetic damage as a velocity error, one row per case x damage type x
    strength: the relative L2 change it makes to the velocity, against the exact velocity,
    sqrt(sum (u~ - u)^2 + (v~ - v)^2) / sqrt(sum u^2 + v^2), over the accuracy audit's
    64 x 64 x 11 test grid (accuracy_audit.accuracy_grid; PROTOCOL.md section 6), the grid
    of the natural errors' relL2_uv. The damage is the benchmark's own code
    (perturbations.apply_perturbation, with U0, k, T and the decay time as
    phs.compute_phs_components passes them), applied to the exact solution in place of
    the model: an added damage changes any field by the same amount, and the time
    shift's size is its change to the exact flow.

    Inputs:
        cases (list[dict]): The cases (case_id, k, U0, Re, phi_x, phi_y), from read_cases().
        epsilons (list[float]): The damage strengths.

    Outputs:
        pd.DataFrame: case_id, k, perturbation, epsilon and velocity_change (a fraction).
    """
    rows = []
    for case in cases:
        U0, Re, k = case["U0"], case["Re"], case["k"]
        nu, T = compute_nu(U0, Re, k), compute_T(U0, Re, k)
        params = {"U0": U0, "k": k, "T": T, "tau_decay": compute_decay_timescale(nu, k)}
        t, x, y = (axis.reshape(-1, 1) for axis in accuracy_grid(T))
        flow = _ExactFlow(case, nu)
        exact = flow(torch.cat([x, y, t], dim=1))
        clean = {"u": exact[:, 0:1], "v": exact[:, 1:2], "p": exact[:, 2:3]}
        coords = {"x": x, "y": y, "t": t}
        for perturbation in PERTURBATION_NAMES:
            for epsilon in epsilons:
                damaged = apply_perturbation(perturbation, clean, coords, params, float(epsilon),
                                             model=flow if perturbation == "temporal_mismatch" else None,
                                             no_grad=True)
                change = relative_l2((damaged["u"], damaged["v"]), (clean["u"], clean["v"]))
                rows.append({"case_id": case["case_id"], "k": k, "perturbation": perturbation,
                             "epsilon": float(epsilon), "velocity_change": change})
    return pd.DataFrame(rows)


def damage_size_table(sizes: pd.DataFrame) -> pd.DataFrame:
    """
    The size of each damage type and strength over the cases: the smallest, median and
    largest velocity change. Refuses a type that changes the velocity of some fields and
    not of others, since the figures leave out only the types that never change it.

    Inputs:
        sizes (pd.DataFrame): Output of damage_sizes().

    Outputs:
        pd.DataFrame: One row per damage type x strength, with DAMAGE_SIZE_COLUMNS.
    """
    rows = []
    for perturbation in sorted(sizes["perturbation"].unique(), key=bm._perturbation_order):
        of_type = sizes[sizes["perturbation"] == perturbation]
        changed = (of_type["velocity_change"] > 0).to_numpy()
        if changed.any() and not changed.all():
            raise ValueError(f"The {perturbation} damage changes the velocity of some fields but not of others")
        for epsilon in sorted(of_type["epsilon"].unique()):
            group = of_type[of_type["epsilon"] == epsilon]
            values = group["velocity_change"]
            rows.append({"perturbation": perturbation, "epsilon": float(epsilon),
                         "n_cases": int(group["case_id"].nunique()), "velocity_change_min": float(values.min()),
                         "velocity_change_median": float(values.median()),
                         "velocity_change_max": float(values.max())})
    return pd.DataFrame(rows)[list(DAMAGE_SIZE_COLUMNS)]


def field_case(cases: list) -> dict:
    """The case fig_damage_fields shows, by a fixed rule: of the k = 1 cases, the one with the median U0 (lower middle)."""
    k1 = sorted((case for case in cases if case["k"] == 1), key=lambda case: case["U0"])
    return k1[(len(k1) - 1) // 2]


def damage_field_maps(case: dict, epsilon: float, n: int = FIELD_GRID, t_fraction: float = 0.5) -> dict:
    """
    What each damage does to the exact solution of a case, on an n x n grid of [0, 2 pi)
    at t = t_fraction * T, with the benchmark's own damage code as damage_sizes() uses it:
      change      the velocity change |(du, dv)| / U0, or, for a damage that leaves the
                  velocity alone, the pressure change |dp| / U0^2;
      momentum    the momentum residual sqrt(R_u^2 + R_v^2) / (U0^2 k);
      continuity  the continuity residual |R_c| / (U0 k);
    the residuals by autograd (navier_stokes.compute_residuals) and scaled as the scores
    scale them (ResidualScaler). The exact solution has no residual (refused otherwise),
    so each residual map is the damage's own.

    Outputs:
        dict: axis (the n grid values), t, and maps ({perturbation: {"change", "change_of",
              "momentum", "continuity"}}, each map an (n, n) array indexed [x, y]).
    """
    U0, Re, k = case["U0"], case["Re"], case["k"]
    nu, T = compute_nu(U0, Re, k), compute_T(U0, Re, k)
    params = {"U0": U0, "k": k, "T": T, "tau_decay": compute_decay_timescale(nu, k)}
    scaler, flow = ResidualScaler(U0, k), _ExactFlow(case, nu)
    axis = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n   # 2 pi left out: on the periodic domain it is 0
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    t_value = t_fraction * T

    def fields_on_grid():
        x = X.reshape(-1, 1).clone().requires_grad_(True)
        y = Y.reshape(-1, 1).clone().requires_grad_(True)
        t = torch.full_like(x, t_value).requires_grad_(True)
        out = flow(torch.cat([x, y, t], dim=1))
        return x, y, t, {"u": out[:, 0:1], "v": out[:, 1:2], "p": out[:, 2:3]}

    def scaled_residuals(field, x, y, t):
        return scaler.scale_residuals(*compute_residuals(field["u"], field["v"], field["p"], x, y, t, nu))

    x, y, t, exact = fields_on_grid()
    largest = max(float(r.detach().abs().max()) for r in scaled_residuals(exact, x, y, t))
    if largest > 1e-8:
        raise ValueError(f"The exact solution of {case['case_id']} has a scaled residual of {largest:.3g}")
    maps = {}
    for perturbation in PERTURBATION_NAMES:
        x, y, t, clean = fields_on_grid()
        damaged = apply_perturbation(perturbation, clean, {"x": x, "y": y, "t": t}, params, float(epsilon),
                                     model=flow if perturbation == "temporal_mismatch" else None, no_grad=False)
        R_u, R_v, R_c = scaled_residuals(damaged, x, y, t)
        velocity = torch.sqrt((damaged["u"] - clean["u"]) ** 2 + (damaged["v"] - clean["v"]) ** 2) / U0
        pressure = (damaged["p"] - clean["p"]).abs() / U0 ** 2
        change, change_of = (velocity, "velocity") if float(velocity.detach().max()) > 0 else (pressure, "pressure")
        maps[perturbation] = {name: value.detach().reshape(n, n).numpy() for name, value in (
            ("change", change), ("momentum", torch.sqrt(R_u ** 2 + R_v ** 2)), ("continuity", R_c.abs()))}
        maps[perturbation]["change_of"] = change_of
    return {"axis": axis.numpy(), "t": t_value, "maps": maps}


def term_auc_table(fields: pd.DataFrame, natural: pd.DataFrame, n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    The AUC of each score term on its own (its normalized value, BAR_NAMES), with its
    case-bootstrap 95% confidence interval (PROTOCOL.md section 7): for the synthetic
    damage, the damaged fields against the clean ones, over every damage type and for
    each type (that type's damaged fields against all clean fields, as #35's
    metrics_by_type); for the natural errors, the inaccurate checkpoints against the
    accurate ones. "unchanged" marks a damage type that leaves the term exactly as it was
    on every field (each damaged value equals its case's clean value), whose AUC is then
    0.5 by construction.

    Inputs:
        fields (pd.DataFrame): The benchmark, from read_fields().
        natural (pd.DataFrame): The checkpoints, from read_natural().
        n_boot, seed: Of the case bootstrap.

    Outputs:
        pd.DataFrame: One row per term x group, with TERM_AUC_COLUMNS.
    """
    clean = fields[fields["label"] == "clean"]
    damaged = fields[fields["label"] != "clean"]
    clean_value = clean.set_index("case_id")
    groups = [(SYNTHETIC, ALL, fields)]
    for perturbation in sorted(damaged["perturbation"].unique(), key=bm._perturbation_order):
        groups.append((SYNTHETIC, perturbation, pd.concat([clean, damaged[damaged["perturbation"] == perturbation]])))
    groups.append((NATURAL, ALL, natural))
    rows = []
    for term in BAR_NAMES:
        for experiment, group_name, group in groups:
            positive = (group["inaccurate"].to_numpy(dtype=bool) if experiment == NATURAL
                        else (group["label"] != "clean").to_numpy())
            view = pd.DataFrame({"case_id": group["case_id"].to_numpy(), "k": group["k"].to_numpy(),
                                 "damaged": positive, "value": group[term].to_numpy(dtype=float)})
            auc, low, high, used = case_bootstrap(view, bm.auc, n_boot=n_boot, seed=seed)
            unchanged = False
            if experiment == SYNTHETIC and group_name != ALL:
                of_type = group[group["label"] != "clean"]
                unchanged = bool((of_type[term].to_numpy() == of_type["case_id"].map(clean_value[term]).to_numpy()).all())
            rows.append({"term": term, "experiment": experiment, "group": group_name,
                         "n_cases": int(group["case_id"].nunique()), "n_positive": int(positive.sum()),
                         "n_negative": int((~positive).sum()), "auc": auc, "auc_low": low, "auc_high": high,
                         "auc_n_boot": used, "unchanged": unchanged})
    return pd.DataFrame(rows)[list(TERM_AUC_COLUMNS)].astype({"auc_n_boot": "Int64"})


# --- LaTeX of the tables ---

def _percent(value) -> str:
    """0.001534 -> '0.15'."""
    return f"{100 * value:.2f}"


def case_table_latex(table: pd.DataFrame, caption: str) -> str:
    """
    The case table as a full-width LaTeX table (booktabs): one row per case, numbers
    rounded, errors in %, the seeds and the errors under grouped headers. The time
    window T gets a column only when it differs between cases (otherwise the caption
    gives it).
    """
    with_t = table["T"].nunique() > 1
    flow = ["$k$", "$U_0$", "Re", "$\\nu$", "$\\tau_d$", *(["$T$"] if with_t else []), "$\\phi_x$", "$\\phi_y$"]
    first_seed = 2 + len(flow)
    lines = [
        "% Written by src/detection/final_figures.py; needs \\usepackage{booktabs}.",
        "\\begin{table*}[tb]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:cases}",
        "\\begin{tabular}{l" + "r" * (len(flow) + 3) + "c" + "r" * 4 + "}",
        "\\toprule",
        " & " * (len(flow) + 1) + "\\multicolumn{3}{c}{Seeds} & & \\multicolumn{3}{c}{Velocity error (\\%)} & "
        "Pressure \\\\",
        f"\\cmidrule(lr){{{first_seed}-{first_seed + 2}}} \\cmidrule(lr){{{first_seed + 4}-{first_seed + 6}}}",
        "Case & " + " & ".join(flow) + " & points & training & scoring & Passed & eval.\\ grid & test grid & at $T$ & "
        "error (\\%) \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        cells = [row.case_id.split("_")[1], str(row.k), f"{row.U0:.3f}", f"{row.Re:.1f}",
                 f"{row.nu:#.3g}".rstrip("."), f"{row.tau_decay:#.3g}".rstrip("."),   # 3 digits, zeros kept: 0.0330
                 *([f"{row.T:g}"] if with_t else []), f"{row.phi_x:.3f}", f"{row.phi_y:.3f}",
                 str(row.seed_points), str(row.seed_train), str(row.seed_scoring),
                 "yes" if row.passed_acceptance else "no", _percent(row.rel_l2_eval_grid), _percent(row.relL2_uv),
                 _percent(row.relL2_uv_T), _percent(row.relL2_p)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def recall_by_k_latex(table: pd.DataFrame, caption: str) -> str:
    """
    The type x strength x k recall as a LaTeX longtable: one row per score x k x type, one
    column per strength. Its 75 rows run over a page, so it is a longtable, which LaTeX
    refuses in two-column mode; the header says so (a paper puts it after \\onecolumn, in
    an appendix).
    """
    epsilons = sorted(table["epsilon"].unique())
    rows = []
    for (score, k, perturbation), group in table.groupby(["score", "k", "perturbation"], sort=False):
        group = group.set_index("epsilon")
        cells = {"Score": bm._tex_escape(score.split("_")[0]), "$k$": str(k), "Damage": DAMAGE_LABELS[perturbation]}
        for epsilon in epsilons:
            cells[f"$\\varepsilon={epsilon:g}$"] = (f"{int(group.loc[epsilon, 'n_damaged_flagged'])}/"
                                                   f"{int(group.loc[epsilon, 'n_damaged'])}")
        rows.append(cells)
    header = ("% Written by src/detection/final_figures.py; needs \\usepackage{booktabs} and \\usepackage{longtable}.\n"
              "% A longtable works only in one-column mode: in a two-column paper, put \\onecolumn before it "
              "(e.g. in an appendix).\n")
    return header + pd.DataFrame(rows).to_latex(index=False, escape=False, longtable=True, caption=caption,
                                                label="tab:recall_by_type_eps_k")


TEX_SCORE_LABELS = {score: label.replace("−", "$-$") for score, label in SCORE_LABELS.items()}


def summary_latex(table: pd.DataFrame, caption: str) -> str:
    """
    table_synthetic_vs_natural as a full-width LaTeX table (booktabs): its rows over every
    k, one per score, the two experiments side by side.
    """
    pooled = table[table["k"] == ALL]
    lines = [
        "% Written by src/detection/final_figures.py; needs \\usepackage{booktabs}.",
        "\\begin{table*}[tb]",
        "\\centering",
        # Six columns with ranges: at \small, or with the default column gaps, wider than a 7.16 in page
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:synthetic_vs_natural}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Synthetic damage} & \\multicolumn{3}{c}{Natural errors} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "Score & AUC & Recall & False alarms & AUC & Recall & False alarms \\\\",
        "\\midrule",
    ]
    for score in SCORE_NAMES:
        cells = [TEX_SCORE_LABELS[score]]
        for experiment in (SYNTHETIC, NATURAL):
            selected = pooled[(pooled["experiment"] == experiment) & (pooled["score"] == score)]
            if len(selected) != 1:
                raise ValueError(f"table_synthetic_vs_natural has {len(selected)} rows for {experiment} {score}")
            row = selected.iloc[0]
            cells += [bm._with_range(row, "auc"), bm._with_range(row, "recall"),
                      f"{int(row['n_false_alarms'])}/{int(row['n_negative'])}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def _two_digits(value: float) -> str:
    """A number with 2 significant digits, trailing zeros kept: 0.0019, 0.15, 1.0, 15."""
    return f"{value:#.2g}".rstrip(".")   # '#' keeps trailing zeros (1.0, not 1); '15.' -> '15'


def _size(value: float) -> str:
    """A velocity change in %, with 2 significant digits: 0.0019, 0.15, 1.0, 5.4; '0' for none."""
    return "0" if value == 0 else _two_digits(100 * value)


def damage_size_latex(table: pd.DataFrame, caption: str) -> str:
    """
    table_damage_size as a full-width LaTeX table (booktabs): one row per strength, one
    column per damage type, each cell the median velocity change and its range over the cases.
    """
    perturbations = list(dict.fromkeys(table["perturbation"]))
    epsilons = sorted(table["epsilon"].unique())
    # Two-word names on two lines: \shortstack{velocity\\divergence}
    headers = ["\\shortstack{" + bm._tex_escape(DAMAGE_LABELS[p]).replace(" ", "\\\\") + "}" for p in perturbations]
    lines = [
        "% Written by src/detection/final_figures.py; needs \\usepackage{booktabs}.",
        "\\begin{table*}[tb]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:damage_size}",
        "\\begin{tabular}{r" + "c" * len(perturbations) + "}",
        "\\toprule",
        f" & \\multicolumn{{{len(perturbations)}}}{{c}}{{Velocity change (\\%): median (smallest--largest)}} \\\\",
        f"\\cmidrule(lr){{2-{len(perturbations) + 1}}}",
        "$\\varepsilon$ & " + " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    indexed = table.set_index(["perturbation", "epsilon"])
    for epsilon in epsilons:
        cells = [f"{epsilon:g}"]
        for perturbation in perturbations:
            row = indexed.loc[(perturbation, epsilon)]
            if row["velocity_change_max"] == 0:
                cells.append("0")
            else:
                cells.append(f"{_size(row['velocity_change_median'])} ({_size(row['velocity_change_min'])}--"
                             f"{_size(row['velocity_change_max'])})")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def term_auc_latex(table: pd.DataFrame, caption: str) -> str:
    """
    table_term_auc as a full-width LaTeX table (booktabs): one row per score term, one
    column per group, each cell the AUC over its 95% confidence interval; a dagger marks
    a damage that leaves the term unchanged.
    """
    synthetic = [g for g in dict.fromkeys(table.loc[table["experiment"] == SYNTHETIC, "group"]) if g != ALL]
    columns = [(SYNTHETIC, ALL, "\\shortstack{all\\\\types}")]
    columns += [(SYNTHETIC, g, "\\shortstack{" + DAMAGE_LABELS[g].replace(" ", "\\\\") + "}") for g in synthetic]
    columns.append((NATURAL, ALL, "\\shortstack{inaccurate vs\\\\accurate}"))
    lines = [
        "% Written by src/detection/final_figures.py; needs \\usepackage{booktabs}.",
        "\\begin{table*}[tb]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:term_auc}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        f" & \\multicolumn{{{len(columns) - 1}}}{{c}}{{Synthetic damage}} & Natural errors \\\\",
        f"\\cmidrule(lr){{2-{len(columns)}}} \\cmidrule(lr){{{len(columns) + 1}-{len(columns) + 1}}}",
        "Score term & " + " & ".join(header for _, _, header in columns) + " \\\\",
        "\\midrule",
    ]
    indexed = table.set_index(["term", "experiment", "group"])
    for term in BAR_NAMES:
        cells = [TERM_LABELS[term]]
        for experiment, group, _ in columns:
            row = indexed.loc[(term, experiment, group)]
            mark = "$^\\dagger$" if row["unchanged"] else ""
            # \\{} so LaTeX doesn't read the interval's "[" as the optional spacing of the line break
            cells.append(f"\\shortstack{{{bm._number(row['auc'])}{mark}\\\\{{}}[{bm._number(row['auc_low'])}, "
                         f"{bm._number(row['auc_high'])}]}}")
        lines.append(" & ".join(cells) + " \\\\[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def table_latex(stem: str, table: pd.DataFrame, caption: str) -> str:
    """The .tex of one of TABLES."""
    writers = {"table_recall_by_type_eps_k": recall_by_k_latex, "table_cases": case_table_latex,
               "table_synthetic_vs_natural": summary_latex, "table_damage_size": damage_size_latex,
               "table_term_auc": term_auc_latex}
    return writers[stem](table, caption)


# --- Facts for the captions ---

def _case_name(case_id: str) -> str:
    """'case_46' -> 'case 46', as a paper names it."""
    return case_id.replace("_", " ")


def _tex_cases(case_ids) -> str:
    """'case 34 and case 45'."""
    names = [_case_name(case_id) for case_id in case_ids]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _short(score: str) -> str:
    return score.split("_")[0]


def _range_text(row, name: str) -> str:
    """'0.927 [0.928, 0.941]'."""
    return bm._with_range(row, name)


def full_recall_epsilon(table: pd.DataFrame, rate: str) -> dict:
    """
    {(score, perturbation): the smallest strength from which every damaged field is
    flagged, at it and every larger strength (rate = 1); NaN if never}, from a table
    with score, perturbation, epsilon and the rate.
    """
    out = {}
    for (score, perturbation), group in table.groupby(["score", "perturbation"], sort=False):
        group = group.sort_values("epsilon")
        full = (group[rate] == 1.0).to_numpy()
        tail = np.flatnonzero(~full)
        start = 0 if len(tail) == 0 else tail[-1] + 1
        out[(score, perturbation)] = group["epsilon"].iloc[start] if start < len(group) else np.nan
    return out


def recall_sentence(table: pd.DataFrame, rate: str, count: str, total: str) -> str:
    """
    'Every score flags every damaged field from eps = 0.01 on (Score2 ... already from
    0.002), except Score1 for boundary damage: 15/30 at 0.01 and 25/30 at 0.02.'
    """
    full = full_recall_epsilon(table, rate)
    values = pd.Series(full).dropna()
    common = values.mode().min()
    earlier = [key for key, value in full.items() if value < common]
    later = [key for key, value in full.items() if not value <= common]
    sentence = f"Every score flags every damaged field from $\\varepsilon = {common:g}$ on"
    if earlier:
        parts = []
        for epsilon in sorted({full[key] for key in earlier}):
            who = []
            for score in SCORE_NAMES:
                types = [DAMAGE_LABELS[p] for s, p in earlier if s == score and full[(s, p)] == epsilon]
                if types:
                    who.append(f"{_short(score)} with {_and(types)} damage")
            parts.append(f"already from {epsilon:g} for {_and(who)}")
        sentence += " (" + "; ".join(parts) + ")"
    if later:
        exceptions = []
        for score, perturbation in later:
            rows = table[(table["score"] == score) & (table["perturbation"] == perturbation)
                         & (table["epsilon"] >= common) & (table[rate] < 1.0)].sort_values("epsilon")
            counts = " and ".join(f"{int(r[count])}/{int(r[total])} at {r['epsilon']:g}" for _, r in rows.iterrows())
            exceptions.append(f"{_short(score)} for {DAMAGE_LABELS[perturbation]} damage ({counts})")
        sentence += ", except " + "; ".join(exceptions)
    return sentence + "."


def smallest_eps_facts(fields: pd.DataFrame, thresholds: dict) -> dict:
    """
    At the smallest strength: for each score, how many damaged fields it flags and how
    many of them belong to cases whose clean field it already flags, and the new
    detections, each with its case and clean score / tau.
    """
    damaged = fields[fields["label"] != "clean"]
    epsilon = damaged["epsilon"].min()
    at = damaged[damaged["epsilon"] == epsilon]
    clean = fields[fields["label"] == "clean"].set_index("case_id")
    facts = {"epsilon": epsilon, "n_cases": fields["case_id"].nunique()}
    for score in SCORE_NAMES:
        flagged = at[at[f"flagged_{score}"]]
        new = flagged[~flagged[f"preflagged_{score}"]]
        facts[score] = {
            "n_flagged": len(flagged), "n_preflagged_fields": int(flagged[f"preflagged_{score}"].sum()),
            "n_preflagged_cases": int(clean[f"flagged_{score}"].sum()),
            "new": [(row["perturbation"], row["case_id"], clean.loc[row["case_id"], score] / thresholds[score])
                    for _, row in new.sort_values(["perturbation", "case_id"]).iterrows()],
        }
    return facts


def auc_above_range(metrics: dict) -> list:
    """Every (table, row) of #35's metrics whose AUC range lies entirely above its estimate."""
    found = []
    for name in ("by_k", "by_type"):
        table = metrics[name]
        for _, row in table[table["auc_low"] > table["auc"]].iterrows():
            found.append((name, row))
    return found


def within_case_concordance(fields: pd.DataFrame, score: str) -> tuple:
    """
    (within, across): the share of (damaged, clean) pairs in which the damaged field
    scores above the clean one (ties count one half), over the pairs of a case's
    damaged fields with its own clean field, and over the pairs with the clean
    fields of the other cases.
    """
    clean = fields[fields["label"] == "clean"].set_index("case_id")[score]
    damaged = fields[fields["label"] != "clean"]
    values = damaged[score].to_numpy()
    own = damaged["case_id"].map(clean).to_numpy()
    within = float(np.mean((values > own) + 0.5 * (values == own)))
    pairs = 0.0
    n_pairs = 0
    for case_id, clean_value in clean.items():
        others = damaged["case_id"].to_numpy() != case_id
        pairs += float(np.sum((values[others] > clean_value) + 0.5 * (values[others] == clean_value)))
        n_pairs += int(others.sum())
    return within, pairs / n_pairs


def zero_width_recalls(metrics_by_k: pd.DataFrame, fields: pd.DataFrame) -> list:
    """
    The by-k recall rows of #35 whose bootstrap range has no width, each with the one
    count every case of the group shares (checked: a zero-width range means no
    resample of cases can change the rate).
    """
    found = []
    rows = metrics_by_k[metrics_by_k["recall_low"] == metrics_by_k["recall_high"]]
    damaged = fields[fields["label"] != "clean"]
    for _, row in rows.iterrows():
        group = damaged[damaged["k"] == row["k"]]
        counts = group.groupby("case_id")[f"flagged_{row['score']}"].sum()
        per_case = group.groupby("case_id").size()
        if counts.nunique() != 1 or per_case.nunique() != 1:
            raise ValueError(f"metrics_by_k: {row['score']} k={row['k']} has a zero-width recall range, but its "
                             "cases don't all have the same count")
        found.append((row, int(counts.iloc[0]), int(per_case.iloc[0])))
    return found


# --- Captions ---

def _and(items: list) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _signed(value: float) -> str:
    """A correlation in LaTeX math, with a real minus sign: '$-0.80$'."""
    return f"${value:.2f}$"


def caption_recall_vs_eps(type_eps: pd.DataFrame, fields: pd.DataFrame, thresholds: dict) -> str:
    facts = smallest_eps_facts(fields, thresholds)
    epsilons = ", ".join(f"{value:g}" for value in sorted(type_eps["epsilon"].unique()))
    n_cases = facts["n_cases"]
    with_preflagged = [s for s in SCORE_NAMES if facts[s]["n_preflagged_cases"]]
    shares = _and([f"{facts[s]['n_preflagged_fields']} of {facts[s]['n_flagged']} for {_short(s)}"
                   for s in with_preflagged])
    others = [s for s in SCORE_NAMES if not facts[s]["n_preflagged_cases"]]
    other_text = "".join(f"; {_short(s)}, which flags no clean field, flags {facts[s]['n_flagged']} "
                         f"field{'s' if facts[s]['n_flagged'] != 1 else ''} there" for s in others)
    return (f"Recall of the damaged fields of the {n_cases} new cases against the damage strength $\\varepsilon$ "
            f"({epsilons}; log scale), one panel per damage type and one line per score. A field is flagged when "
            f"its score is above the frozen $\\tau$. Each point is out of {n_cases} fields, one per case; bands: "
            "exact (Clopper--Pearson) 95\\% confidence intervals, drawn once in grey where several scores have the "
            "same count and so the same interval. "
            + recall_sentence(type_eps, "recall", "n_damaged_flagged", "n_damaged")
            + f" At $\\varepsilon = {facts['epsilon']:g}$ nearly all flagged fields belong to cases whose clean "
              f"field the score already flags ({shares}{other_text}); "
              "Fig.~\\ref{fig:recall_vs_eps_new_detections} leaves those cases out. " + SCORE_KEY)


def caption_recall_new(type_eps: pd.DataFrame, fields: pd.DataFrame, thresholds: dict) -> str:
    facts = smallest_eps_facts(fields, thresholds)
    n_cases = facts["n_cases"]
    counts = ", ".join(f"{_short(s)} {n_cases - facts[s]['n_preflagged_cases']}" for s in SCORE_NAMES)
    new = []
    for score in SCORE_NAMES:
        by_type = {}
        for perturbation, case_id, ratio in facts[score]["new"]:
            by_type.setdefault(perturbation, []).append((case_id, ratio))
        for perturbation, found in by_type.items():
            cases = _tex_cases([case_id for case_id, _ in found])
            ratios = _and([f"{ratio:.2f}" for _, ratio in found])
            new.append(f"{_short(score)}, {DAMAGE_LABELS[perturbation]} damage: {cases} (clean score {ratios} "
                       "times $\\tau$)")
    if new:
        smallest = (f" At $\\varepsilon = {facts['epsilon']:g}$ the only new detections are cases whose clean score "
                    f"is just below $\\tau$: {'; '.join(new)}.")
    else:
        smallest = f" At $\\varepsilon = {facts['epsilon']:g}$ no score detects anything new."
    return ("Recall as in Fig.~\\ref{fig:recall_vs_eps}, but only over the cases whose clean field the score does "
            "not flag, since flagging a damaged field of an already-flagged case is no new detection. Cases per "
            f"score: {counts}; bands: exact 95\\% confidence intervals." + smallest + " "
            + recall_sentence(type_eps, "recall_new", "n_damaged_new_flagged", "n_damaged_new"))


def caption_recall_by_k(by_k: pd.DataFrame, metrics: dict, fields: pd.DataFrame) -> str:
    second = sorted(by_k["epsilon"].unique())[1]
    at = by_k[by_k["epsilon"] == second]
    pooled = at.groupby(["score", "k"])[["n_damaged_flagged", "n_damaged"]].sum()
    parts = []
    for score in SCORE_NAMES:
        counts = [f"{int(pooled.loc[(score, k), 'n_damaged_flagged'])}" for k in sorted(by_k["k"].unique())]
        parts.append(f"{_short(score)} {', '.join(counts)}")
    n = int(pooled["n_damaged"].iloc[0])
    zero = zero_width_recalls(metrics["by_k"], fields)
    zero_text = ""
    if zero:
        row, count, per_case = zero[0]
        others = "; ".join(f"{_short(r['score'])} for $k = {r['k']}$, {_range_text(r, 'recall')}, every case "
                           f"{c} of {n_fields}" for r, c, n_fields in zero[1:])
        zero_text = (" Pooled over the six strengths instead (Table~\\ref{tab:benchmark_metrics_by_k}), some "
                     f"case-bootstrap confidence intervals have no width: {_short(row['score'])}'s recall for "
                     f"$k = {row['k']}$ is "
                     f"{_range_text(row, 'recall')}, because every $k = {row['k']}$ case has {count} of its "
                     f"{per_case} damaged fields flagged, so no resample of cases can change the rate"
                     + (f" (also {others})" if others else "") + ". The exact intervals per strength shown here don't "
                     "have this problem.")
    ks = sorted(by_k["k"].unique())
    k_text = ", ".join(str(k) for k in ks)
    n_types = by_k["perturbation"].nunique()
    falls = all((np.diff([pooled.loc[(score, k), "n_damaged_flagged"] for k in ks]) < 0).all() for score in SCORE_NAMES)
    trend = "Recall falls with $k$ for every score" if falls else "Recall depends on $k$"
    return (f"Recall as in Fig.~\\ref{{fig:recall_vs_eps}}, split by $k$ (rows): each point is out of the "
            f"{int(by_k['n_damaged'].iloc[0])} cases with that $k$, with exact 95\\% confidence intervals "
            f"(Table~\\ref{{tab:recall_by_type_eps_k}}). {trend}: at "
            f"$\\varepsilon = {second:g}$, the damaged fields flagged out of {n} ({n_types} types $\\times$ "
            f"{n // n_types} cases) for $k = {k_text}$ are {'; '.join(parts)}." + zero_text)


def caption_roc(overall: pd.DataFrame, metrics: dict, fields: pd.DataFrame) -> str:
    n_clean = int((fields["label"] == "clean").sum())
    n_damaged = len(fields) - n_clean
    lean = overall.assign(gap=(overall["auc_high"] - overall["auc"]) - (overall["auc"] - overall["auc_low"]))
    leaning = lean.loc[lean["gap"].idxmax()]
    above = auc_above_range(metrics)
    is_clean = (fields["label"] == "clean").to_numpy()
    points = {}  # {score: (false alarms, damaged fields flagged)} at tau
    for score in SCORE_NAMES:
        flags = fields[f"flagged_{score}"].to_numpy(dtype=bool)
        points[score] = (int((flags & is_clean).sum()), int((flags & ~is_clean).sum()))
    same =[(a, b) for i, a in enumerate(SCORE_NAMES) for b in SCORE_NAMES[i + 1:]
            if points[a][0] == points[b][0] and abs(points[a][1] - points[b][1]) <= 0.01 * n_damaged]
    text = (f"ROC curves of the five scores on the {len(fields)} fields of the {fields['case_id'].nunique()} new "
            f"cases ({n_damaged} damaged, {n_clean} clean); the inset enlarges the top-left corner. Legend: AUC "
            "[95\\% confidence interval: case bootstrap, 2000 resamples of whole cases within each $k$]. Markers: the "
            "frozen "
            f"$\\tau$ of each score, at its false-alarm rate on the {n_clean} clean fields and its recall")
    if same:
        a, b = same[0]
        text += (f" ({_short(a)} and {_short(b)} nearly coincide: {points[a][0]} false alarms each, "
                 f"{points[a][1]} and {points[b][1]} damaged fields flagged)")
    text += (f"; the false-alarm axis moves in steps of 1/{n_clean}. The intervals lean above their estimates "
             f"({_short(leaning['score'])}: {_range_text(leaning, 'auc')})")
    if above:
        name, row = above[0]
        subset = f"$k = {row['k']}$" if name == "by_k" else f"{DAMAGE_LABELS[row['perturbation']]} damage"
        label = "tab:benchmark_metrics_by_k" if name == "by_k" else "tab:benchmark_metrics_by_type"
        within, across = within_case_concordance(
            fields[fields["k"] == row["k"]] if name == "by_k" else
            fields[fields["perturbation"].isin([CLEAN, row["perturbation"]])], row["score"])
        text += (f", and for {len(above)} subsets in Tables~\\ref{{tab:benchmark_metrics_by_k}} and "
                 f"\\ref{{tab:benchmark_metrics_by_type}} they lie entirely above them, e.g. {_short(row['score'])} "
                 f"for {subset} (Table~\\ref{{{label}}}): {_range_text(row, 'auc')}. A resample that draws a case "
                 "twice adds the pairs of that case's clean field with its own damaged fields, which are nearly "
                 f"always in the right order (there, {100 * within:.1f}\\% of them, against {100 * across:.1f}\\% of "
                 "the pairs across cases), so the resampled AUCs run high.")
    else:
        text += "."
    return text + " " + SCORE_KEY


def relative_energy_error(clean: pd.DataFrame) -> pd.Series:
    """
    sqrt(S_E) / E(0) of each clean field: the root-mean-square gap between the model's
    kinetic energy and its exact decay, over the exact energy at t = 0, U0^2 / 4 (the
    Taylor-Green vortex's mean of (u^2 + v^2) / 2). clean needs E and U0.
    """
    return np.sqrt(clean["E"]) / (clean["U0"] ** 2 / 4)


def caption_clean_scores(fields: pd.DataFrame, old: pd.DataFrame, thresholds: dict, cases: list) -> str:
    case_by_id = {case["case_id"]: case for case in cases}
    clean = fields[fields["label"] == "clean"].set_index("case_id")
    clean = clean.assign(U0=[case_by_id[case_id]["U0"] for case_id in clean.index])
    n_new, n_old = len(clean), len(old)
    n_old_above = {score: int((old[score] > thresholds[score]).sum()) for score in SCORE_NAMES}
    old_above = (f"{n_old_above[SCORE_NAMES[0]]} of them lie" if len(set(n_old_above.values())) == 1
                 else "a few of them lie")
    alarms = []
    for score in SCORE_NAMES:
        flagged = clean[clean[f"flagged_{score}"]]
        listed = "none" if flagged.empty else _and([f"{_case_name(case_id)} ($k = {row['k']}$)"
                                                   for case_id, row in flagged.iterrows()])
        alarms.append(f"{_short(score)} {listed}")
    text = (f"The clean field of each of the {n_new} new cases (filled), and of the {n_old} old cases the calibration "
            "was fitted on (open), for each score and $k$; line: the frozen $\\tau$, the 95th percentile of the "
            f"open points (so {old_above} above it). Each new clean field above $\\tau$ is a false alarm, labelled "
            f"with its case number: {'; '.join(alarms)}.")

    rho = {score: spearmanr(clean[score], clean["k"]).statistic for score in SCORE_NAMES}
    k_scores = [score for score in SCORE_NAMES if rho[score] < -0.5]
    if k_scores:
        text += (f" {_and([_short(s) for s in k_scores])} score the clean fields of $k = 1$ highest, as the "
                 "pre-registered protocol expects (Spearman correlation of the clean score with $k$: "
                 + _and([_signed(rho[s]) for s in k_scores]) + ").")
    energy_scores = [score for score in SCORE_NAMES if "E" in SCORE_DEFINITIONS[score]]
    energy_alarms = sorted(set().union(*(set(clean.index[clean[f"flagged_{s}"]]) for s in energy_scores)))
    if energy_alarms:
        u0 = _and([f"{case_by_id[c]['U0']:.2f}" for c in energy_alarms])
        e_bar = _and([f"{clean.loc[c, 'E_bar']:.3g}" for c in energy_alarms])
        text += (f" {_and([_short(s) for s in energy_scores])} flag {_tex_cases(energy_alarms)} instead "
                 f"($k = {', '.join(str(case_by_id[c]['k']) for c in energy_alarms)}$), whose normalized energy terms "
                 f"$\\bar S_E$ are {e_bar}. $S_E$, the mean squared gap between the kinetic energy and its exact "
                 "decay, is not scaled by $U_0$, so it grows like $U_0^4$ times the squared relative energy error. "
                 f"These cases have $U_0$ = {u0}")
        energy = pd.concat([relative_energy_error(clean), relative_energy_error(old.set_index("case_id"))])
        top = energy.sort_values(ascending=False).index[:2].tolist()
        if set(top) <= set(energy_alarms):
            old_max = relative_energy_error(old.set_index("case_id")).max()
            text += (f", and {_tex_cases(top)} have the largest relative energy errors of all {n_new + n_old} "
                     f"models ($\\sqrt{{S_E}}$ over the exact initial energy $U_0^2/4$: "
                     f"{_and([f'{100 * energy[c]:.2f}' for c in top])}\\%, against at most {100 * old_max:.2f}\\% "
                     "for the old cases)")
        text += " (Fig.~\\ref{fig:energy_term_vs_u0})."
    return text + " " + SCORE_KEY


def _seed_rule(table: pd.DataFrame, column: str) -> str:
    """' ($3000 + n$)' when every case n has seed 3000 + n in column, else ''."""
    offsets = {int(seed) - int(case_id.split("_")[1]) for case_id, seed in zip(table["case_id"], table[column])}
    return f" (${offsets.pop()} + n$)" if len(offsets) == 1 else ""


def caption_cases(table: pd.DataFrame) -> str:
    n_passed = int(table["passed_acceptance"].sum())
    by_k = table.groupby("k")["relL2_uv"]
    ranges = "; ".join(f"$k = {k}$: {_percent(values.min())}--{_percent(values.max())}\\%" for k, values in by_k)
    passed = (f"All {len(table)} runs passed" if n_passed == len(table)
              else f"{n_passed} of {len(table)} runs passed (failed runs stay in every table)")
    windows = sorted(table["T"].unique())
    window = f"which is {windows[0]:g} for every case" if len(windows) == 1 else "given in its own column"
    return (f"The {len(table)} new cases, pre-registered before training. Each is the decaying Taylor--Green vortex on "
            "the periodic square $[0, 2\\pi]^2$, $u = U_0 \\sin(kx + \\phi_x)\\cos(ky + \\phi_y)\\,e^{-t/\\tau_d}$, "
            "$v = -U_0 \\cos(kx + \\phi_x)\\sin(ky + \\phi_y)\\,e^{-t/\\tau_d}$, with wavenumber $k$, velocity scale "
            "$U_0$, phases $\\phi_x$, $\\phi_y$, Reynolds number Re (on the length $1/k$), viscosity "
            "$\\nu = U_0/(\\mathrm{Re}\\,k)$ and velocity decay time $\\tau_d = 1/(2\\nu k^2)$. Each model is trained "
            f"over $0 \\le t \\le T$, $T = \\min(2, \\tau_d)$, {window}. Seeds of case $n$: of its training and "
            f"evaluation points{_seed_rule(table, 'seed_points')}, of its weights and "
            f"mini-batches{_seed_rule(table, 'seed_train')} and of its scoring points"
            f"{_seed_rule(table, 'seed_scoring')}. Passed: the final model passes the training acceptance test "
            f"(velocity relative $L_2$ error below {ACCEPT_REL_L2_BELOW:g} on the $64\\times64\\times20$ evaluation "
            f"grid, MSE($R_c$) below $10^{{{int(np.log10(ACCEPT_MSE_RC_BELOW))}}}$ and MSE($R_m$) below "
            f"$10^{{{int(np.log10(ACCEPT_MSE_RM_BELOW))}}}$). Velocity error: the final model's relative $L_2$ "
            "velocity error on that evaluation grid, on an independent $64\\times64\\times11$ test grid, and on that "
            "test grid at $t = T$ alone; pressure error: on the test grid, with the spatial mean of the pressure "
            f"removed at each time from both the model and the exact solution. {passed}; test-grid velocity errors "
            f"{ranges}; at $t = T$, {_percent(table['relL2_uv_T'].min())}--{_percent(table['relL2_uv_T'].max())}\\%; "
            f"pressure errors {_percent(table['relL2_p'].min())}--{_percent(table['relL2_p'].max())}\\%.")


def caption_recall_by_k_table(table: pd.DataFrame) -> str:
    n_cases = int(table["n_damaged"].iloc[0])
    return (f"Damaged fields flagged by each score, by $k$, damage type and strength, out of the {n_cases} cases "
            "with that $k$ (frozen calibration, flagged above $\\tau$). The .csv adds the exact 95\\% confidence "
            "intervals and the "
            "counts over the cases whose clean field the score doesn't flag.")


def _flags(table: pd.DataFrame, score: str) -> np.ndarray:
    return table[f"flagged_{score}"].to_numpy(dtype=bool)


def caption_natural(natural: pd.DataFrame) -> str:
    final = (natural["stage"] == FINAL_STAGE).to_numpy()
    inaccurate = natural["inaccurate"].to_numpy(dtype=bool)
    adam_steps = sorted(int(step) for step in natural.loc[~final, "adam_step"].unique())
    n_inaccurate = int(inaccurate.sum())
    text = (f"The score of each of the {len(natural)} checkpoints of the {natural['case_id'].nunique()} new runs "
            f"(after {_and([str(step) for step in adam_steps])} Adam steps, and the final model after L-BFGS) against "
            "its velocity error relative to the exact solution, on the $64\\times64\\times11$ test grid. Each score is "
            "divided by its frozen $\\tau$, so the horizontal line at 1 is $\\tau$ (flagged above it); dashed line: a "
            f"model is inaccurate above {100 * INACCURATE_ABOVE:g}\\% error. Colour: the training stage, light to dark; "
            "marker: $k$. Shaded: where false alarms (accurate, flagged) and missed failures (inaccurate, not flagged) "
            "would lie.")
    missed = {score: int((inaccurate & ~_flags(natural, score)).sum()) for score in SCORE_NAMES}
    if not any(missed.values()):
        text += f" Every score flags all {n_inaccurate} inaccurate checkpoints."
    else:
        text += (" Missed failures, of " + str(n_inaccurate) + ": "
                 + _and([f"{_short(score)} {missed[score]}" for score in SCORE_NAMES]) + ".")
    accurate_adam = ~inaccurate & ~final
    flagged_by_all = np.logical_and.reduce([_flags(natural, score) for score in SCORE_NAMES])
    if accurate_adam.any() and flagged_by_all[accurate_adam].all():
        lowest = 100 * natural.loc[accurate_adam, "relL2_uv"].min()
        on_final = _and([f"{_short(s)} {int((final & ~inaccurate & _flags(natural, s)).sum())}" for s in SCORE_NAMES])
        text += (f" The false alarms are the {int(accurate_adam.sum())} Adam checkpoints with errors from "
                 f"{lowest:.2f} to {100 * INACCURATE_ABOVE:g}\\%, which every score flags, and the false alarms on the "
                 f"{int((final & ~inaccurate).sum())} accurate final models ({on_final}).")
    worst_final = natural[final].sort_values("relL2_uv").iloc[-1]
    best_adam = natural[~final].sort_values("relL2_uv").iloc[0]
    if worst_final["relL2_uv"] < best_adam["relL2_uv"]:
        text += (f" No checkpoint has an error between {_percent(worst_final['relL2_uv'])}\\% (the least accurate "
                 f"final model, {_case_name(worst_final['case_id'])}) and {_percent(best_adam['relL2_uv'])}\\% "
                 "(the most accurate Adam checkpoint).")
    return text + " " + SCORE_KEY


def _damaged_with_sizes(fields: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """The damaged fields of the benchmark, each with the velocity change its damage makes (damage_sizes())."""
    damaged = fields[fields["label"] != "clean"]
    merged = damaged.merge(sizes[["case_id", "perturbation", "epsilon", "velocity_change"]],
                           on=["case_id", "perturbation", "epsilon"], how="left", validate="one_to_one")
    if merged["velocity_change"].isna().any():
        raise ValueError("Some damaged fields have no damage size")
    return merged


def _unchanged_types(sizes: pd.DataFrame) -> list:
    """The damage types that never change the velocity (pressure)."""
    return sorted((p for p, group in sizes.groupby("perturbation") if (group["velocity_change"] == 0).all()),
                  key=bm._perturbation_order)


def score_rises(fields: pd.DataFrame, sizes: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    The damaged fields that change the velocity, each with the velocity change its damage
    makes (velocity_change) and, for each score, what the damage adds to the score over
    its case's own clean field, divided by the score's tau (rise_<score>). That is the
    part of the score that the damage causes, as a checkpoint's whole score is the part
    its own error causes (the exact solution has no residual and no energy gap, so it
    scores zero). The exception is the bc term, a near/far ratio of the residual that
    doesn't shrink with the error; caption_shared() says how much of the final models'
    scores it makes up.
    """
    damaged = _damaged_with_sizes(fields, sizes)
    damaged = damaged[damaged["velocity_change"] > 0].copy()
    clean = fields[fields["label"] == "clean"].set_index("case_id")
    for score in SCORE_NAMES:
        damaged[f"rise_{score}"] = (damaged[score] - damaged["case_id"].map(clean[score])) / thresholds[score]
    return damaged


def caption_shared(fields: pd.DataFrame, sizes: pd.DataFrame, natural: pd.DataFrame, thresholds: dict) -> str:
    rises = score_rises(fields, sizes, thresholds)
    final = natural[(natural["stage"] == FINAL_STAGE).to_numpy()]
    n_cases = fields["case_id"].nunique()
    unchanged = _unchanged_types(sizes)
    text = ("Both experiments on one axis: what a deviation adds to each score, over the score's frozen $\\tau$, "
            "against the size of the deviation, relative to the exact velocity on the $64\\times64\\times11$ test grid. "
            f"Synthetic damage (orange; marker: damage type): each damaged field of the {n_cases} new cases, at the "
            "velocity change its damage makes (Table~\\ref{tab:damage_size}) and the rise of its score over its own "
            "clean field. Natural errors (blue, light to dark: the training stage): each checkpoint, at its own "
            "velocity error and its whole score, since the exact solution scores zero (it has no residual and no "
            "energy gap). Line: a rise of one $\\tau$; a field is flagged when its whole score, not its rise, is "
            "above $\\tau$.")
    hidden = {score: rises[rises[f"rise_{score}"] <= 0] for score in SCORE_NAMES}
    left_out = []
    if unchanged:
        left_out.append(f"the {_and([DAMAGE_LABELS[p] for p in unchanged])} damage, which changes only the pressure")
    if any(len(rows) for rows in hidden.values()):
        strengths = sorted(set().union(*(set(rows["epsilon"]) for rows in hidden.values())))
        left_out.append(f"the fields whose score did not rise ({_and([str(len(hidden[s])) for s in SCORE_NAMES])} "
                        f"for Score1 to Score5, all at $\\varepsilon$ = {_and([f'{e:g}' for e in strengths])})")
    if left_out:
        text += " Not shown: " + "; ".join(left_out) + "."
    low, high = final["relL2_uv"].min(), final["relL2_uv"].max()
    same_size = rises[(rises["velocity_change"] >= low) & (rises["velocity_change"] <= high)]
    if len(same_size):
        rise = _and([_two_digits(same_size[f"rise_{s}"].median()) for s in SCORE_NAMES])
        whole = _and([_two_digits((final[s] / thresholds[s]).median()) for s in SCORE_NAMES])
        text += (f" Among the {len(same_size)} damages of the same size as the final models' own errors "
                 f"({_percent(low)}--{_percent(high)}\\%), the median rise is {rise} $\\tau$ for Score1 to Score5; "
                 f"the final models' median scores are {whole} $\\tau$.")
    # bc is a ratio of residuals, so it doesn't shrink with the error: part of a checkpoint's whole score is not
    # "what its error adds" (the exact solution's 0 comes only from the guard in the ratio's denominator)
    bc_scores = [score for score in SCORE_NAMES if "bc" in SCORE_DEFINITIONS[score]]
    if bc_scores:
        part = _and([_two_digits((final["bc_bar"] / thresholds[s]).median()) for s in bc_scores])
        text += (" Not all of a checkpoint's score is what its error adds: the bc term of "
                 f"{_and([_short(s) for s in bc_scores])}, a ratio of the residual near the edges to the residual far "
                 "from them, does not shrink with the error, and makes up a median of "
                 f"{part} $\\tau$ of the final models' scores.")
    return text + " " + SCORE_KEY


def caption_summary(summary: pd.DataFrame, natural: pd.DataFrame, fields: pd.DataFrame, type_eps: pd.DataFrame) -> str:
    pooled = summary[summary["k"] == ALL]
    synthetic = pooled[pooled["experiment"] == SYNTHETIC].iloc[0]
    checkpoints = pooled[pooled["experiment"] == NATURAL].iloc[0]
    final = (natural["stage"] == FINAL_STAGE).to_numpy()
    accurate = ~natural["inaccurate"].to_numpy(dtype=bool)
    flagged_by_all = np.logical_and.reduce([_flags(natural, score) for score in SCORE_NAMES])
    half_trained = accurate & ~final
    # The pooled synthetic recall leans on the strengths tested: above and below the strength from which
    # most scores flag everything
    common = pd.Series(full_recall_epsilon(type_eps, "recall")).dropna().mode().min()
    damaged = fields[fields["label"] != "clean"]
    strong, weak = damaged[damaged["epsilon"] >= common], damaged[damaged["epsilon"] < common]
    recall_strong = [_flags(strong, score).mean() for score in SCORE_NAMES]
    recall_weak = [_flags(weak, score).mean() for score in SCORE_NAMES]
    n_strong, n_weak = strong["epsilon"].nunique(), weak["epsilon"].nunique()
    energy_scores = [score for score in SCORE_NAMES if "E" in SCORE_DEFINITIONS[score]]
    caveats = (f" Caveats for reading the two side by side. The pooled synthetic recall depends on which strengths "
               f"were tested: over the {n_strong} strengths from $\\varepsilon = {common:g}$ on it is "
               f"{min(recall_strong):.3f}--{max(recall_strong):.3f}, over the {n_weak} below it "
               f"{min(recall_weak):.3f}--{max(recall_weak):.3f}, depending on the score. The natural numbers depend on "
               f"which checkpoints were saved and on the {100 * INACCURATE_ABOVE:g}\\% rule")
    if half_trained.any() and flagged_by_all[half_trained].all():
        errors = 100 * natural.loc[half_trained, "relL2_uv"]
        caveats += (f": the {int(half_trained.sum())} accurate Adam checkpoints ({errors.min():.2f}--"
                    f"{100 * INACCURATE_ABOVE:g}\\% error) are flagged by every score and make most of the false "
                    "alarms")
    caveats += (f". {_and([_short(s) for s in energy_scores])} hold the energy term, the only term that compares the "
                "model with the exact solution, against which the true error is also measured, so their higher AUC "
                "on the checkpoints is partly built in.")
    return (f"Both experiments at the frozen $\\tau$, in the same terms. Synthetic damage: the "
            f"{int(synthetic['n_positive'] + synthetic['n_negative'])} fields of the {int(synthetic['n_cases'])} new "
            f"cases; positive: the {int(synthetic['n_positive'])} damaged fields, negative: the "
            f"{int(synthetic['n_negative'])} clean ones. Natural errors: the "
            f"{int(checkpoints['n_positive'] + checkpoints['n_negative'])} checkpoints of the "
            f"{int(checkpoints['n_cases'])} runs; positive: the {int(checkpoints['n_positive'])} inaccurate ones "
            f"(velocity error above {100 * INACCURATE_ABOVE:g}\\%), negative: the {int(checkpoints['n_negative'])} "
            f"accurate ones, {int((accurate & final).sum())} final models and {int((accurate & ~final).sum())} Adam "
            "checkpoints. AUC: positives against negatives; recall: the share of positives flagged; both with a "
            f"case-bootstrap 95\\% confidence interval ({N_BOOT} resamples of whole cases within each $k$). False "
            "alarms: negatives flagged. The .csv adds the confidence intervals of the false-alarm rates (exact for the "
            "clean fields, one per case; case bootstrap for the checkpoints) and the same numbers for each $k$."
            + caveats + " " + SCORE_KEY)


def caption_term_auc(table: pd.DataFrame, checkpoints: pd.DataFrame) -> str:
    synthetic = table[(table["experiment"] == SYNTHETIC) & (table["group"] == ALL)].set_index("term")
    per_type = table[(table["experiment"] == SYNTHETIC) & (table["group"] != ALL)]
    natural = table[table["experiment"] == NATURAL].set_index("term")
    first = synthetic.iloc[0]
    name = {term: label.split(":")[0] for term, label in TERM_LABELS.items()}
    text = ("The AUC of each score term on its own, with its case-bootstrap 95\\% confidence interval "
            f"({N_BOOT} resamples of whole cases within each $k$). Synthetic damage: the "
            f"{int(first['n_positive'])} damaged fields of the {int(first['n_cases'])} new cases against their "
            f"{int(first['n_negative'])} clean fields, over every type and for each type "
            f"({int(per_type['n_positive'].iloc[0])} against {int(per_type['n_negative'].iloc[0])}). Natural errors: "
            f"the {int(natural['n_positive'].iloc[0])} inaccurate checkpoints against the "
            f"{int(natural['n_negative'].iloc[0])} accurate ones. Below 0.5, the term is lower on the damaged (or "
            "inaccurate) fields than on the others.")
    unchanged = per_type[per_type["unchanged"]]
    if len(unchanged):
        by_type = {}
        for _, row in unchanged.iterrows():
            by_type.setdefault(row["group"], []).append(name[row["term"]])
        text += (" $^\\dagger$: the damage leaves the term exactly as it was on every field ("
                 + "; ".join(f"{DAMAGE_LABELS[g]} damage: {_and(terms)}" for g, terms in by_type.items())
                 + "), so its AUC is 0.5 by construction.")
    best = synthetic["auc"].idxmax()
    below = [term for term in BAR_NAMES if synthetic.loc[term, "auc"] < 0.5]
    text += f" Over all types, {name[best]} alone reaches {_range_text(synthetic.loc[best], 'auc')}"
    if below:
        text += ("; " + _and([f"{name[t]} alone ranks damaged fields below clean ones "
                              f"({_range_text(synthetic.loc[t], 'auc')})" for t in below]))
    text += "."
    below_natural = [term for term in BAR_NAMES if natural.loc[term, "auc"] < 0.5]
    if below_natural:
        text += (" On the checkpoints, " + _and([f"{name[t]} alone ranks the inaccurate ones below the accurate ones "
                                                f"({_range_text(natural.loc[t], 'auc')})" for t in below_natural]))
        if "bc_bar" in below_natural:
            ratio = checkpoints.groupby("stage")["bc"].median()
            first = [stage for stage in ne.STAGES if stage in ratio.index][0]
            text += (f": its near-to-far residual ratio grows during training, from a median of {ratio[first]:.2f} "
                     f"({ne.stage_label(first)}) to {ratio[FINAL_STAGE]:.2f} for the final models, since a half-trained "
                     "model's residual lies mostly away from the edges")
        text += "."
    energy = "E_bar"
    if energy in natural.index:
        text += (f" On the checkpoints, {name[energy]} alone reaches {_range_text(natural.loc[energy], 'auc')}; it is "
                 "the only term that compares the model with the exact solution, against which the true error is "
                 "also measured.")
    return text + " " + SCORE_KEY


def _energy_scores() -> list:
    """The scores that hold the energy term."""
    return [score for score in SCORE_NAMES if "E" in SCORE_DEFINITIONS[score]]


def _energy_points(fields: pd.DataFrame, old: pd.DataFrame, cases: list) -> tuple:
    """
    (new, old): the clean fields of the new and the old cases with case_id, k, U0, E,
    E_bar and their relative energy error (relative_energy_error()); the new ones also
    with energy_alarm, whether a score holding the energy term flags them.
    """
    case_by_id = {case["case_id"]: case for case in cases}
    clean = fields[fields["label"] == "clean"]
    new = pd.DataFrame({"case_id": clean["case_id"].to_numpy(), "k": clean["k"].to_numpy(),
                        "U0": [case_by_id[c]["U0"] for c in clean["case_id"]], "E": clean["E"].to_numpy(),
                        "E_bar": clean["E_bar"].to_numpy(),
                        "energy_alarm": np.logical_or.reduce([_flags(clean, s) for s in _energy_scores()])})
    new["relative"] = relative_energy_error(new).to_numpy()
    old_points = old[["case_id", "k", "U0", "E", "E_bar"]].copy()
    old_points["relative"] = relative_energy_error(old_points).to_numpy()
    return new, old_points


def _vanishing(maps: dict, key: str) -> list:
    """The damage types whose map `key` is zero to rounding (below 1e-9 of the largest map of that row)."""
    largest = max(m[key].max() for m in maps.values())
    return [p for p, m in maps.items() if m[key].max() <= 1e-9 * largest]


def caption_damage_fields(info: dict, case: dict, epsilon: float) -> str:
    maps = info["maps"]
    n = len(info["axis"])
    pressure_only = [p for p, m in maps.items() if m["change_of"] == "pressure"]
    text = (f"What each synthetic damage does, at $\\varepsilon = {epsilon:g}$, to the exact solution of "
            f"{_case_name(case['case_id'])} ($k = {case['k']}$, $U_0 = {case['U0']:.2f}$, Re $= {case['Re']:.1f}$) at "
            f"$t = {info['t']:g}$, half its time window, on a ${n}\\times{n}$ grid of the periodic square ($x$ across, "
            "$y$ up). Top row: the change it makes, the velocity change $|\\Delta\\mathbf{u}|/U_0$")
    if pressure_only:
        text += (f", or for the {_and([DAMAGE_LABELS[p] for p in pressure_only])} damage, which leaves the velocity "
                 "alone, the pressure change $|\\Delta p|/U_0^2$")
    text += ("; middle row: the momentum residual it creates, $|\\mathbf{R}_m|/(U_0^2 k)$; bottom row: its continuity "
             "residual, $|R_c|/(U_0 k)$. The residuals are scaled as in the scores, whose momentum and continuity "
             "terms are the mean squares of such maps over space and time. The exact solution has no residual, so "
             "every residual shown is the damage's own; a trained model, which differs from the exact solution by "
             "less than 1\\%, gets nearly the same residual from the same damage. Each row has one colour scale.")
    free = _vanishing(maps, "continuity")
    if free:
        verb = "damages keep" if len(free) > 1 else "damage keeps"
        text += f" The {_and([DAMAGE_LABELS[p] for p in free])} {verb} the flow divergence-free."
    if "temporal_mismatch" in maps:
        text += (" The time shift reports the velocity at a later time but keeps the pressure of the original one; "
                 "its momentum residual is that mismatch.")
    return text


def caption_energy(fields: pd.DataFrame, old: pd.DataFrame, cases: list) -> str:
    new, old_points = _energy_points(fields, old, cases)
    both = pd.concat([new, old_points])
    slope = np.polyfit(np.log(both["U0"]), np.log(both["E_bar"]), 1)[0]
    rho = spearmanr(both["relative"], both["U0"]).statistic
    text = (f"The energy term of the clean field of each of the {len(new)} new cases (filled) and the {len(old_points)} "
            "old calibration cases (open) against the velocity scale $U_0$ (log axes; marker: $k$). (a) The normalized "
            "energy term $\\bar S_E$. (b) The relative energy error $\\sqrt{S_E}/(U_0^2/4)$: the root-mean-square gap "
            "between the model's kinetic energy and its exact decay, over the exact initial energy. $S_E$ is that gap "
            "squared in absolute units, so it grows like $U_0^4$ times the squared relative error (dashed line: "
            f"slope 4). Over the {len(both)} fields the fitted slope is {slope:.1f}, steeper, because the relative "
            f"error itself grows with $U_0$ (Spearman correlation {rho:.2f}).")
    labelled = new[new["energy_alarm"]]
    if len(labelled):
        text += (f" Labelled: the new clean fields that {_and([_short(s) for s in _energy_scores()])} flag, "
                 f"{_tex_cases(labelled['case_id'].tolist())} ($U_0$ = "
                 f"{_and([f'{u:.2f}' for u in labelled['U0']])})")
        old_max = old_points.groupby("k")["U0"].max()
        if all(row["U0"] > old_max[row["k"]] for _, row in labelled.iterrows()):
            text += (", each with a $U_0$ above every old case of its $k$ (old maxima "
                     + _and([f"{old_max[k]:.2f} for $k = {k}$" for k in sorted(set(labelled["k"]))]) + ")")
        text += "."
    return text


def caption_damage_size(table: pd.DataFrame, n_cases: int) -> str:
    unchanged = [p for p, group in table.groupby("perturbation", sort=False) if (group["velocity_change_max"] == 0).all()]
    text = ("The size of each synthetic damage as a velocity error: the relative $L_2$ change it makes to the velocity, "
            "against the exact velocity on the $64\\times64\\times11$ test grid of the accuracy audit, in \\%: the "
            f"median over the {n_cases} new cases, and the smallest and largest. Computed with the benchmark's damage "
            "code on the exact solution in place of the model: an added damage changes any field by the same amount, "
            "and the time shift's size is its change to the exact flow.")
    if unchanged:
        text += f" The {_and([DAMAGE_LABELS[p] for p in unchanged])} damage changes only the pressure."
    shift = table[table["perturbation"] == "temporal_mismatch"].sort_values("epsilon")
    smallest = shift["velocity_change_min"].to_numpy()
    flat = [i for i in range(1, len(smallest)) if smallest[i] < 1.01 * smallest[i - 1]]
    if flat:
        start = shift["epsilon"].iloc[flat[0] - 1]
        text += (f" For the time shift, the smallest change stops growing from $\\varepsilon = {start:g}$ on: in the "
                 "cases with the shortest decay time the shifted time reaches the end of the window $T$ and is held "
                 "there.")
    return text


# --- Figures ---

def save_figure(fig, output_dir: Path, stem: str, top_legend=None) -> list:
    """
    Saves a figure as <stem>.pdf, for the paper, without a title (its caption is the
    title) and without a creation date, so a rerun gives the same file; then as
    <stem>.png with its one-line title (PNG_TITLES), to be read on its own.

    A figure legend placed above the panels (top_legend) sits at the very top of the
    figure, where a title would land on it, so there the title becomes the legend's own
    title and the layout makes room for both. Otherwise the title goes in a band added
    above the figure, and the layout below it stays the PDF's.
    """
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    # dpi sets only the resolution of an embedded image (fig_damage_fields); lines and text stay vector
    fig.savefig(pdf_path, metadata={"CreationDate": None}, dpi=PNG_DPI)
    if top_legend is not None:
        top_legend.set_title(PNG_TITLES[stem], prop={"size": PNG_TITLE_SIZE})
    else:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + PNG_TITLE_BAND)
        fig.get_layout_engine().set(rect=(0, 0, 1, height / (height + PNG_TITLE_BAND)))
        fig.text(0.5, 1 - 0.5 * PNG_TITLE_BAND / (height + PNG_TITLE_BAND), PNG_TITLES[stem], ha="center",
                 va="center", fontsize=PNG_TITLE_SIZE, color=ne.INK)
    fig.savefig(png_path, dpi=PNG_DPI)
    return [pdf_path, png_path]


def figure_tex(stem: str, caption: str, wide: bool) -> str:
    """The figure environment of a figure, with its caption and label fig:<stem without fig_>."""
    environment, width = ("figure*", "\\textwidth") if wide else ("figure", "\\columnwidth")
    return (f"% Written by src/detection/final_figures.py; needs \\usepackage{{graphicx}} and results/figures/ "
            f"in \\graphicspath.\n"
            f"\\begin{{{environment}}}[tb]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width={width}]{{{stem}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:{stem[len('fig_'):]}}}\n"
            f"\\end{{{environment}}}\n")


def _style(ax) -> None:
    """Hairline grid behind the data."""
    ax.grid(True, which="major", color=ne.GRID_COLOUR, linewidth=0.5)
    ax.set_axisbelow(True)


def _letter(ax, index: int) -> None:
    """The panel's letter, (a), (b), ..., in bold in its top-left corner, on white so no line runs through it."""
    ax.text(0.03, 0.97, f"({chr(ord('a') + index)})", transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            fontweight="bold", color=ne.INK, zorder=6,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none"})


def _score_handles(bands: bool = False) -> list:
    """The legend entries of the five scores, and with bands, of the grey band of a shared range."""
    handles = [Line2D([], [], color=SCORE_COLOURS[s], linewidth=LINE_WIDTH, marker=SCORE_MARKERS[s],
                      markersize=MARKER_SIZE + 0.6, markeredgecolor="white", markeredgewidth=0.5, label=SCORE_LABELS[s])
               for s in SCORE_NAMES]
    if bands:
        handles.append(Patch(facecolor=SHARED_BAND_COLOUR, alpha=SHARED_BAND_ALPHA, linewidth=0,
                             label="shared 95% CI"))
    return handles


def _epsilon_axis(ax, epsilons) -> None:
    """Log axis over the strengths, decade ticks, a little room at both ends."""
    ax.set_xscale("log")
    ax.set_xlim(min(epsilons) / 2.2, max(epsilons) * 1.8)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: {1e-4: "10\u207b\u2074", 1e-3: "10\u207b\u00b3",
                                                                 1e-2: "10\u207b\u00b2"}.get(value, "")))
    ax.xaxis.set_minor_locator(FixedLocator(list(epsilons)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _recall_lines(ax, table: pd.DataFrame, perturbation: str, rate: str) -> None:
    """
    One line per score for one damage type, with its exact range as a band. Equal counts
    give equal ranges, so a stretch of range that several scores share is drawn once, in
    grey, instead of five times on top of itself; a range only one score has keeps its colour.
    """
    rows = {score: table[(table["score"] == score) & (table["perturbation"] == perturbation)].sort_values("epsilon")
            for score in SCORE_NAMES}
    x = rows[SCORE_NAMES[0]]["epsilon"].to_numpy(dtype=float)
    if any(not np.array_equal(r["epsilon"].to_numpy(dtype=float), x) for r in rows.values()):
        raise ValueError(f"The scores don't share the same strengths for {perturbation}")
    low = {score: r[f"{rate}_low"].to_numpy(dtype=float) for score, r in rows.items()}
    high = {score: r[f"{rate}_high"].to_numpy(dtype=float) for score, r in rows.items()}
    for i in range(len(x) - 1):
        stretches = {}  # (low, high at x[i], low, high at x[i + 1]) -> the scores with that range
        for score in SCORE_NAMES:
            key = (low[score][i], high[score][i], low[score][i + 1], high[score][i + 1])
            if not np.isnan(key).any():
                stretches.setdefault(key, []).append(score)
        for (low_a, high_a, low_b, high_b), scores in stretches.items():
            shared = len(scores) > 1
            ax.fill_between(x[i:i + 2], [low_a, low_b], [high_a, high_b], linewidth=0,
                            color=SHARED_BAND_COLOUR if shared else SCORE_COLOURS[scores[0]],
                            alpha=SHARED_BAND_ALPHA if shared else BAND_ALPHA)
    for score in SCORE_NAMES:
        ax.plot(x, rows[score][rate].to_numpy(dtype=float), color=SCORE_COLOURS[score], linewidth=LINE_WIDTH,
                marker=SCORE_MARKERS[score], markersize=MARKER_SIZE, markeredgecolor="white",
                markeredgewidth=0.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def plot_recall(table: pd.DataFrame, rate: str, output_dir, stem: str) -> list:
    """
    fig_recall_vs_eps and fig_recall_vs_eps_new_detections: recall (rate "recall" or
    "recall_new") against the strength, one panel per damage type, one line per score,
    with the exact ranges as bands.
    """
    perturbations = sorted(table["perturbation"].unique(), key=bm._perturbation_order)
    epsilons = sorted(table["epsilon"].unique())
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(1, len(perturbations), figsize=(TEXT_WIDTH, 2.05), sharey=True, layout="constrained")
        for i, (ax, perturbation) in enumerate(zip(axes, perturbations)):
            _style(ax)
            _recall_lines(ax, table, perturbation, rate)
            _epsilon_axis(ax, epsilons)
            ax.set_ylim(-0.04, 1.04)
            ax.set_title(DAMAGE_LABELS[perturbation])
            _letter(ax, i)
        axes[0].set_ylabel("recall" if rate == "recall" else "recall, new detections")
        fig.supxlabel("damage strength \u03b5")
        legend = fig.legend(handles=_score_handles(bands=True), loc="outside upper center", ncol=len(SCORE_NAMES) + 1,
                            frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0)
        paths = save_figure(fig, Path(output_dir), stem, top_legend=legend)
        plt.close(fig)
    return paths


def plot_recall_by_k(table: pd.DataFrame, output_dir) -> list:
    """fig_recall_vs_eps_by_k: the recall of every score, one row of panels per k, one column per damage type."""
    perturbations = sorted(table["perturbation"].unique(), key=bm._perturbation_order)
    epsilons = sorted(table["epsilon"].unique())
    ks = sorted(table["k"].unique())
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(len(ks), len(perturbations), figsize=(TEXT_WIDTH, 1.42 * len(ks) + 0.55),
                                 sharex=True, sharey=True, layout="constrained", squeeze=False)
        for i, k in enumerate(ks):
            of_k = table[table["k"] == k]
            for j, perturbation in enumerate(perturbations):
                ax = axes[i, j]
                _style(ax)
                _recall_lines(ax, of_k, perturbation, "recall")
                _epsilon_axis(ax, epsilons)
                ax.set_ylim(-0.04, 1.04)
                if i == 0:
                    ax.set_title(DAMAGE_LABELS[perturbation])
                _letter(ax, i * len(perturbations) + j)
            axes[i, 0].set_ylabel(f"k = {k}\nrecall")
        fig.supxlabel("damage strength \u03b5")
        legend = fig.legend(handles=_score_handles(bands=True), loc="outside upper center", ncol=len(SCORE_NAMES) + 1,
                            frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0)
        paths = save_figure(fig, Path(output_dir), "fig_recall_vs_eps_by_k", top_legend=legend)
        plt.close(fig)
    return paths


def _tau_point(ax, x: float, y: float, score: str, size: float) -> None:
    """The point of a score's frozen tau: its marker as an outline only, so that points that nearly coincide all show."""
    ax.plot([x], [y], linestyle="", marker=SCORE_MARKERS[score], markersize=size, markerfacecolor="none",
            markeredgecolor=SCORE_COLOURS[score], markeredgewidth=1.0, zorder=5)


def plot_roc(fields: pd.DataFrame, overall: pd.DataFrame, output_dir) -> list:
    """
    fig_roc: the ROC curve of every score, with its AUC and range in the legend and the
    point of its frozen tau (open marker), and an inset that enlarges the top-left
    corner, where the curves differ and the tau points lie.
    """
    damaged = (fields["label"] != "clean").to_numpy()
    curves, points = {}, {}
    for score in SCORE_NAMES:
        fpr, tpr, _ = roc_curve(damaged, fields[score].to_numpy(dtype=float), drop_intermediate=False)
        curves[score] = (fpr, tpr)
        flagged = fields[f"flagged_{score}"].to_numpy(dtype=bool)
        points[score] = ((flagged & ~damaged).sum() / (~damaged).sum(), (flagged & damaged).sum() / damaged.sum())
    # The inset: false-alarm rates up to 0.2, recall from just below the lowest curve start or tau point
    starts = [tpr[fpr == 0].max() for fpr, tpr in curves.values()]
    low = np.floor(20 * min(*starts, *(y for _, y in points.values()))) / 20 - 0.02
    high = max(tpr[fpr <= 0.2].max() for fpr, tpr in curves.values())
    inset_x, inset_y = (-0.008, 0.205), (low, min(1.0, np.ceil(20 * high) / 20 + 0.01))

    with plt.rc_context(PRINT_RC):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.4), layout="constrained")
        inset = ax.inset_axes([0.40, 0.40, 0.57, 0.40])
        for axis in (ax, inset):
            _style(axis)
            axis.plot([0, 1], [0, 1], color=ne.INK_MUTED, linewidth=0.6, zorder=1)
            for score in SCORE_NAMES:
                fpr, tpr = curves[score]
                axis.plot(fpr, tpr, color=SCORE_COLOURS[score], linewidth=LINE_WIDTH, solid_joinstyle="round",
                          zorder=2)
        for score in SCORE_NAMES:
            for axis, size in ((ax, MARKER_SIZE + 1.6), (inset, MARKER_SIZE + 2.6)):
                _tau_point(axis, *points[score], score, size)
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0.0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("false-alarm rate (clean fields flagged)")
        ax.set_ylabel("recall (damaged fields flagged)")
        inset.set_xlim(*inset_x)
        inset.set_ylim(*inset_y)
        inset.tick_params(labelsize=7)
        inset.set_facecolor("white")
        for side in ("top", "right"):
            inset.spines[side].set_visible(True)
        # The enlarged corner as a frame, without connector lines across the curves
        zoom = ax.indicate_inset_zoom(inset, edgecolor=ne.INK_MUTED, linewidth=0.5, alpha=1.0)
        for connector in getattr(zoom, "connectors", None) or []:
            connector.set_visible(False)

        # Legend in two columns, the scores and their AUC [range], so the numbers line up
        handles = [Line2D([], [], color=SCORE_COLOURS[s], linewidth=LINE_WIDTH, marker=SCORE_MARKERS[s],
                          markersize=MARKER_SIZE + 0.6, markerfacecolor="none", markeredgecolor=SCORE_COLOURS[s],
                          markeredgewidth=0.9, label=SCORE_LABELS[s]) for s in SCORE_NAMES]
        blanks = [Line2D([], [], linestyle="", label=_range_text(overall[overall["score"] == s].iloc[0], "auc"))
                  for s in SCORE_NAMES]
        ax.legend(handles=handles + blanks, ncol=2, title="AUC [95% CI]; open marker: the frozen \u03c4",
                  loc="lower right", frameon=False, alignment="left", handlelength=2.0, handletextpad=0.5,
                  columnspacing=-1.2, borderaxespad=0.2)
        paths = save_figure(fig, Path(output_dir), "fig_roc")
        plt.close(fig)
    return paths


def _strip_offsets(n: int, half_width: float) -> np.ndarray:
    """n positions spread evenly over [-half_width, half_width], in order (one point: 0)."""
    return np.zeros(1) if n == 1 else np.linspace(-half_width, half_width, n)


def _log_ticks(low: float, high: float) -> list:
    """The 1-2-5 ticks between low and high."""
    ticks = []
    for exponent in range(int(np.floor(np.log10(low))) - 1, int(np.ceil(np.log10(high))) + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10.0 ** exponent
            if low <= value <= high:
                ticks.append(value)
    return ticks


def _label_positions(values: list, min_ratio: float, floor: float = 0.0) -> list:
    """
    Label heights for points at `values` (log axis): each at its point but at least at
    floor, moved up where needed so that labels are at least a factor min_ratio apart, in
    the points' order. Only up, so the label of a point above tau never lands on or below
    the tau line (floor a little above tau).
    """
    placed = {}
    previous = None
    for i in np.argsort(values, kind="stable"):
        y = max(values[i], floor) if previous is None else max(values[i], floor, previous * min_ratio)
        placed[i] = y
        previous = y
    return [placed[i] for i in range(len(values))]


def plot_clean_scores(fields: pd.DataFrame, old: pd.DataFrame, thresholds: dict, output_dir) -> list:
    """
    fig_clean_scores_vs_tau: for each score (panel) and k (group), the clean scores of
    the new cases (filled, right) and of the old calibration cases (open, left), the
    frozen tau, and each false alarm labelled with its case number. Each case keeps its
    place in its group in every panel.
    """
    clean = fields[fields["label"] == "clean"].sort_values("case_id")
    ks = sorted(clean["k"].unique())
    old_x, new_x, half = -0.2, 0.12, 0.08
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.3), layout="constrained")
        for panel, (ax, score) in enumerate(zip(axes.flat, SCORE_NAMES)):
            _letter(ax, panel)
            tau = thresholds[score]
            values = np.concatenate([clean[score].to_numpy(dtype=float), old[score].to_numpy(dtype=float), [tau]])
            low, high = values.min() / 1.5, values.max() * 1.6
            _style(ax)
            ax.set_yscale("log")
            ax.set_ylim(low, high)
            ax.yaxis.set_major_locator(FixedLocator(_log_ticks(low, high)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.axhline(tau, color=ne.INK_SECONDARY, linewidth=0.8, zorder=1)
            # Named just outside the right edge, where no point can hide it; the value is in the title
            ax.text(1.01, tau, "\u03c4", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=ne.INK_SECONDARY, fontsize=7.5)
            n_alarms = 0
            for k in ks:
                old_k = old[old["k"] == k].sort_values("case_id")
                ax.scatter(k + old_x + _strip_offsets(len(old_k), half), old_k[score], s=11, facecolors="white",
                           edgecolors=OLD_COLOUR, linewidths=0.7, zorder=2)
                new_k = clean[clean["k"] == k]
                xs = k + new_x + _strip_offsets(len(new_k), half)
                ax.scatter(xs, new_k[score], s=13, color=K_COLOUR, edgecolors="white", linewidths=0.4, zorder=3)
                alarms = new_k[new_k[f"flagged_{score}"].to_numpy(dtype=bool)]
                n_alarms += len(alarms)
                if not alarms.empty:
                    alarm_x = xs[new_k[f"flagged_{score}"].to_numpy(dtype=bool)]
                    heights = _label_positions(alarms[score].tolist(), min_ratio=1.16, floor=tau * 1.08)
                    for x, y, label_y, case_id in zip(alarm_x, alarms[score], heights, alarms["case_id"]):
                        ax.annotate(case_id.split("_")[1], xy=(x, y), xytext=(k + new_x + half + 0.07, label_y),
                                    ha="left", va="center", fontsize=7, color=ne.INK,
                                    arrowprops={"arrowstyle": "-", "color": ne.INK_MUTED, "linewidth": 0.5,
                                                "shrinkA": 0, "shrinkB": 1.5})
            ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
            ax.set_xticks(ks, [f"k = {k}" for k in ks])
            ax.tick_params(axis="x", length=0)
            ax.grid(False, axis="x")
            ax.set_title(f"{SCORE_LABELS[score]}\n\u03c4 = {tau:.3g}, false alarms: {n_alarms} of {len(clean)}")
            ax.set_ylabel("clean-field score")

        legend_ax = axes.flat[len(SCORE_NAMES)]
        legend_ax.axis("off")
        handles = [
            Line2D([], [], linestyle="", marker="o", markersize=4.2, markerfacecolor=K_COLOUR, markeredgecolor="white",
                   label=f"new cases ({len(clean)}): clean field of the final model"),
            Line2D([], [], linestyle="", marker="o", markersize=4, markerfacecolor="white", markeredgecolor=OLD_COLOUR,
                   label=f"old cases ({len(old)}): the calibration fields"),
            Line2D([], [], color=ne.INK_SECONDARY, linewidth=0.8, label="\u03c4: 95th percentile of the old fields"),
        ]
        legend_ax.legend(handles=handles, loc="upper left", frameon=False, alignment="left", handletextpad=0.6,
                         borderaxespad=0.0)
        legend_ax.text(0.0, 0.42, "Numbers: the case number of each new\nclean field above \u03c4 (a false alarm).\n"
                                  "Each case keeps its place in its k group\nin every panel.",
                       transform=legend_ax.transAxes, ha="left", va="top", color=ne.INK_SECONDARY, fontsize=7)
        paths = save_figure(fig, Path(output_dir), "fig_clean_scores_vs_tau")
        plt.close(fig)
    return paths


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _power_label(value, _=None) -> str:
    """A log-axis tick: 0.1, 1, 10, 100, 1000 as they are, other powers of ten as 10⁴."""
    exponent = int(round(np.log10(value)))
    return f"{value:g}" if -1 <= exponent <= 3 else "10" + str(exponent).translate(_SUPERSCRIPTS)


def _ratio_panels(n_panels: int, x_range: tuple, y_range: tuple, ylabel: str = "score / τ"):
    """
    The 2 x 3 grid of a figure with one panel per score and the legend in the last slot:
    shared log axes, x an error in % and y a score / tau, each with its line at tau.
    """
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 4.5), sharex=True, sharey=True, layout="constrained")
    decades = range(int(np.floor(np.log10(x_range[0]))), int(np.ceil(np.log10(x_range[1]))) + 1)
    for ax in axes.flat[:n_panels]:
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in decades]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(_power_label))
        ax.axhline(1.0, color=ne.INK_SECONDARY, linewidth=0.8, zorder=1)
        ax.text(1.01, 1.0, "τ", transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=ne.INK_SECONDARY, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    legend_ax = axes.flat[n_panels]
    legend_ax.axis("off")
    # The panel above the legend has no panel below it to carry the shared x labels
    axes.flat[n_panels - 3].tick_params(axis="x", labelbottom=True)
    return fig, axes, legend_ax


def _short_stage_label(stage: str) -> str:
    """'Adam 100', ..., 'L-BFGS (final)'."""
    return "L-BFGS (final)" if stage == FINAL_STAGE else f"Adam {int(stage.split('_')[1])}"


def _stage_handles(label=ne.stage_label) -> list:
    return [Line2D([], [], linestyle="", marker=NATURAL_MARKER, markersize=4.2, markerfacecolor=ne.STAGE_COLOURS[s],
                   markeredgecolor="white", markeredgewidth=0.4, label=label(s)) for s in ne.STAGES]


def plot_natural(natural: pd.DataFrame, thresholds: dict, output_dir) -> list:
    """
    fig_natural_score_vs_error: #39's figure in the style of the others. For each score,
    the score / tau of each checkpoint against its velocity error (relL2_uv, in %), colour
    by training stage, marker by k; the line at 1 is tau (flagged above it), the dashed
    line the 10% rule. Shaded: where false alarms (left of the rule, above tau) and missed
    failures (right of it, below tau) would lie.
    """
    errors = 100 * natural["relL2_uv"].to_numpy(dtype=float)
    ratios = {score: natural[score].to_numpy(dtype=float) / thresholds[score] for score in SCORE_NAMES}
    x_range = (errors.min() / 1.6, errors.max() * 1.6)
    y_range = (min(r.min() for r in ratios.values()) / 1.8, max(r.max() for r in ratios.values()) * 1.8)
    rule = 100 * INACCURATE_ABOVE
    inaccurate = natural["inaccurate"].to_numpy(dtype=bool)
    with plt.rc_context(PRINT_RC):
        fig, axes, legend_ax = _ratio_panels(len(SCORE_NAMES), x_range, y_range)
        for i, (ax, score) in enumerate(zip(axes.flat, SCORE_NAMES)):
            _letter(ax, i)
            ax.fill_between([x_range[0], rule], 1.0, y_range[1], color=ne.WASH, linewidth=0, zorder=0)
            ax.fill_between([rule, x_range[1]], y_range[0], 1.0, color=ne.WASH, linewidth=0, zorder=0)
            ax.axvline(rule, color=ne.INK_SECONDARY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
            for stage in ne.STAGES:
                for k, marker in ne.K_MARKERS.items():
                    chosen = ((natural["stage"] == stage) & (natural["k"] == k)).to_numpy()
                    ax.scatter(errors[chosen], ratios[score][chosen], s=9, marker=marker,
                               color=ne.STAGE_COLOURS[stage], edgecolors="white", linewidths=0.3, zorder=3)
            flagged = _flags(natural, score)
            ax.set_title(f"{SCORE_LABELS[score]}\nmissed {int((inaccurate & ~flagged).sum())} of "
                         f"{int(inaccurate.sum())}, false alarms {int((~inaccurate & flagged).sum())} of "
                         f"{int((~inaccurate).sum())}")
        fig.supxlabel("velocity error against the exact solution (%)")
        stages = legend_ax.legend(handles=_stage_handles(), title="Training stage", loc="upper left", frameon=False,
                                  alignment="left", handletextpad=0.4, borderaxespad=0.0)
        legend_ax.add_artist(stages)
        others = [Line2D([], [], linestyle="", marker=marker, markersize=4, markerfacecolor=ne.INK_MUTED,
                         markeredgecolor="white", markeredgewidth=0.4, label=f"k = {k}")
                  for k, marker in ne.K_MARKERS.items()]
        others += [Line2D([], [], color=ne.INK_SECONDARY, linewidth=0.8, label="τ"),
                   Line2D([], [], color=ne.INK_SECONDARY, linewidth=0.8, linestyle=(0, (4, 2)),
                          label=f"{rule:g}% error")]
        legend_ax.legend(handles=others, loc="upper left", bbox_to_anchor=(0.62, 1.0), frameon=False,
                         alignment="left", handletextpad=0.4, borderaxespad=0.0)
        legend_ax.text(0.0, 0.0, f"Flagged above τ; inaccurate above {rule:g}% error.\nShaded: false alarms "
                                 "(accurate, flagged) and\nmissed failures (inaccurate, not flagged).",
                       transform=legend_ax.transAxes, ha="left", va="bottom", color=ne.INK_SECONDARY, fontsize=7)
        paths = save_figure(fig, Path(output_dir), "fig_natural_score_vs_error")
        plt.close(fig)
    return paths


def plot_shared(fields: pd.DataFrame, sizes: pd.DataFrame, natural: pd.DataFrame, thresholds: dict,
                output_dir) -> list:
    """
    fig_score_vs_velocity_error: both experiments on one axis, as what a deviation adds to
    a score against its size. For each score: each damaged field (orange, marker by damage
    type) at the velocity change its damage makes and the rise of its score over its own
    clean field (score_rises()); each checkpoint (blue, by training stage) at its own
    velocity error and its whole score, which is what its error adds over the exact
    solution. Both over tau. A damage type that never changes the velocity (pressure),
    and the fields whose score did not rise (no place on a log axis), are left out; the
    caption counts them.
    """
    rises = score_rises(fields, sizes, thresholds)
    perturbations = sorted(rises["perturbation"].unique(), key=bm._perturbation_order)
    x_damage = 100 * rises["velocity_change"].to_numpy(dtype=float)
    x_natural = 100 * natural["relL2_uv"].to_numpy(dtype=float)
    y_damage = {score: rises[f"rise_{score}"].to_numpy(dtype=float) for score in SCORE_NAMES}
    y_natural = {score: natural[score].to_numpy(dtype=float) / thresholds[score] for score in SCORE_NAMES}
    x_range = (min(x_damage.min(), x_natural.min()) / 1.6, max(x_damage.max(), x_natural.max()) * 1.6)
    y_all = np.concatenate([*(y[y > 0] for y in y_damage.values()), *y_natural.values()])
    y_range = (y_all.min() / 1.8, y_all.max() * 1.8)
    with plt.rc_context(PRINT_RC):
        fig, axes, legend_ax = _ratio_panels(len(SCORE_NAMES), x_range, y_range, ylabel="added score / τ")
        for i, (ax, score) in enumerate(zip(axes.flat, SCORE_NAMES)):
            for perturbation in perturbations:
                chosen = (rises["perturbation"] == perturbation).to_numpy() & (y_damage[score] > 0)
                ax.scatter(x_damage[chosen], y_damage[score][chosen], s=7, marker=DAMAGE_MARKERS[perturbation],
                           color=SYNTHETIC_COLOUR, edgecolors="white", linewidths=0.25, alpha=0.85, zorder=2)
            for stage in ne.STAGES:
                chosen = (natural["stage"] == stage).to_numpy()
                ax.scatter(x_natural[chosen], y_natural[score][chosen], s=9, marker=NATURAL_MARKER,
                           color=ne.STAGE_COLOURS[stage], edgecolors="white", linewidths=0.3, zorder=3)
            ax.set_title(SCORE_LABELS[score])
            _letter(ax, i)
        fig.supxlabel("size of the deviation (% of the exact velocity)")
        # Two legends stacked: the long damage names in one column, the short stage names in two
        damage_handles = [Line2D([], [], linestyle="", marker=DAMAGE_MARKERS[p], markersize=4.2,
                                 markerfacecolor=SYNTHETIC_COLOUR, markeredgecolor="white", markeredgewidth=0.4,
                                 label=DAMAGE_LABELS[p]) for p in perturbations]
        synthetic = legend_ax.legend(handles=damage_handles, title="Synthetic damage", loc="upper left",
                                     frameon=False, alignment="left", handletextpad=0.3, borderaxespad=0.0,
                                     labelspacing=0.3)
        legend_ax.add_artist(synthetic)
        legend_ax.legend(handles=_stage_handles(_short_stage_label), title="Natural errors (training stage)", ncol=2,
                         loc="upper left", bbox_to_anchor=(0.0, 0.62), frameon=False, alignment="left",
                         handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0, labelspacing=0.3)
        note = "Synthetic: damage size, score it adds.\nNatural: own error, whole score."
        legend_ax.text(0.0, 0.0, note, transform=legend_ax.transAxes, ha="left", va="bottom",
                       color=ne.INK_SECONDARY, fontsize=7)
        paths = save_figure(fig, Path(output_dir), "fig_score_vs_velocity_error")
        plt.close(fig)
    return paths


FIELD_CMAP = "Blues"   # One hue, white at zero to dark
FIELD_ROWS = (("change", "change\n$|\\Delta\\mathbf{u}|/U_0$ (pressure: $|\\Delta p|/U_0^2$)"),
              ("momentum", "momentum residual\n$|\\mathbf{R}_m|/(U_0^2 k)$"),
              ("continuity", "continuity residual\n$|R_c|/(U_0 k)$"))


def plot_damage_fields(info: dict, output_dir) -> list:
    """
    fig_damage_fields: for each damage type (column), the change it makes to the exact
    solution and the momentum and continuity residuals it creates (rows), from
    damage_field_maps(); one colour scale per row, white at zero.
    """
    maps = info["maps"]
    perturbations = list(maps)
    ticks, labels = [0, math.pi, 2 * math.pi], ["0", "π", "2π"]
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(len(FIELD_ROWS), len(perturbations), figsize=(TEXT_WIDTH, 4.4), sharex=True,
                                 sharey=True, layout="constrained", squeeze=False)
        for r, (key, label) in enumerate(FIELD_ROWS):
            largest = max(m[key].max() for m in maps.values())
            vanishing = _vanishing(maps, key)
            for c, perturbation in enumerate(perturbations):
                ax = axes[r, c]
                image = ax.imshow(maps[perturbation][key].T, origin="lower", extent=(0, 2 * math.pi, 0, 2 * math.pi),
                                  cmap=FIELD_CMAP, vmin=0, vmax=largest, interpolation="bilinear")
                if perturbation in vanishing:
                    ax.text(math.pi, math.pi, "none", ha="center", va="center", color=ne.INK_SECONDARY, fontsize=7)
                ax.set_xticks(ticks, labels)
                ax.set_yticks(ticks, labels)
                ax.tick_params(length=2)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                if r == 0:
                    ax.set_title(DAMAGE_LABELS[perturbation])
                if r == len(FIELD_ROWS) - 1:
                    ax.set_xlabel("x")
                _letter(ax, r * len(perturbations) + c)
            axes[r, 0].set_ylabel("y")
            bar = fig.colorbar(image, ax=axes[r, :], location="right", shrink=0.92, pad=0.015, aspect=14)
            bar.set_label(label, fontsize=7)
            bar.ax.tick_params(labelsize=7, length=2)
            bar.outline.set_linewidth(0.5)
        paths = save_figure(fig, Path(output_dir), "fig_damage_fields")
        plt.close(fig)
    return paths


def plot_energy(fields: pd.DataFrame, old: pd.DataFrame, cases: list, output_dir) -> list:
    """
    fig_energy_term_vs_u0: the energy term of the clean fields against U0, for the new
    cases (filled) and the old calibration cases (open), marker by k: (a) the normalized
    term S_E bar, with a line of slope 4; (b) the relative energy error. The new fields
    that a score holding the energy term flags are labelled with their case number.
    """
    new, old_points = _energy_points(fields, old, cases)
    both = pd.concat([new, old_points])
    x_range = (both["U0"].min() / 1.15, both["U0"].max() * 1.15)
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), layout="constrained")
        for i, (ax, column, label) in enumerate(((axes[0], "E_bar", "normalized energy term $\\bar S_E$"),
                                                 (axes[1], "relative", "relative energy error (%)"))):
            scale = 100 if column == "relative" else 1
            _style(ax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*x_range)
            ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            values = scale * both[column].to_numpy(dtype=float)
            low, high = values.min() / 1.6, values.max() * 1.6
            ax.set_ylim(low, high)
            if high / low < 1e3:   # Under three decades: plain 1-2-5 ticks, not 6 x 10^-1
                ax.yaxis.set_major_locator(FixedLocator(_log_ticks(low, high)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(_power_label))
            for k, marker in ne.K_MARKERS.items():
                of_k = old_points[old_points["k"] == k]
                ax.scatter(of_k["U0"], scale * of_k[column], s=13, marker=marker, facecolors="white",
                           edgecolors=OLD_COLOUR, linewidths=0.7, zorder=2)
                of_k = new[new["k"] == k]
                ax.scatter(of_k["U0"], scale * of_k[column], s=15, marker=marker, color=K_COLOUR, edgecolors="white",
                           linewidths=0.4, zorder=3)
            for _, row in new[new["energy_alarm"]].iterrows():
                ax.annotate(row["case_id"].split("_")[1], xy=(row["U0"], scale * row[column]), xytext=(4, 2),
                            textcoords="offset points", fontsize=7, color=ne.INK)
            if column == "E_bar":
                # Slope 4 through the middle of the points (geometric means), over the U0 range
                centre_x, centre_y = np.exp(np.log(both["U0"]).mean()), np.exp(np.log(both["E_bar"]).mean())
                xs = np.array(x_range)
                ax.plot(xs, centre_y * (xs / centre_x) ** 4, color=ne.INK_MUTED, linewidth=0.8,
                        linestyle=(0, (4, 2)), zorder=1)
            ax.set_xlabel("velocity scale $U_0$")
            ax.set_ylabel(label)
            _letter(ax, i)
        handles = [Line2D([], [], linestyle="", marker="o", markersize=4.2, markerfacecolor=K_COLOUR,
                          markeredgecolor="white", label="new cases (final models)"),
                   Line2D([], [], linestyle="", marker="o", markersize=4, markerfacecolor="white",
                          markeredgecolor=OLD_COLOUR, label="old cases (calibration)"),
                   *(Line2D([], [], linestyle="", marker=m, markersize=4, markerfacecolor=ne.INK_MUTED,
                            markeredgecolor="white", label=f"k = {k}") for k, m in ne.K_MARKERS.items()),
                   Line2D([], [], color=ne.INK_MUTED, linewidth=0.8, linestyle=(0, (4, 2)),
                          label="∝ U₀⁴")]
        legend = fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False,
                            handlelength=1.6, handletextpad=0.4, columnspacing=1.0)
        paths = save_figure(fig, Path(output_dir), "fig_energy_term_vs_u0", top_legend=legend)
        plt.close(fig)
    return paths


# --- Running it ---

def write_table(table: pd.DataFrame, latex: str, output_dir: Path, stem: str) -> list:
    """Writes <stem>.csv (full precision, "\\n" line ends) and <stem>.tex."""
    csv_path, tex_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False, na_rep="NaN", lineterminator="\n")
    with open(tex_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(latex)
    return [csv_path, tex_path]


def write_text(path: Path, text: str) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def build(benchmark_dir=DEFAULT_BENCHMARK_DIR, errors_csv=DEFAULT_ERRORS_CSV, metadata=DEFAULT_METADATA,
          loss_summary_csv=DEFAULT_LOSS_SUMMARY, calibration_json=DEFAULT_CALIBRATION_JSON,
          source_csv=DEFAULT_SOURCE_CSV, natural_dir=DEFAULT_NATURAL_DIR) -> dict:
    """
    Reads and checks every input, and builds the tables and captions (no file is written).

    Outputs:
        dict: fields, old (the old clean fields), metrics ({breakdown: DataFrame}), calib,
              cases, natural (the checkpoints), sizes (damage_sizes()), tables ({stem:
              DataFrame}) and captions ({stem: LaTeX text}).
    """
    calib = frozen_calibration.load(calibration_json)
    thresholds = calib["thresholds"]
    fields = read_fields(benchmark_dir, calib, metadata)
    metrics = {name: read_metrics(benchmark_dir, name) for name in ("overall", "by_k", "by_type", "by_type_eps")}
    check_metrics(fields, metrics)
    old = read_old_clean_fields(source_csv, calib)
    cases, errors, loss = read_case_inputs(metadata, errors_csv, loss_summary_csv)
    natural = read_natural(natural_dir, calib, errors, cases, fields)

    by_k = recall_by_type_eps_k(fields)
    cases_table = case_table(cases, errors, loss, fields)
    summary = summary_table(metrics, natural, read_natural_stage_table(natural_dir), thresholds)
    sizes = damage_sizes(cases, sorted(fields.loc[fields["label"] != "clean", "epsilon"].unique()))
    size_table = damage_size_table(sizes)
    terms = term_auc_table(fields, natural)
    shown_case = field_case(cases)
    damage_maps = damage_field_maps(shown_case, FIELD_EPSILON)
    type_eps = metrics["by_type_eps"]
    captions = {
        "fig_recall_vs_eps": caption_recall_vs_eps(type_eps, fields, thresholds),
        "fig_recall_vs_eps_new_detections": caption_recall_new(type_eps, fields, thresholds),
        "fig_recall_vs_eps_by_k": caption_recall_by_k(by_k, metrics, fields),
        "fig_roc": caption_roc(metrics["overall"], metrics, fields),
        "fig_clean_scores_vs_tau": caption_clean_scores(fields, old, thresholds, cases),
        "fig_natural_score_vs_error": caption_natural(natural),
        "fig_score_vs_velocity_error": caption_shared(fields, sizes, natural, thresholds),
        "fig_damage_fields": caption_damage_fields(damage_maps, shown_case, FIELD_EPSILON),
        "fig_energy_term_vs_u0": caption_energy(fields, old, cases),
        "table_recall_by_type_eps_k": caption_recall_by_k_table(by_k),
        "table_cases": caption_cases(cases_table),
        "table_synthetic_vs_natural": caption_summary(summary, natural, fields, type_eps),
        "table_damage_size": caption_damage_size(size_table, len(cases)),
        "table_term_auc": caption_term_auc(terms, natural),
    }
    tables = {"table_recall_by_type_eps_k": by_k, "table_cases": cases_table,
              "table_synthetic_vs_natural": summary, "table_damage_size": size_table, "table_term_auc": terms}
    return {"fields": fields, "old": old, "metrics": metrics, "calib": calib, "cases": cases, "natural": natural,
            "sizes": sizes, "damage_maps": damage_maps, "tables": tables, "captions": captions}


def run(benchmark_dir=DEFAULT_BENCHMARK_DIR, errors_csv=DEFAULT_ERRORS_CSV, metadata=DEFAULT_METADATA,
        loss_summary_csv=DEFAULT_LOSS_SUMMARY, calibration_json=DEFAULT_CALIBRATION_JSON,
        source_csv=DEFAULT_SOURCE_CSV, output_dir=DEFAULT_OUTPUT_DIR, natural_dir=DEFAULT_NATURAL_DIR) -> tuple:
    """
    Builds everything (build()) and writes it to output_dir. See the module docstring.

    Outputs:
        (built, written): the output of build(), and the list of files written.
    """
    built = build(benchmark_dir, errors_csv, metadata, loss_summary_csv, calibration_json, source_csv, natural_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields, metrics, captions = built["fields"], built["metrics"], built["captions"]
    thresholds = built["calib"]["thresholds"]
    tables = built["tables"]

    written = []
    written += plot_recall(metrics["by_type_eps"], "recall", output_dir, "fig_recall_vs_eps")
    written += plot_recall(metrics["by_type_eps"], "recall_new", output_dir, "fig_recall_vs_eps_new_detections")
    written += plot_recall_by_k(tables["table_recall_by_type_eps_k"], output_dir)
    written += plot_roc(fields, metrics["overall"], output_dir)
    written += plot_clean_scores(fields, built["old"], thresholds, output_dir)
    written += plot_natural(built["natural"], thresholds, output_dir)
    written += plot_shared(fields, built["sizes"], built["natural"], thresholds, output_dir)
    written += plot_damage_fields(built["damage_maps"], output_dir)
    written += plot_energy(fields, built["old"], built["cases"], output_dir)
    for stem in FIGURES:
        written.append(write_text(output_dir / f"{stem}.tex",
                                  figure_tex(stem, captions[stem], wide=stem not in NARROW_FIGURES)))
    for stem in TABLES:
        written += write_table(tables[stem], table_latex(stem, tables[stem], captions[stem]), output_dir, stem)
    return built, written


def print_summary(built: dict, written: list) -> None:
    """Prints what was written and the headline numbers."""
    fields = built["fields"]
    cases = built["tables"]["table_cases"]
    summary = built["tables"]["table_synthetic_vs_natural"]
    natural = built["natural"]
    print(f"Read {len(fields)} fields and {len(natural)} checkpoints of {fields['case_id'].nunique()} cases; every "
          "input table agrees")
    for experiment in (SYNTHETIC, NATURAL):
        print(f"  {experiment}:")
        for _, row in summary[(summary["experiment"] == experiment) & (summary["k"] == ALL)].iterrows():
            print(f"    {row['score']:28s} AUC {_range_text(row, 'auc')}  recall {bm._number(row['recall'])}  "
                  f"false alarms {int(row['n_false_alarms'])}/{int(row['n_negative'])}")
    print(f"  {int(cases['passed_acceptance'].sum())}/{len(cases)} runs passed; final velocity error "
          f"{_percent(cases['relL2_uv'].min())}-{_percent(cases['relL2_uv'].max())}% (test grid)")
    print(f"Wrote {len(written)} files")


def parse_args(argv=None):
    """
    Parses the command line.

    Outputs:
        argparse.Namespace: benchmark_dir, natural_dir, errors, metadata, loss_summary,
                            calibration, source, output_dir (str).
    """
    parser = argparse.ArgumentParser(description="Write the final figures and the case table (#36) from the saved "
                                                 "results of the benchmark, the natural errors, the accuracy audit "
                                                 "and the training.")
    parser.add_argument("--benchmark_dir", type=str, default=str(DEFAULT_BENCHMARK_DIR),
                        help="Folder with #35's tables (default: results/benchmark).")
    parser.add_argument("--natural_dir", type=str, default=str(DEFAULT_NATURAL_DIR),
                        help="Folder with #39's tables (default: results/natural_errors).")
    parser.add_argument("--errors", type=str, default=str(DEFAULT_ERRORS_CSV),
                        help="The true errors of #37 (default: results/accuracy/reference_errors_confirmation.csv).")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="The case file (default: cases/confirmation_cases.json).")
    parser.add_argument("--loss_summary", type=str, default=str(DEFAULT_LOSS_SUMMARY),
                        help="The training summary of #40 (default: models/confirmation/loss_summary.csv).")
    parser.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_JSON),
                        help="The frozen calibration (default: calibration/frozen_calibration.json).")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE_CSV),
                        help="The old scores the calibration was fitted on "
                             "(default: calibration/source_phs_components_raw.csv).")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Folder to write the figures and tables into (default: results/figures).")
    return parser.parse_args(argv)


def main(argv=None):
    """Writes every figure and table and prints a summary."""
    args = parse_args(argv)
    built, written = run(args.benchmark_dir, args.errors, args.metadata, args.loss_summary, args.calibration,
                         args.source, args.output_dir, natural_dir=args.natural_dir)
    print_summary(built, written)


if __name__ == "__main__":
    enable_utf8_output()
    main()
