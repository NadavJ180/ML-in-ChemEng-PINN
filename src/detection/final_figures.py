"""
The final figures and the case table of the final iteration (#36).

Barak asks for "figures in PDF/PNG, tables in CSV/LaTeX". This script turns the
saved results of the perturbation benchmark (#35) on the 30 new cases into the
paper's figures, and writes the table of the 30 cases. It reads only saved
tables, never a model, and fits nothing:

    results/benchmark/confirmation_scores_raw.csv   every field: its scores and flags (#35)
    results/benchmark/metrics_overall.csv           the AUC of each score and its range (#35)
    results/benchmark/metrics_by_k.csv              the same by k, for the captions (#35)
    results/benchmark/metrics_by_type.csv           the same by damage type, for the captions (#35)
    results/benchmark/metrics_by_type_eps.csv       recall by damage type x strength, exact ranges (#35)
    results/accuracy/reference_errors_confirmation.csv   the true velocity error of each model (#37)
    cases/confirmation_cases.json                   the cases and their seeds (#44)
    models/confirmation/loss_summary.csv            the training acceptance test of each run (#40)
    calibration/frozen_calibration.json             the alarm level tau of each score (#45)
    calibration/source_phs_components_raw.csv       the 30 old clean fields tau was fitted on (#45)

Before drawing anything it checks that these tables agree: rescoring the raw
components with the frozen calibration must give back every normalized
component, score and flag of the benchmark bit for bit; the benchmark must hold
every case of the case file, each with its 31 fields; #35's metrics tables must
match the flags (counts) and the scores (AUC); and the pass/fail of each run must
be the same in loss_summary.csv, the benchmark and the accuracy audit, and must
follow from its acceptance metrics. A mismatch stops the run, since a figure
built from tables that disagree would be wrong without anyone noticing.

Outputs, in results/figures/. Each figure as .pdf and .png (300 dpi), sized for
a two-column page (7.16 in wide, one column 3.5 in), plus a .tex file holding
its figure environment and caption; each table as .csv (full precision) and .tex:
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
    table_recall_by_type_eps_k         the numbers behind fig_recall_vs_eps_by_k, computed from
                                       the flags of confirmation_scores_raw.csv (#35's tables
                                       have no type x strength x k breakdown)
    table_cases                        one row per new case: k, U0, Re, phases, the three seeds,
                                       acceptance pass/fail and the final velocity errors
The captions are written from the numbers, so they can't drift from the figures.
A paper includes a figure with \\input{fig_roc.tex}, with results/figures/ in its
\\graphicspath. Two runs on the same machine give the same files, byte for byte.

Usage:
    python src/detection/final_figures.py
    python src/detection/final_figures.py --benchmark_dir results/benchmark --output_dir results/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.hallucinations.perturbations import PERTURBATION_NAMES
from src.models.accuracy_audit import read_cases
from src.models.checkpoints import FINAL_STAGE
from src.utils.console import enable_utf8_output

DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmark"
DEFAULT_ERRORS_CSV = PROJECT_ROOT / "results" / "accuracy" / "reference_errors_confirmation.csv"
DEFAULT_METADATA = PROJECT_ROOT / "cases" / "confirmation_cases.json"
DEFAULT_LOSS_SUMMARY = PROJECT_ROOT / "models" / "confirmation" / "loss_summary.csv"
DEFAULT_CALIBRATION_JSON = frozen_calibration.DEFAULT_CALIBRATION_JSON
DEFAULT_SOURCE_CSV = frozen_calibration.DEFAULT_SOURCE_CSV
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

FIELDS_CSV = "confirmation_scores_raw.csv"
SCORE_NAMES = bm.SCORE_NAMES
BAR_NAMES = bm.BAR_NAMES
FLAG_NAMES = tuple(f"flagged_{name}" for name in SCORE_NAMES)
CLEAN = bm.CLEAN
PHS = "Score3_PHS_full"

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
DAMAGE_LABELS = {name: name.replace("_", " ") for name in PERTURBATION_NAMES}
K_COLOUR = SCORE_COLOURS[SCORE_NAMES[0]]   # The new clean fields (one series: the first slot)
OLD_COLOUR = ne.INK_MUTED                  # The old calibration fields, a grey reference

TEXT_WIDTH = 7.16    # in: a two-column page, as publication_figures.py uses
COLUMN_WIDTH = 3.5   # in: one column
PNG_DPI = 300
# natural_errors.py's style (#39: font, ink, grid and axis colours), at print size
PRINT_RC = {
    **ne.FIGURE_RC,
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.titlesize": 7.5, "figure.labelsize": 7.5,
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
           "fig_clean_scores_vs_tau")
TABLES = ("table_recall_by_type_eps_k", "table_cases")

RECALL_BY_K_COLUMNS = ("score", "k", "perturbation", "epsilon", "n_cases", "n_damaged", "n_damaged_flagged",
                       "recall", "recall_low", "recall_high", "n_preflagged_cases", "n_damaged_new",
                       "n_damaged_new_flagged", "recall_new", "recall_new_low", "recall_new_high")
CASE_COLUMNS = ("case_id", "k", "U0", "Re", "phi_x", "phi_y", "seed_points", "seed_train", "seed_scoring",
                "passed_acceptance", "rel_l2_eval_grid", "relL2_uv")


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
    settings, its seeds, whether its run passed the training acceptance test, and the
    final model's relative L2 velocity error on the acceptance test's 64 x 64 x 20
    evaluation grid (rel_l2_eval_grid, from loss_summary.csv) and on the independent
    64 x 64 x 11 grid of the accuracy audit (relL2_uv, #37; PROTOCOL.md section 6: the
    main number).

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
        rows.append({"case_id": case_id, "k": case["k"], "U0": case["U0"], "Re": case["Re"],
                     "phi_x": case["phi_x"], "phi_y": case["phi_y"], "seed_points": case["seed_points"],
                     "seed_train": case["seed_train"], "seed_scoring": case["seed_scoring"],
                     "passed_acceptance": passed, "rel_l2_eval_grid": float(summary["rel_l2"]),
                     "relL2_uv": float(final.loc[case_id, "relL2_uv"])})
    return pd.DataFrame(rows)[list(CASE_COLUMNS)]


# --- LaTeX of the tables ---

def _percent(value) -> str:
    """0.001534 -> '0.15'."""
    return f"{100 * value:.2f}"


def case_table_latex(table: pd.DataFrame, caption: str) -> str:
    """
    The case table as a full-width LaTeX table (booktabs): one row per case, numbers
    rounded, errors in %, the seeds and the errors under grouped headers.
    """
    lines = [
        "% Written by src/detection/final_figures.py; needs \\usepackage{booktabs}.",
        "\\begin{table*}[tb]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:cases}",
        "\\begin{tabular}{lrrrrrrrrcrr}",
        "\\toprule",
        " & & & & & & \\multicolumn{3}{c}{Seeds} & & \\multicolumn{2}{c}{Velocity error (\\%)} \\\\",
        "\\cmidrule(lr){7-9} \\cmidrule(lr){11-12}",
        "Case & $k$ & $U_0$ & Re & $\\phi_x$ & $\\phi_y$ & points & training & scoring & Passed & eval.\\ grid & "
        "test grid \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        cells = [bm._tex_escape(row.case_id), str(row.k), f"{row.U0:.3f}", f"{row.Re:.1f}", f"{row.phi_x:.3f}",
                 f"{row.phi_y:.3f}", str(row.seed_points), str(row.seed_train), str(row.seed_scoring),
                 "yes" if row.passed_acceptance else "no", _percent(row.rel_l2_eval_grid), _percent(row.relL2_uv)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def recall_by_k_latex(table: pd.DataFrame, caption: str) -> str:
    """The type x strength x k recall as a LaTeX longtable: one row per score x k x type, one column per strength."""
    epsilons = sorted(table["epsilon"].unique())
    rows = []
    for (score, k, perturbation), group in table.groupby(["score", "k", "perturbation"], sort=False):
        group = group.set_index("epsilon")
        cells = {"Score": bm._tex_escape(score.split("_")[0]), "$k$": str(k), "Damage": bm._tex_escape(perturbation)}
        for epsilon in epsilons:
            cells[f"$\\varepsilon={epsilon:g}$"] = (f"{int(group.loc[epsilon, 'n_damaged_flagged'])}/"
                                                   f"{int(group.loc[epsilon, 'n_damaged'])}")
        rows.append(cells)
    return pd.DataFrame(rows).to_latex(index=False, escape=False, longtable=True, caption=caption,
                                       label="tab:recall_by_type_eps_k")


# --- Facts for the captions ---

def _tex_cases(case_ids) -> str:
    """'case\\_34 and case\\_45'."""
    names = [bm._tex_escape(case_id) for case_id in case_ids]
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
            "exact (Clopper--Pearson) 95\\% ranges, drawn once in grey where several scores have the same count "
            "and so the same range. "
            + recall_sentence(type_eps, "recall", "n_damaged_flagged", "n_damaged")
            + f" At $\\varepsilon = {facts['epsilon']:g}$ nearly all flagged fields belong to cases whose clean "
              f"field the score already flags ({shares}{other_text}); "
              "Fig.~\\ref{fig:recall_vs_eps_new_detections} leaves those cases out.")


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
            f"score: {counts}; bands: exact 95\\% ranges." + smallest + " "
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
                     f"case-bootstrap ranges have no width: {_short(row['score'])}'s recall for $k = {row['k']}$ is "
                     f"{_range_text(row, 'recall')}, because every $k = {row['k']}$ case has {count} of its "
                     f"{per_case} damaged fields flagged, so no resample of cases can change the rate"
                     + (f" (also {others})" if others else "") + ". The exact ranges per strength shown here don't "
                     "have this problem.")
    ks = sorted(by_k["k"].unique())
    k_text = ", ".join(str(k) for k in ks)
    n_types = by_k["perturbation"].nunique()
    falls = all((np.diff([pooled.loc[(score, k), "n_damaged_flagged"] for k in ks]) < 0).all() for score in SCORE_NAMES)
    trend = "Recall falls with $k$ for every score" if falls else "Recall depends on $k$"
    return (f"Recall as in Fig.~\\ref{{fig:recall_vs_eps}}, split by $k$ (rows): each point is out of the "
            f"{int(by_k['n_damaged'].iloc[0])} cases with that $k$, with exact 95\\% ranges "
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
            "[95\\% range: case bootstrap, 2000 resamples of whole cases within each $k$]. Markers: the frozen "
            f"$\\tau$ of each score, at its false-alarm rate on the {n_clean} clean fields and its recall")
    if same:
        a, b = same[0]
        text += (f" ({_short(a)} and {_short(b)} nearly coincide: {points[a][0]} false alarms each, "
                 f"{points[a][1]} and {points[b][1]} damaged fields flagged)")
    text += (f"; the false-alarm axis moves in steps of 1/{n_clean}. The ranges lean above their estimates "
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
    return text


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
        listed = "none" if flagged.empty else _and([f"{bm._tex_escape(case_id)} ($k = {row['k']}$)"
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
        text += "."
    return text


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
    return (f"The {len(table)} new cases, pre-registered before training: wavenumber $k$, velocity scale $U_0$, "
            "Reynolds number Re, phases $\\phi_x$, $\\phi_y$, and the seeds of case $n$: of its training and "
            f"evaluation points{_seed_rule(table, 'seed_points')}, of its weights and "
            f"mini-batches{_seed_rule(table, 'seed_train')} and of its scoring points"
            f"{_seed_rule(table, 'seed_scoring')}. Passed: the final model passes the training acceptance test "
            f"(velocity relative $L_2$ error below {ACCEPT_REL_L2_BELOW:g} on the $64\\times64\\times20$ evaluation "
            f"grid, MSE($R_c$) below $10^{{{int(np.log10(ACCEPT_MSE_RC_BELOW))}}}$ and MSE($R_m$) below "
            f"$10^{{{int(np.log10(ACCEPT_MSE_RM_BELOW))}}}$). Velocity error: the final model's relative $L_2$ "
            "velocity error on that evaluation grid and on an independent $64\\times64\\times11$ test grid. "
            f"{passed}; test-grid errors {ranges}.")


def caption_recall_by_k_table(table: pd.DataFrame) -> str:
    n_cases = int(table["n_damaged"].iloc[0])
    return (f"Damaged fields flagged by each score, by $k$, damage type and strength, out of the {n_cases} cases "
            "with that $k$ (frozen calibration, flagged above $\\tau$). The .csv adds the exact 95\\% ranges and the "
            "counts over the cases whose clean field the score doesn't flag.")


# --- Figures ---

def save_figure(fig, output_dir: Path, stem: str) -> list:
    """Saves a figure as <stem>.pdf and <stem>.png; the PDF without a creation date, so a rerun gives the same file."""
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None})
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


def _score_handles(bands: bool = False) -> list:
    """The legend entries of the five scores, and with bands, of the grey band of a shared range."""
    handles = [Line2D([], [], color=SCORE_COLOURS[s], linewidth=LINE_WIDTH, marker=SCORE_MARKERS[s],
                      markersize=MARKER_SIZE + 0.6, markeredgecolor="white", markeredgewidth=0.5, label=SCORE_LABELS[s])
               for s in SCORE_NAMES]
    if bands:
        handles.append(Patch(facecolor=SHARED_BAND_COLOUR, alpha=SHARED_BAND_ALPHA, linewidth=0,
                             label="shared 95% range"))
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
        for ax, perturbation in zip(axes, perturbations):
            _style(ax)
            _recall_lines(ax, table, perturbation, rate)
            _epsilon_axis(ax, epsilons)
            ax.set_ylim(-0.04, 1.04)
            ax.set_title(DAMAGE_LABELS[perturbation])
        axes[0].set_ylabel("recall" if rate == "recall" else "recall, new detections")
        fig.supxlabel("damage strength \u03b5")
        fig.legend(handles=_score_handles(bands=True), loc="outside upper center", ncol=len(SCORE_NAMES) + 1,
                   frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0)
        paths = save_figure(fig, Path(output_dir), stem)
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
            axes[i, 0].set_ylabel(f"k = {k}\nrecall")
        fig.supxlabel("damage strength \u03b5")
        fig.legend(handles=_score_handles(bands=True), loc="outside upper center", ncol=len(SCORE_NAMES) + 1,
                   frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0)
        paths = save_figure(fig, Path(output_dir), "fig_recall_vs_eps_by_k")
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
        inset.tick_params(labelsize=6)
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
        ax.legend(handles=handles + blanks, ncol=2, title="AUC [95% range]; open marker: the frozen \u03c4",
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
        for ax, score in zip(axes.flat, SCORE_NAMES):
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
                                    ha="left", va="center", fontsize=6.5, color=ne.INK,
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
          source_csv=DEFAULT_SOURCE_CSV) -> dict:
    """
    Reads and checks every input, and builds the tables and captions (no file is written).

    Outputs:
        dict: fields, old (the old clean fields), metrics ({breakdown: DataFrame}), calib,
              cases, tables ({stem: DataFrame}) and captions ({stem: LaTeX text}).
    """
    calib = frozen_calibration.load(calibration_json)
    thresholds = calib["thresholds"]
    fields = read_fields(benchmark_dir, calib, metadata)
    metrics = {name: read_metrics(benchmark_dir, name) for name in ("overall", "by_k", "by_type", "by_type_eps")}
    check_metrics(fields, metrics)
    old = read_old_clean_fields(source_csv, calib)
    cases, errors, loss = read_case_inputs(metadata, errors_csv, loss_summary_csv)

    by_k = recall_by_type_eps_k(fields)
    cases_table = case_table(cases, errors, loss, fields)
    type_eps = metrics["by_type_eps"]
    captions = {
        "fig_recall_vs_eps": caption_recall_vs_eps(type_eps, fields, thresholds),
        "fig_recall_vs_eps_new_detections": caption_recall_new(type_eps, fields, thresholds),
        "fig_recall_vs_eps_by_k": caption_recall_by_k(by_k, metrics, fields),
        "fig_roc": caption_roc(metrics["overall"], metrics, fields),
        "fig_clean_scores_vs_tau": caption_clean_scores(fields, old, thresholds, cases),
        "table_recall_by_type_eps_k": caption_recall_by_k_table(by_k),
        "table_cases": caption_cases(cases_table),
    }
    return {"fields": fields, "old": old, "metrics": metrics, "calib": calib, "cases": cases,
            "tables": {"table_recall_by_type_eps_k": by_k, "table_cases": cases_table}, "captions": captions}


def run(benchmark_dir=DEFAULT_BENCHMARK_DIR, errors_csv=DEFAULT_ERRORS_CSV, metadata=DEFAULT_METADATA,
        loss_summary_csv=DEFAULT_LOSS_SUMMARY, calibration_json=DEFAULT_CALIBRATION_JSON,
        source_csv=DEFAULT_SOURCE_CSV, output_dir=DEFAULT_OUTPUT_DIR) -> tuple:
    """
    Builds everything (build()) and writes it to output_dir. See the module docstring.

    Outputs:
        (built, written): the output of build(), and the list of files written.
    """
    built = build(benchmark_dir, errors_csv, metadata, loss_summary_csv, calibration_json, source_csv)
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
    for stem in FIGURES:
        written.append(write_text(output_dir / f"{stem}.tex", figure_tex(stem, captions[stem], wide=stem != "fig_roc")))
    written += write_table(tables["table_recall_by_type_eps_k"],
                           recall_by_k_latex(tables["table_recall_by_type_eps_k"], captions["table_recall_by_type_eps_k"]),
                           output_dir, "table_recall_by_type_eps_k")
    written += write_table(tables["table_cases"], case_table_latex(tables["table_cases"], captions["table_cases"]),
                           output_dir, "table_cases")
    return built, written


def print_summary(built: dict, written: list) -> None:
    """Prints what was written and the headline numbers."""
    fields = built["fields"]
    overall = built["metrics"]["overall"]
    cases = built["tables"]["table_cases"]
    print(f"Read {len(fields)} fields of {fields['case_id'].nunique()} cases; every input table agrees")
    for _, row in overall.iterrows():
        print(f"  {row['score']:28s} AUC {_range_text(row, 'auc')}  recall {bm._number(row['recall'])}  "
              f"false alarms {int(row['n_false_alarms'])}/{int(row['n_clean'])}")
    print(f"  {int(cases['passed_acceptance'].sum())}/{len(cases)} runs passed; final velocity error "
          f"{_percent(cases['relL2_uv'].min())}-{_percent(cases['relL2_uv'].max())}% (test grid)")
    print(f"Wrote {len(written)} files")


def parse_args(argv=None):
    """
    Parses the command line.

    Outputs:
        argparse.Namespace: benchmark_dir, errors, metadata, loss_summary, calibration,
                            source, output_dir (str).
    """
    parser = argparse.ArgumentParser(description="Write the final figures and the case table (#36) from the saved "
                                                 "results of the benchmark, the accuracy audit and the training.")
    parser.add_argument("--benchmark_dir", type=str, default=str(DEFAULT_BENCHMARK_DIR),
                        help="Folder with #35's tables (default: results/benchmark).")
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
                         args.source, args.output_dir)
    print_summary(built, written)


if __name__ == "__main__":
    enable_utf8_output()
    main()
