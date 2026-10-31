"""
Perturbation benchmark on the 30 new cases (#35): the result tables.

Reads the scores of the benchmark's fields, written by

    python src/detection/evaluate_phs.py --calibration calibration/frozen_calibration.json ...

(30 cases x 31 fields: the clean field of each case's lbfgs_final model, and the
5 damage types at the 6 strengths), the frozen calibration, the case file
(cases/confirmation_cases.json: every case in it must be scored, since no case
is dropped, PROTOCOL.md section 1), and each case's lbfgs_final.json sidecar,
and writes every table of the benchmark into
results/benchmark/. It loads no model and fits nothing. A field is flagged by a
score when the score is above that score's frozen tau (PROTOCOL.md section 5);
the metrics follow PROTOCOL.md section 7.

Metrics, for each of the five scores. A damaged field is a positive, a clean
field a negative, and a flagged field a predicted positive:
    auc          ROC-AUC of the score, damaged vs clean fields: the chance that a
                 damaged field scores above a clean one, ties counting one half
                 (sklearn's roc_auc_score, computed here from ranks)
    precision    flagged damaged fields / flagged fields
    recall       flagged damaged fields / damaged fields
    recall_new   recall over the cases whose clean field the score does not flag.
                 A case whose clean field is already above tau ("already flagged",
                 preflagged_<score>) raises a false alarm before any damage, so
                 flagging its damaged fields is no new detection.
    f1           2 TP / (2 TP + FP + FN)
    fpr          flagged clean fields / clean fields: the false-alarm rate
A metric that is undefined (precision when nothing is flagged) is NaN.

Each is reported overall, by k, by damage type, and by damage type x strength. A
group holds its damaged fields and the clean fields of its cases. The clean
fields are the negatives of every damage type, so the by-type tables repeat the
overall false-alarm rate.

Ranges (src/detection/grouped_eval.py, #41), 95%:
    case_bootstrap   auc, precision and f1, and recall / recall_new over several
                     fields per case: 2000 resamples of whole cases within each k,
                     seed 0. <metric>_n_boot says how many resamples were used
                     (a resample where the metric is undefined is left out).
    clopper_pearson  counts with one field per case: the false alarms (fpr), and
                     recall / recall_new at one damage type and strength.

Failed runs stay in. passed_acceptance is read from each case's lbfgs_final.json
sidecar (status PASSED or FAILED, PROTOCOL.md section 2). The per-field tables
carry it, every metrics table counts the failed runs of each group
(n_failed_runs), and if any run failed, metrics_passed_only repeats the overall
and by-k metrics on the runs that passed only.

Outputs, in results/benchmark/, each table as .csv and .tex (the .tex files need
the booktabs and longtable packages):
    confirmation_scores_raw     every field: case_id, k, perturbation, epsilon,
                                label, the raw and normalized components, the 5
                                scores, the 5 flags, passed_acceptance and
                                preflagged_<score>
    metrics_overall             every metric of every score, with its range
    metrics_by_k                the same, by k
    metrics_by_type             the same, by damage type
    metrics_by_type_eps         the same, by damage type x strength
    metrics_passed_only         the overall and by-k metrics of the runs that
                                passed; written only if any run failed
    clean_false_positives       each score's false alarms, overall (out of 30)
                                and per k (out of 10), with exact ranges
    preflagged_cases            the already-flagged cases of each score
    false_alarms                every clean field above tau, one row per field x score
    missed_detections           every damaged field not above tau, one row per field x score

Usage:
    python src/detection/benchmark_metrics.py
    python src/detection/benchmark_metrics.py --scores data/confirmation/phs_scores/phs_components_frozen.csv \\
        --calibration calibration/frozen_calibration.json --metadata cases/confirmation_cases.json \\
        --models_dir models/confirmation --output_dir results/benchmark
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detection import frozen_calibration
from src.detection.grouped_eval import case_bootstrap, clopper_pearson
from src.detection.phs import PHS_COMPONENT_NAMES, SCORE_DEFINITIONS
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.hallucinations.perturbations import PERTURBATION_NAMES
from src.models.checkpoints import FINAL_STAGE
from src.utils.console import enable_utf8_output

DEFAULT_SCORES_CSV = PROJECT_ROOT / "data" / "confirmation" / "phs_scores" / "phs_components_frozen.csv"
DEFAULT_CALIBRATION_JSON = PROJECT_ROOT / "calibration" / "frozen_calibration.json"
DEFAULT_METADATA = PROJECT_ROOT / "cases" / "confirmation_cases.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "confirmation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark"

N_BOOT = 2000        # Resamples of case_bootstrap (PROTOCOL.md section 7)
BOOT_SEED = 0
CLEAN = "none"       # The clean field's perturbation in the hallucination index
SCORE_NAMES = tuple(SCORE_DEFINITIONS)
BAR_NAMES = tuple(f"{component}_bar" for component in PHS_COMPONENT_NAMES)

PASSED_ONLY = "metrics_passed_only"


# --- Reading the inputs ---

def load_scores(scores_csv, calib: dict) -> pd.DataFrame:
    """
    Reads evaluate_phs.py's scored table exactly (float_precision="round_trip")
    and checks it against the frozen calibration: rescoring its raw components
    with frozen_calibration.apply() must give back its normalized components,
    scores and flags bit for bit. So the tables can't come from fields scored
    with another calibration, or from a table edited by hand.

    Inputs:
        scores_csv (str | Path): phs_components_frozen.csv from evaluate_phs.py --calibration.
        calib (dict): Output of frozen_calibration.load().

    Outputs:
        pd.DataFrame: The table, with perturbation_type renamed perturbation.
    """
    scores = pd.read_csv(scores_csv, float_precision="round_trip")
    required = ["case_id", "k", "perturbation_type", "epsilon", "label", *PHS_COMPONENT_NAMES, *BAR_NAMES,
                *SCORE_NAMES, *(f"flagged_{name}" for name in SCORE_NAMES)]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"{scores_csv} has no column(s) {missing}. Score with "
                         "evaluate_phs.py --calibration calibration/frozen_calibration.json.")

    rescored = frozen_calibration.apply(scores[["case_id", *PHS_COMPONENT_NAMES]], calib)
    for column in [*BAR_NAMES, *SCORE_NAMES, *(f"flagged_{name}" for name in SCORE_NAMES)]:
        if not np.array_equal(rescored[column].to_numpy(), scores[column].to_numpy()):
            raise ValueError(f"Column {column} of {scores_csv} doesn't match the frozen calibration "
                             f"{calib.get('source_csv', '')}: the fields were scored with another calibration, "
                             "or the file was changed")
    scores = scores.rename(columns={"perturbation_type": "perturbation"})
    check_layout(scores)
    return scores


def check_layout(fields: pd.DataFrame) -> None:
    """
    Refuses a table the metrics would silently get wrong: each case needs exactly
    one clean field (perturbation "none"), the same damaged (perturbation, epsilon)
    fields as every other case, each once, and a single k.

    Inputs:
        fields (pd.DataFrame): One row per field, with case_id, k, perturbation, epsilon, label.
    """
    labels = set(fields["label"])
    if not labels <= {"clean", "hallucinated"}:
        raise ValueError(f"Unknown labels {sorted(labels - {'clean', 'hallucinated'})}")
    clean = fields["label"] == "clean"
    if not (clean == (fields["perturbation"] == CLEAN)).all():
        raise ValueError(f"A field is labeled clean if and only if its perturbation is '{CLEAN}'")
    n_clean = clean.groupby(fields["case_id"]).sum()
    if (n_clean != 1).any():
        raise ValueError(f"Each case needs exactly one clean field: {n_clean[n_clean != 1].to_dict()}")
    duplicated = fields.duplicated(["case_id", "perturbation", "epsilon"])
    if duplicated.any():
        raise ValueError(f"Fields scored twice: {fields.loc[duplicated, ['case_id', 'perturbation', 'epsilon']].values.tolist()}")
    # With no field twice, "every damaged field is in every case" means every case has the same damaged fields
    damaged = fields[~clean]
    cases_per_field = damaged.groupby(["perturbation", "epsilon"])["case_id"].nunique()
    if damaged.empty or (cases_per_field != fields["case_id"].nunique()).any():
        raise ValueError("Every case needs the same damaged fields (perturbation, epsilon), and there must be some")
    n_k = fields.groupby("case_id")["k"].nunique()
    if (n_k != 1).any():
        raise ValueError(f"Cases with several k values: {n_k.index[n_k != 1].tolist()}")


def check_cases(fields: pd.DataFrame, cases: dict) -> None:
    """
    Refuses scores that don't cover exactly the cases of the case file, each with
    its k. No case is dropped or added (PROTOCOL.md section 1), so scores from a
    --case_id run of evaluate_phs.py, or from another case file, stop here
    instead of giving tables of fewer cases.

    Inputs:
        fields (pd.DataFrame): One row per field, with case_id and k.
        cases (dict): {case_id: case} of the case file, from load_case_metadata().
    """
    scored = fields.groupby("case_id")["k"].first()
    missing = sorted(set(cases) - set(scored.index))
    extra = sorted(set(scored.index) - set(cases))
    if missing or extra:
        raise ValueError(f"The scores must cover every case of the case file: missing {missing}, "
                         f"not in the case file {extra}")
    wrong_k = [case_id for case_id, k in scored.items() if k != cases[case_id]["k"]]
    if wrong_k:
        raise ValueError(f"Cases whose k differs from the case file: {wrong_k}")


def read_passed_acceptance(models_dir, cases: pd.Series) -> dict:
    """
    Reads whether each case's run passed the training acceptance test, from the
    status of its lbfgs_final.json sidecar (PASSED or FAILED; PROTOCOL.md section 2).

    Inputs:
        models_dir (str | Path): Folder with one {case_id}/ subfolder per case (models/confirmation/).
        cases (pd.Series): k of each case, indexed by case_id; checked against the sidecar.

    Outputs:
        dict: {case_id: True if the run passed}.
    """
    passed = {}
    for case_id, k in cases.items():
        sidecar_path = Path(models_dir) / case_id / f"{FINAL_STAGE}.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"No sidecar {sidecar_path}: pass --models_dir with the models that were scored")
        with open(sidecar_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        status = info.get("status")
        if status not in ("PASSED", "FAILED"):
            raise ValueError(f"{sidecar_path} has status {status!r}, expected PASSED or FAILED")
        metrics_passed = (info.get("metrics") or {}).get("passed")
        if metrics_passed is not None and metrics_passed != (status == "PASSED"):
            raise ValueError(f"{sidecar_path}: status {status} disagrees with metrics.passed={metrics_passed}")
        if info.get("case_id", case_id) != case_id or ("k" in info and info["k"] != k):
            raise ValueError(f"{sidecar_path} is not the model of {case_id} (k={k}): it says "
                             f"case_id={info.get('case_id')}, k={info.get('k')}")
        passed[case_id] = status == "PASSED"
    return passed


def _perturbation_order(perturbation) -> int:
    """Sort key: the clean field first, then the damage types in PERTURBATION_NAMES order."""
    if perturbation == CLEAN:
        return -1
    return PERTURBATION_NAMES.index(perturbation) if perturbation in PERTURBATION_NAMES else len(PERTURBATION_NAMES)


def build_field_table(scores: pd.DataFrame, passed: dict) -> pd.DataFrame:
    """
    The per-field table (confirmation_scores_raw): the scored fields plus
    passed_acceptance and, for each score, preflagged_<score> (True on every field
    of a case whose clean field that score flags), in a fixed column and row order.

    Inputs:
        scores (pd.DataFrame): Output of load_scores().
        passed (dict): Output of read_passed_acceptance().

    Outputs:
        pd.DataFrame: One row per field, sorted by case, clean field first, then
                      damage type and strength.
    """
    fields = scores.copy()
    passed_acceptance = fields["case_id"].map(passed)
    if passed_acceptance.isna().any():
        raise ValueError(f"No passed_acceptance for {sorted(fields.loc[passed_acceptance.isna(), 'case_id'].unique())}")
    fields["passed_acceptance"] = passed_acceptance.astype(bool)
    clean = fields[fields["label"] == "clean"].set_index("case_id")
    for name in SCORE_NAMES:
        fields[f"preflagged_{name}"] = fields["case_id"].map(clean[f"flagged_{name}"]).astype(bool)

    columns = ["case_id", "k", "perturbation", "epsilon", "label", *PHS_COMPONENT_NAMES, *BAR_NAMES, *SCORE_NAMES,
               *(f"flagged_{name}" for name in SCORE_NAMES), "passed_acceptance",
               *(f"preflagged_{name}" for name in SCORE_NAMES)]
    fields = fields[columns]
    order = fields["perturbation"].map(_perturbation_order)
    fields = fields.assign(_order=order).sort_values(["case_id", "_order", "epsilon"], kind="stable")
    return fields.drop(columns="_order").reset_index(drop=True)


# --- Metrics of one score on one group of fields ---
# Each takes a score_view() of the fields, works over rows and returns NaN where undefined, as
# grouped_eval.py's case_bootstrap needs. They are computed from counts and ranks rather than with
# sklearn, whose input checks would take most of the time of the 2000 resamples; the unit tests
# check them against sklearn.

def score_view(fields: pd.DataFrame, score: str) -> pd.DataFrame:
    """
    The columns the metrics of one score need, under fixed names: case_id, k,
    damaged (True for a damaged field), flagged and preflagged (of this score) and
    value (the score). Small, so that each bootstrap resample is quick to copy.
    """
    return pd.DataFrame({
        "case_id": fields["case_id"].to_numpy(),
        "k": fields["k"].to_numpy(),
        "damaged": (fields["label"] != "clean").to_numpy(),
        "flagged": fields[f"flagged_{score}"].to_numpy(dtype=bool),
        "preflagged": fields[f"preflagged_{score}"].to_numpy(dtype=bool),
        "value": fields[score].to_numpy(dtype=float),
    })


def auc(view: pd.DataFrame) -> float:
    """
    ROC-AUC of the score, damaged fields vs clean ones: the chance that a damaged
    field scores above a clean one, ties counting one half (the Mann-Whitney
    statistic, from average ranks). NaN without both kinds of field.
    """
    damaged = view["damaged"].to_numpy()
    n_damaged = int(damaged.sum())
    n_clean = len(damaged) - n_damaged
    if n_damaged == 0 or n_clean == 0:
        return np.nan
    ranks = rankdata(view["value"].to_numpy())
    return float((ranks[damaged].sum() - n_damaged * (n_damaged + 1) / 2) / (n_damaged * n_clean))


def precision(view: pd.DataFrame) -> float:
    """Flagged damaged fields / flagged fields; NaN when nothing is flagged."""
    flagged = view["flagged"].to_numpy()
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return np.nan
    return int((flagged & view["damaged"].to_numpy()).sum()) / n_flagged


def recall(view: pd.DataFrame) -> float:
    """Flagged damaged fields / damaged fields; NaN without damaged fields."""
    damaged = view["damaged"].to_numpy()
    n_damaged = int(damaged.sum())
    if n_damaged == 0:
        return np.nan
    return int((view["flagged"].to_numpy() & damaged).sum()) / n_damaged


def recall_new(view: pd.DataFrame) -> float:
    """recall over the cases the score doesn't already flag (preflagged False)."""
    return recall(view[~view["preflagged"].to_numpy()])


def f1(view: pd.DataFrame) -> float:
    """2 TP / (2 TP + FP + FN); NaN when there is nothing to find and nothing is flagged."""
    flagged = view["flagged"].to_numpy()
    damaged = view["damaged"].to_numpy()
    tp = int((flagged & damaged).sum())
    denominator = 2 * tp + int((flagged & ~damaged).sum()) + int((~flagged & damaged).sum())
    if denominator == 0:
        return np.nan
    return 2 * tp / denominator


BOOTSTRAP_METRICS = {"auc": auc, "precision": precision, "recall": recall, "recall_new": recall_new, "f1": f1}


def _bootstrap(row: dict, name: str, view: pd.DataFrame, n_boot: int, seed: int) -> None:
    """Adds <name>, <name>_low, <name>_high and <name>_n_boot to row, from case_bootstrap()."""
    estimate, low, high, used = case_bootstrap(view, BOOTSTRAP_METRICS[name], n_boot=n_boot, seed=seed)
    row.update({name: estimate, f"{name}_low": low, f"{name}_high": high, f"{name}_n_boot": used})


def _exact(row: dict, name: str, count: int, n: int) -> None:
    """Adds <name>, <name>_low and <name>_high to row: the rate count/n and its clopper_pearson() range."""
    if n == 0:
        row.update({name: np.nan, f"{name}_low": np.nan, f"{name}_high": np.nan})
        return
    low, high = clopper_pearson(count, n)
    row.update({name: count / n, f"{name}_low": low, f"{name}_high": high})


def metrics_row(fields: pd.DataFrame, score: str, tau: float, one_field_per_case: bool,
                n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """
    Every metric of one score on one group of fields, with its range.

    Inputs:
        fields (pd.DataFrame): The group: its damaged fields and the clean fields of its cases,
                               with the columns of build_field_table().
        score (str): One of SCORE_NAMES.
        tau (float): The score's frozen alarm level (reported only; the flags are in fields).
        one_field_per_case (bool): True for one damage type and strength, where each case has one
                                   damaged field: recall and recall_new then get the exact range
                                   of their count instead of the bootstrap.
        n_boot (int): Resamples of case_bootstrap.
        seed (int): Seed of case_bootstrap. Every metric uses the same resamples.

    Outputs:
        dict: The metrics row (see the module docstring for the columns).
    """
    view = score_view(fields, score)
    flagged = view["flagged"].to_numpy()
    preflagged = view["preflagged"].to_numpy()
    damaged = view["damaged"].to_numpy()
    clean = ~damaged
    if one_field_per_case and (view.loc[damaged, "case_id"].value_counts() != 1).any():
        raise ValueError("one_field_per_case=True needs exactly one damaged field per case")
    passed_by_case = fields.groupby("case_id")["passed_acceptance"].first()

    row = {"score": score, "tau": tau,
           "n_cases": int(view["case_id"].nunique()), "n_failed_runs": int((~passed_by_case).sum()),
           "n_clean": int(clean.sum()), "n_damaged": int(damaged.sum())}
    _bootstrap(row, "auc", view, n_boot, seed)
    _bootstrap(row, "precision", view, n_boot, seed)

    row["n_damaged_flagged"] = int((flagged & damaged).sum())
    if one_field_per_case:
        _exact(row, "recall", row["n_damaged_flagged"], row["n_damaged"])
    else:
        _bootstrap(row, "recall", view, n_boot, seed)

    new = damaged & ~preflagged
    row["n_preflagged_cases"] = int(view.loc[preflagged, "case_id"].nunique())
    row["n_damaged_new"] = int(new.sum())
    row["n_damaged_new_flagged"] = int((flagged & new).sum())
    if one_field_per_case:
        _exact(row, "recall_new", row["n_damaged_new_flagged"], row["n_damaged_new"])
    else:
        _bootstrap(row, "recall_new", view, n_boot, seed)

    _bootstrap(row, "f1", view, n_boot, seed)
    row["n_false_alarms"] = int((flagged & clean).sum())
    _exact(row, "fpr", row["n_false_alarms"], row["n_clean"])
    return row


def _metric_columns(one_field_per_case: bool) -> list:
    """The metric columns of a metrics table, in order."""
    def ranged(name, bootstrap=True):
        return [name, f"{name}_low", f"{name}_high"] + ([f"{name}_n_boot"] if bootstrap else [])
    return ["tau", "n_cases", "n_failed_runs", "n_clean", "n_damaged",
            *ranged("auc"), *ranged("precision"),
            *ranged("recall", not one_field_per_case), "n_damaged_flagged",
            *ranged("recall_new", not one_field_per_case), "n_preflagged_cases", "n_damaged_new",
            "n_damaged_new_flagged",
            *ranged("f1"), "fpr", "fpr_low", "fpr_high", "n_false_alarms"]


def _groups(fields: pd.DataFrame, by: str):
    """
    Yields (keys, group) for one breakdown of the fields:
        "overall"   all fields
        "k"         the fields of the cases with each k
        "type"      each damage type's fields plus all clean fields
        "type_eps"  each (damage type, strength)'s fields plus all clean fields
    """
    clean = fields[fields["label"] == "clean"]
    damaged = fields[fields["label"] != "clean"]
    perturbations = sorted(damaged["perturbation"].unique(), key=_perturbation_order)
    if by == "overall":
        yield {}, fields
    elif by == "k":
        for k in sorted(fields["k"].unique()):
            yield {"k": k}, fields[fields["k"] == k]
    elif by == "type":
        for perturbation in perturbations:
            yield {"perturbation": perturbation}, pd.concat([clean, damaged[damaged["perturbation"] == perturbation]])
    elif by == "type_eps":
        for perturbation in perturbations:
            of_type = damaged[damaged["perturbation"] == perturbation]
            for epsilon in sorted(of_type["epsilon"].unique()):
                yield ({"perturbation": perturbation, "epsilon": epsilon},
                       pd.concat([clean, of_type[of_type["epsilon"] == epsilon]]))
    else:
        raise ValueError(f"Unknown breakdown {by!r}")


def metrics_table(fields: pd.DataFrame, thresholds: dict, by: str,
                  n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    The metrics of every score for one breakdown ("overall", "k", "type" or
    "type_eps"; see _groups()), one row per score x group, scores in SCORE_NAMES order.

    Inputs:
        fields (pd.DataFrame): Output of build_field_table().
        thresholds (dict): {score: tau} of the frozen calibration.
        by (str): The breakdown.
        n_boot, seed: Of case_bootstrap.

    Outputs:
        pd.DataFrame: The table.
    """
    one_field_per_case = by == "type_eps"
    rows = []
    for score in SCORE_NAMES:
        for keys, group in _groups(fields, by):
            row = metrics_row(group, score, thresholds[score], one_field_per_case, n_boot, seed)
            rows.append({"score": score, **keys, **{key: value for key, value in row.items() if key != "score"}})
    key_columns = {"overall": [], "k": ["k"], "type": ["perturbation"], "type_eps": ["perturbation", "epsilon"]}[by]
    return pd.DataFrame(rows)[["score", *key_columns, *_metric_columns(one_field_per_case)]]


def passed_only_table(fields: pd.DataFrame, thresholds: dict, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> pd.DataFrame:
    """
    The overall and by-k metrics on the runs that passed only (metrics_passed_only),
    with a subset column: "all", then "k=1", "k=2", ...

    Inputs:
        fields (pd.DataFrame): Output of build_field_table(), failed runs included.
        thresholds (dict): {score: tau}.
        n_boot, seed: Of case_bootstrap.
    """
    passed = fields[fields["passed_acceptance"]]
    if passed.empty:
        raise ValueError("No run passed, so there is no passed-only table to make")
    overall = metrics_table(passed, thresholds, "overall", n_boot, seed)
    overall.insert(1, "subset", "all")
    by_k = metrics_table(passed, thresholds, "k", n_boot, seed)
    by_k.insert(1, "subset", by_k.pop("k").map(lambda k: f"k={k}"))
    table = pd.concat([overall, by_k], ignore_index=True)
    order = {score: i for i, score in enumerate(SCORE_NAMES)}
    return table.sort_values("score", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


# --- The per-score lists ---

def clean_false_positives(fields: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    The false alarms on clean fields of each score: overall (out of all clean
    fields, 30) and per k (out of 10), with exact ranges and the cases named.
    """
    clean = fields[fields["label"] == "clean"]
    subsets = [("all", clean)] + [(f"k={k}", clean[clean["k"] == k]) for k in sorted(clean["k"].unique())]
    rows = []
    for score in SCORE_NAMES:
        for subset, group in subsets:
            flagged = group[group[f"flagged_{score}"]]
            row = {"score": score, "tau": thresholds[score], "subset": subset,
                   "n_cases": len(group), "n_failed_runs": int((~group["passed_acceptance"]).sum()),
                   "n_clean": len(group), "n_false_alarms": len(flagged)}
            _exact(row, "fpr", len(flagged), len(group))
            row["false_alarm_cases"] = ";".join(sorted(flagged["case_id"]))
            rows.append(row)
    return pd.DataFrame(rows)


def _score_rows(fields: pd.DataFrame, thresholds: dict, select) -> pd.DataFrame:
    """One row per (field, score) that select(fields, score) keeps, with the score's value and tau."""
    rows = []
    for score in SCORE_NAMES:
        chosen = fields[select(fields, score)]
        rows.append(pd.DataFrame({
            "score": score, "tau": thresholds[score],
            "case_id": chosen["case_id"], "k": chosen["k"], "perturbation": chosen["perturbation"],
            "epsilon": chosen["epsilon"], "score_value": chosen[score],
            "score_over_tau": chosen[score] / thresholds[score],
            "preflagged": chosen[f"preflagged_{score}"], "passed_acceptance": chosen["passed_acceptance"],
        }))
    return pd.concat(rows, ignore_index=True)


def false_alarms(fields: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Every clean field above tau, one row per field x score."""
    table = _score_rows(fields, thresholds,
                        lambda f, score: (f["label"] == "clean") & f[f"flagged_{score}"])
    return table.drop(columns="preflagged")  # A clean false alarm is its case's preflagging itself


def missed_detections(fields: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Every damaged field not above tau, one row per field x score."""
    return _score_rows(fields, thresholds, lambda f, score: (f["label"] != "clean") & ~f[f"flagged_{score}"])


def preflagged_cases(fields: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    The already-flagged cases of each score: those whose clean field is above
    tau, with that clean score and how many of the case's damaged fields the
    score flags (detections that are not new).
    """
    rows = []
    for score in SCORE_NAMES:
        for case_id, case in fields[fields[f"preflagged_{score}"]].groupby("case_id", sort=True):
            clean = case[case["label"] == "clean"].iloc[0]
            damaged = case[case["label"] != "clean"]
            rows.append({"score": score, "tau": thresholds[score], "case_id": case_id, "k": clean["k"],
                         "clean_score": clean[score], "clean_score_over_tau": clean[score] / thresholds[score],
                         "n_damaged_flagged": int(damaged[f"flagged_{score}"].sum()), "n_damaged": len(damaged),
                         "passed_acceptance": bool(clean["passed_acceptance"])})
    columns = ["score", "tau", "case_id", "k", "clean_score", "clean_score_over_tau", "n_damaged_flagged",
               "n_damaged", "passed_acceptance"]
    return pd.DataFrame(rows, columns=columns)


# --- LaTeX ---

_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex_escape(text) -> str:
    """Escapes LaTeX's special characters in a cell or header."""
    return "".join(_TEX_SPECIAL.get(char, char) for char in str(text))


def _number(value, digits: int = 3) -> str:
    """A number with `digits` decimals, or "--" when undefined."""
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def _with_range(row, name: str) -> str:
    """'0.953 [0.921, 0.978]', or '--' when the metric is undefined."""
    if pd.isna(row[name]):
        return "--"
    return f"{_number(row[name])} [{_number(row[f'{name}_low'])}, {_number(row[f'{name}_high'])}]"


def _count_with_range(row, count: str, total: str, rate: str) -> str:
    """'2/30 [0.008, 0.221]': a count, and the exact range of its rate."""
    return f"{int(row[count])}/{int(row[total])} [{_number(row[f'{rate}_low'])}, {_number(row[f'{rate}_high'])}]"


METRICS_CAPTION = ("Five scores, frozen calibration (score above $\\tau$ is flagged). Each cell: estimate [95\\% range]. "
                   "AUC, precision, F1, and recall over several strengths: case bootstrap, 2000 resamples, whole cases "
                   "within each $k$. False alarms (flagged clean fields), and recall at one damage type and strength: "
                   "exact Clopper--Pearson range. Recall (new): over the cases whose clean field the score does not "
                   "flag. -- : undefined.")


def metrics_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """A metrics table as LaTeX: one formatted cell per metric (estimate and range)."""
    rows = []
    for _, row in table.iterrows():
        cells = {"Score": _tex_escape(row["score"])}
        if "subset" in row:
            cells["Subset"] = _tex_escape(row["subset"])
        if "k" in row:
            cells["$k$"] = str(row["k"])
        if "perturbation" in row:
            cells["Damage"] = _tex_escape(row["perturbation"])
        if "epsilon" in row:
            cells["$\\varepsilon$"] = f"{row['epsilon']:g}"
        cells.update({
            "Cases": str(int(row["n_cases"])), "Failed runs": str(int(row["n_failed_runs"])),
            "AUC": _with_range(row, "auc"), "Precision": _with_range(row, "precision"),
            "Recall": _with_range(row, "recall"), "Recall (new)": _with_range(row, "recall_new"),
            "F1": _with_range(row, "f1"),
            "False alarms": _count_with_range(row, "n_false_alarms", "n_clean", "fpr"),
        })
        rows.append(cells)
    return pd.DataFrame(rows).to_latex(index=False, escape=False, longtable=len(rows) > 40,
                                       caption=caption, label=label)


# Raw components are of order 1e-6, so they are written with 3 significant digits instead of 3 decimals
_SIGNIFICANT_DIGIT_COLUMNS = set(PHS_COMPONENT_NAMES)


def list_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """A per-field or per-case table as a LaTeX longtable, every column as in the .csv, numbers rounded."""
    def cell(column, value):
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return "--"
            if column in _SIGNIFICANT_DIGIT_COLUMNS:
                return f"{value:.3g}"
            if column == "epsilon":
                return f"{value:g}"
            return _number(value)
        return _tex_escape(value)

    formatted = pd.DataFrame({_tex_escape(column): [cell(column, value) for value in table[column]]
                              for column in table.columns})
    return formatted.to_latex(index=False, escape=False, longtable=True, caption=caption, label=label)


CAPTIONS = {
    "confirmation_scores_raw": "All fields of the 30 new cases: raw and normalized components, the five scores, "
                               "their flags (score above $\\tau$), passed\\_acceptance and preflagged\\_$<$score$>$.",
    "metrics_overall": "Detection of the damaged fields, all cases. " + METRICS_CAPTION,
    "metrics_by_k": "Detection of the damaged fields, by $k$. " + METRICS_CAPTION,
    "metrics_by_type": "Detection of the damaged fields, by damage type (against all clean fields, so the false "
                       "alarms repeat the overall ones). " + METRICS_CAPTION,
    "metrics_by_type_eps": "Detection of the damaged fields, by damage type and strength (against all clean "
                           "fields). " + METRICS_CAPTION,
    "metrics_passed_only": "Detection of the damaged fields, runs that passed the acceptance test only. "
                           + METRICS_CAPTION,
    "clean_false_positives": "False alarms on the clean fields: count, rate and exact 95\\% range, overall and by $k$.",
    "preflagged_cases": "Cases whose clean field a score already flags, and how many of their damaged fields it flags.",
    "false_alarms": "Every clean field above $\\tau$, one row per field and score.",
    "missed_detections": "Every damaged field not above $\\tau$, one row per field and score.",
}


def to_latex(name: str, table: pd.DataFrame) -> str:
    """The LaTeX of one output table."""
    label = f"tab:benchmark_{name}"
    if name.startswith("metrics_"):
        return metrics_latex(table, CAPTIONS[name], label)
    return list_latex(table, CAPTIONS[name], label)


# --- Running it ---

def make_tables(scores: pd.DataFrame, passed: dict, thresholds: dict, n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> dict:
    """
    Builds every output table from the scored fields.

    Inputs:
        scores (pd.DataFrame): Output of load_scores().
        passed (dict): Output of read_passed_acceptance().
        thresholds (dict): {score: tau} of the frozen calibration.
        n_boot, seed: Of case_bootstrap.

    Outputs:
        dict: {table name: DataFrame}; metrics_passed_only only if any run failed.
    """
    fields = build_field_table(scores, passed)
    tables = {
        "confirmation_scores_raw": fields,
        "metrics_overall": metrics_table(fields, thresholds, "overall", n_boot, seed),
        "metrics_by_k": metrics_table(fields, thresholds, "k", n_boot, seed),
        "metrics_by_type": metrics_table(fields, thresholds, "type", n_boot, seed),
        "metrics_by_type_eps": metrics_table(fields, thresholds, "type_eps", n_boot, seed),
    }
    if not fields["passed_acceptance"].all():
        tables[PASSED_ONLY] = passed_only_table(fields, thresholds, n_boot, seed)
    tables.update({
        "clean_false_positives": clean_false_positives(fields, thresholds),
        "preflagged_cases": preflagged_cases(fields, thresholds),
        "false_alarms": false_alarms(fields, thresholds),
        "missed_detections": missed_detections(fields, thresholds),
    })
    return tables


def write_tables(tables: dict, output_dir) -> list:
    """
    Writes each table as <name>.csv (full precision, "NaN" where undefined) and
    <name>.tex, with "\\n" line endings on every OS. Removes a metrics_passed_only
    left by an earlier run when this run has no failed run.

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
    if PASSED_ONLY not in tables:
        for stale in (output_dir / f"{PASSED_ONLY}.csv", output_dir / f"{PASSED_ONLY}.tex"):
            if stale.exists():
                stale.unlink()
                print(f"Removed {stale}: every run passed, so there is no passed-only table")
    return written


def run(scores_csv=DEFAULT_SCORES_CSV, calibration_json=DEFAULT_CALIBRATION_JSON, models_dir=DEFAULT_MODELS_DIR,
        output_dir=DEFAULT_OUTPUT_DIR, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
        metadata=DEFAULT_METADATA) -> dict:
    """
    Reads the inputs, builds every table and writes them. See the module docstring.
    metadata is the case file whose every case must be in scores_csv.

    Outputs:
        dict: {table name: DataFrame}, as written.
    """
    calib = frozen_calibration.load(calibration_json)
    scores = load_scores(scores_csv, calib)
    check_cases(scores, load_case_metadata(Path(metadata)))
    passed = read_passed_acceptance(models_dir, scores.groupby("case_id")["k"].first())
    tables = make_tables(scores, passed, calib["thresholds"], n_boot, seed)
    write_tables(tables, output_dir)
    return tables


def print_summary(tables: dict) -> None:
    """Prints the headline numbers of each score."""
    fields = tables["confirmation_scores_raw"]
    n_cases = fields["case_id"].nunique()
    n_failed = int((~fields.groupby("case_id")["passed_acceptance"].first()).sum())
    print(f"{len(fields)} fields: {n_cases} cases x {len(fields) // max(n_cases, 1)} fields; "
          f"{n_failed} failed run(s), kept in every table")
    clean_fp = tables["clean_false_positives"]
    for _, row in tables["metrics_overall"].iterrows():
        per_k = clean_fp[(clean_fp["score"] == row["score"]) & (clean_fp["subset"] != "all")]
        per_k_text = ", ".join(f"{r.subset}: {r.n_false_alarms}/{r.n_clean}" for r in per_k.itertuples())
        print(f"  {row['score']:28s} AUC {_with_range(row, 'auc')}  recall {_number(row['recall'])}  "
              f"recall (new) {_number(row['recall_new'])}  precision {_number(row['precision'])}  "
              f"F1 {_number(row['f1'])}  false alarms {row['n_false_alarms']}/{row['n_clean']} ({per_k_text})")
    for name in ("false_alarms", "missed_detections"):
        counts = tables[name].groupby("score").size().reindex(list(SCORE_NAMES), fill_value=0)
        print(f"  {name}: " + ", ".join(f"{score.split('_')[0]} {n}" for score, n in counts.items()))


def parse_args(argv=None):
    """
    Parses the command line.

    Outputs:
        argparse.Namespace: scores, calibration, metadata, models_dir, output_dir (str).
    """
    parser = argparse.ArgumentParser(description="Write the tables of the perturbation benchmark (#35) from the "
                                                 "scored fields and the frozen calibration.")
    parser.add_argument("--scores", type=str, default=str(DEFAULT_SCORES_CSV),
                        help="evaluate_phs.py --calibration's phs_components_frozen.csv "
                             "(default: data/confirmation/phs_scores/phs_components_frozen.csv).")
    parser.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_JSON),
                        help="The frozen calibration the fields were scored with "
                             "(default: calibration/frozen_calibration.json).")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="The case file the fields were made from; every case in it must be scored "
                             "(default: cases/confirmation_cases.json).")
    parser.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Folder with the {case_id}/lbfgs_final.json sidecars, for passed_acceptance "
                             "(default: models/confirmation).")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Folder to write the tables into (default: results/benchmark).")
    return parser.parse_args(argv)


def main(argv=None):
    """Writes every table of the benchmark and prints the headline numbers."""
    args = parse_args(argv)
    tables = run(args.scores, args.calibration, args.models_dir, args.output_dir, metadata=args.metadata)
    print(f"Wrote {len(tables)} tables (.csv + .tex) to {args.output_dir}")
    if PASSED_ONLY not in tables:
        print("Every run passed, so metrics_passed_only was not written")
    print_summary(tables)


if __name__ == "__main__":
    enable_utf8_output()
    main()
