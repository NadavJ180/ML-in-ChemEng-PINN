"""
Frozen calibration of the final iteration (#45).

Barak: "Use only the 30 existing clean fields to calculate the component
normalizers and each score's empirical 95th-percentile threshold. Freeze these
before evaluating the new cases."

The calibration is fitted once, on the clean field of each of the 30 old cases
(case_00 ... case_29, all three old splits) in
calibration/source_phs_components_raw.csv, and saved to
calibration/frozen_calibration.json:

    normalizer of each component (mom, div, bc, E) = its mean over the 30 fields
    tau of each of the five scores (phs.SCORE_DEFINITIONS) = its 95th percentile
        over the same 30 fields (np.percentile, linear interpolation)

New fields are scored with apply(), which only reads these numbers and never
fits anything. The formulas are the old pipeline's (phs.normalize_components,
phs.compute_baseline_scores, phs.select_threshold); only the fields the numbers
come from differ (the old run used the 5 clean validation fields), plus the two
new scores. A field is flagged by a score when the score is above its tau.

    fit(csv_path)      -> calibration dict, from the 30 clean rows only
    save(calib, path)  writes it as JSON
    load(path)         reads it back, checking it was fitted for phs.SCORE_DEFINITIONS
    apply(df, calib)   -> df plus the normalized components, the 5 scores and the 5 flags

Usage:
    python src/detection/frozen_calibration.py      # writes calibration/frozen_calibration.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.detection.phs import (
    compute_baseline_scores,
    compute_normalizers,
    normalize_components,
    select_threshold,
    PHS_COMPONENT_NAMES,
    SCORE_DEFINITIONS,
)
from src.models.checkpoints import _git_state  # The same commit/dirty record as the checkpoint sidecars
from src.utils.console import enable_utf8_output

DEFAULT_SOURCE_CSV = PROJECT_ROOT / "calibration" / "source_phs_components_raw.csv"
DEFAULT_CALIBRATION_JSON = PROJECT_ROOT / "calibration" / "frozen_calibration.json"

FORMAT_VERSION = 1
N_CLEAN_FIELDS = 30       # One clean field per old case
PERCENTILE = 95.0
PERCENTILE_METHOD = "linear"  # np.percentile's default, which phs.select_threshold uses
NORMALIZER_EPS = 1e-12    # S_bar = S / (normalizer + eps), phs.normalize_components' default
FLAG_RULE = "score > tau"


def _sha256(path) -> str:
    """Returns the SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    """Returns path relative to the project root with forward slashes, or only its name if it is outside."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def fit(csv_path) -> dict:
    """
    Fits the calibration on the clean field of each of the 30 old cases in
    csv_path, and on nothing else: damaged fields are never used.

    Inputs:
        csv_path (str | Path): One row per field, with "case_id", "label" and the
                               raw components "mom", "div", "bc", "E"
                               (calibration/source_phs_components_raw.csv).

    Outputs:
        dict: The calibration as save() writes it: the normalizers, the scores with
              their tau, how both were computed, and where they come from (the
              source file's SHA-256 and the git commit).
    """
    csv_path = Path(csv_path)
    # round_trip reads every value back exactly as it was written
    fields = pd.read_csv(csv_path, float_precision="round_trip")
    clean = fields[fields["label"] == "clean"]
    n_cases = clean["case_id"].nunique()
    if len(clean) != N_CLEAN_FIELDS or n_cases != N_CLEAN_FIELDS:
        raise ValueError(f"{csv_path} must have exactly {N_CLEAN_FIELDS} clean fields, one per case, "
                         f"but has {len(clean)} for {n_cases} cases")

    normalizers = compute_normalizers(clean)
    scored = compute_baseline_scores(normalize_components(clean, normalizers, eps=NORMALIZER_EPS),
                                     SCORE_DEFINITIONS)
    thresholds = {name: select_threshold(scored[name].values, PERCENTILE) for name in SCORE_DEFINITIONS}

    git_commit, git_dirty = _git_state()
    return {
        "format_version": FORMAT_VERSION,
        "description": "Frozen calibration of the final iteration (#45): normalizers and alarm levels "
                       "fitted once on the clean fields of the 30 old cases. Score new fields with "
                       "src/detection/frozen_calibration.py apply(); never refit.",
        "source_csv": _repo_path(csv_path),
        "source_csv_sha256": _sha256(csv_path),
        "fit_case_ids": sorted(clean["case_id"]),
        "normalizers": normalizers,
        "normalizer_eps": NORMALIZER_EPS,
        "scores": {name: list(components) for name, components in SCORE_DEFINITIONS.items()},
        "thresholds": thresholds,
        "percentile": PERCENTILE,
        "percentile_method": PERCENTILE_METHOD,
        "flag_rule": FLAG_RULE,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def save(calib: dict, json_path) -> None:
    """
    Writes a calibration to json_path. Line endings are "\\n" on every OS, so the
    file's SHA-256 (recorded in PROTOCOL.md) is the same everywhere.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(calib, f, indent=2)
        f.write("\n")


def load(json_path) -> dict:
    """
    Reads a calibration written by save(). Refuses a file fitted for other scores
    than phs.SCORE_DEFINITIONS, since its tau values would not belong to the
    scores computed here.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    if calib.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{json_path} has format_version {calib.get('format_version')}, expected {FORMAT_VERSION}")
    if calib["scores"] != {name: list(components) for name, components in SCORE_DEFINITIONS.items()}:
        raise ValueError(f"{json_path} was fitted for other scores than phs.SCORE_DEFINITIONS")
    if set(calib["thresholds"]) != set(calib["scores"]) or set(calib["normalizers"]) != set(PHS_COMPONENT_NAMES):
        raise ValueError(f"{json_path} must have one tau per score and one normalizer per component")
    return calib


def apply(df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    Scores fields with a frozen calibration, without fitting anything on them.
    Each row is scored on its own, so a field's scores don't depend on which other
    fields are in df.

    Adds, for each component j, j_bar = S_j / (normalizer_j + eps); for each score,
    its column (the sum of its components' j_bar); and for each score,
    flagged_<score> = score > tau. Columns that already have these names, such as
    the old pipeline's, are replaced.

    A field with a missing (NaN) component is refused: its scores would be NaN,
    and NaN > tau is False, so it would silently count as not flagged.

    Inputs:
        df (pd.DataFrame): One row per field, with the raw components "mom", "div", "bc", "E".
        calib (dict): Output of fit() or load().

    Outputs:
        pd.DataFrame: A copy of df with those columns.
    """
    normalizers = calib["normalizers"]
    missing = df[list(normalizers)].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} field(s) have a missing component, which would count as not "
                         "flagged; find out why before scoring them")
    out = normalize_components(df, normalizers, eps=calib["normalizer_eps"], component_names=list(normalizers))
    out = compute_baseline_scores(out, calib["scores"])
    for score_name, tau in calib["thresholds"].items():
        out[f"flagged_{score_name}"] = out[score_name] > tau
    return out


def parse_args(argv=None):
    """
    Parses the command line.

    Inputs:
        argv (list[str] | None): Arguments; None reads sys.argv.

    Outputs:
        argparse.Namespace: source (str), out (str), overwrite (bool).
    """
    parser = argparse.ArgumentParser(description="Fit the frozen calibration (#45) on the 30 clean fields "
                                                 "of the old cases.")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE_CSV),
                        help="The old scores to fit on (default: calibration/source_phs_components_raw.csv).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_CALIBRATION_JSON),
                        help="Where to write the calibration (default: calibration/frozen_calibration.json).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace --out if it exists. The committed file is frozen by the tag protocol-v1 "
                             "and is never rewritten after it.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Fits the calibration on --source and writes it to --out.

    Inputs:
        argv (list[str] | None): Command-line arguments; None reads sys.argv.

    Outputs:
        None. Writes the JSON and prints the fitted values.
    """
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists. It is frozen; pass --overwrite only to replace it on purpose.")

    calib = fit(args.source)
    save(calib, out)

    print(f"Wrote {out}")
    print(f"Fitted on the clean fields of {len(calib['fit_case_ids'])} cases "
          f"({calib['source_csv']}, SHA-256 {calib['source_csv_sha256']})")
    for component, normalizer in calib["normalizers"].items():
        print(f"  normalizer {component:4s} {normalizer:.6e}")
    for score_name, tau in calib["thresholds"].items():
        print(f"  tau {score_name:28s} {tau:.6f}")
    if calib["git_dirty"] is not False:
        print("WARNING: tracked files have uncommitted changes, or this is not a git checkout, so git_commit "
              "does not describe the code exactly. Commit first, then rerun with --overwrite.")


if __name__ == "__main__":
    enable_utf8_output()
    main()
