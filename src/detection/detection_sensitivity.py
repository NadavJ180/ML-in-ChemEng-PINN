"""
Detection Sensitivity Analysis: rescores the damage strengths of EPSILON_VALUES against an
EXISTING calibration (normalizers_and_thresholds.json from an evaluate_phs.py run, loaded,
never refitted), and reports where recall crosses 50% and 90%.

Recall is reported against epsilon and against the relative change a damage makes to the
field (phs.compute_relative_error). Epsilon means a different thing for each damage type
(a fraction of U0, of a bump amplitude, of a decay time); the relative change means the
same for all of them.

Outputs:
  data/phs_scores/sensitivity_probe_raw.csv
      One row per (case, perturbation, epsilon): raw components, Score3_PHS_full with the
      loaded calibration, the relative change, and whether it was detected.
  plots/phs_evaluation/sensitivity_recall_vs_epsilon.png
      Recall against epsilon, one line per damage type, plus all types pooled.
  plots/phs_evaluation/sensitivity_boundary_summary.csv / .json
      The epsilon and relative change at which recall crosses 50% and 90% (linear
      interpolation in log x), pooled and per damage type. With about 35 fields per type,
      the per-type crossings are rough.

Usage:
    python src/detection/detection_sensitivity.py
    python src/detection/detection_sensitivity.py --calibration_dir plots/phs_evaluation
    python src/detection/detection_sensitivity.py --case_id case_25 --case_id case_26
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.models.scaling import ResidualScaler
from src.physics.taylor_green import compute_nu, compute_T, compute_decay_timescale
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.hallucinations.perturbations import PERTURBATION_NAMES, EPSILON_VALUES
from src.detection.phs import compute_phs_components, compute_relative_error, PHS_COMPONENT_NAMES
from src.utils.console import enable_utf8_output

# Recall levels to report boundary crossings for.
BOUNDARY_LEVELS = [0.5, 0.9]

_LINE_STYLES = [
    ("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ("-", "v"), ("--", "P"), ("-.", "X"),
]


def parse_args():
    """
    Parses command-line arguments controlling which case(s) to probe, the
    resolution of each PHS/relative-error computation, and where the
    existing calibration is loaded from.

    Inputs:
        None (reads directly from sys.argv).

    Outputs:
        args (argparse.Namespace).
    """
    parser = argparse.ArgumentParser(description="Probe the detection boundary below the canonical epsilon range.")
    parser.add_argument("--case_id", action="append", default=None,
                        help="Restrict to this case_id. Repeatable. Defaults to every TEST-split case with a "
                             "trained model, since the loaded calibration was tuned to generalize to test cases.")
    parser.add_argument("--calibration_dir", type=str, default=None,
                        help="Directory containing normalizers_and_thresholds.json from a prior evaluate_phs.py "
                             "run. Defaults to <project_root>/plots/phs_evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_interior", type=int, default=4000)
    parser.add_argument("--n_bc", type=int, default=400)
    parser.add_argument("--n_time", type=int, default=8)
    parser.add_argument("--energy_res", type=int, default=16)
    parser.add_argument("--n_relative_error_points", type=int, default=5000)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory for plots. Defaults to <project_root>/plots/phs_evaluation.")
    return parser.parse_args()


def load_model(case_id: str, k: float, device: str):
    """
    Loads the trained BaselinePINN of one case from models/{case_id}.pth.

    Inputs:
        case_id (str): The case identifier (e.g. "case_00").
        k (float): The case's wavenumber, required by the BaselinePINN constructor.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        model (nn.Module): The loaded, float64, eval-mode model on `device`.
    """
    model_path = project_root / "models" / f"{case_id}.pth"
    model = BaselinePINN(k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.to(torch.float64)
    model.eval()
    return model


def probe_sensitivity(case_ids: list, case_meta_by_id: dict, models_dir: Path, args) -> pd.DataFrame:
    """
    For every case in `case_ids`, computes PHS components AND relative L2
    error at every EPSILON_VALUES x PERTURBATION_NAMES
    combination.

    Inputs:
        case_ids (list[str]): Which cases to probe.
        case_meta_by_id (dict): Output of load_case_metadata().
        models_dir (Path): Directory containing {case_id}.pth files.
        args (argparse.Namespace): Parsed CLI arguments.

    Outputs:
        pd.DataFrame: One row per (case_id, perturbation_type, epsilon),
            with raw components, "relative_error", "case_id",
            "perturbation_type", "epsilon".
    """
    rows = []
    for case_id in case_ids:
        model_path = models_dir / f"{case_id}.pth"
        if not model_path.exists():
            print(f"⏭️  Skipping {case_id}: no trained model found at {model_path}")
            continue
        case_meta = case_meta_by_id.get(case_id)
        if case_meta is None:
            print(f"⏭️  Skipping {case_id}: not found in cases_metadata.json")
            continue

        Re, U0, k = case_meta["Re"], case_meta["U0"], case_meta["k"]
        nu = compute_nu(U0, Re, k)
        T = compute_T(U0, Re, k)
        scaler = ResidualScaler(U0, k)
        model = load_model(case_id, k, args.device)
        params = {"U0": U0, "k": k, "T": T, "tau_decay": compute_decay_timescale(nu, k)}
        seed = 1000 + int(case_id.split("_")[1])

        print(f"[{case_id}] probing {len(PERTURBATION_NAMES)} perturbation types x "
              f"{len(EPSILON_VALUES)} epsilons...")
        for perturbation_name in PERTURBATION_NAMES:
            for epsilon in EPSILON_VALUES:
                components = compute_phs_components(
                    model, case_meta, nu, T, scaler, perturbation_name, epsilon,
                    n_interior=args.n_interior, n_bc_per_axis=args.n_bc,
                    n_time=args.n_time, energy_res=args.energy_res,
                    chunk_size=args.chunk_size, device=args.device, seed=seed,
                )
                rel_error = compute_relative_error(
                    model, T, params, perturbation_name, epsilon,
                    n_points=args.n_relative_error_points, seed=seed, device=args.device,
                )
                rows.append({
                    "case_id": case_id, "perturbation_type": perturbation_name, "epsilon": epsilon,
                    "relative_error": rel_error, **components,
                })

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def score_against_existing_calibration(df: pd.DataFrame, normalizers: dict, tau: float) -> pd.DataFrame:
    """
    Normalizes and scores an already-probed DataFrame using a PRE-EXISTING
    calibration (loaded from disk, not recomputed) -- this script never
    fits normalizers or a threshold itself.

    Inputs:
        df (pd.DataFrame): Output of probe_sensitivity(); must have raw
            "mom", "div", "bc", "E" columns.
        normalizers (dict): {component_name: normalizer}, loaded from a
            prior evaluate_phs.py run's normalizers_and_thresholds.json.
        tau (float): Score3_PHS_full's threshold, from the same file.

    Outputs:
        pd.DataFrame: `df` with "Score3_PHS_full" and "detected" columns
            appended (copy).
    """
    out = df.copy()
    for c in PHS_COMPONENT_NAMES:
        out[f"{c}_bar"] = out[c] / (normalizers[c] + 1e-12)
    out["Score3_PHS_full"] = sum(out[f"{c}_bar"] for c in PHS_COMPONENT_NAMES)
    out["detected"] = out["Score3_PHS_full"] > tau
    return out


def _binned_recall(df: pd.DataFrame, x_col: str, n_bins: int = 15, min_points_per_bin: int = 4) -> tuple:
    """
    Sorts the rows by x_col, splits them into bins with equal numbers of rows, and returns
    each bin's detection rate against its mean x (geometric mean), sorted by x.

    Used for the relative change, whose values never coincide between rows, so they must be
    binned to pool anything. Bins of equal counts, not of equal width in log x, so a bin can't
    end up holding a single row. The number of bins is lowered so that each holds on average
    at least min_points_per_bin rows.

    Inputs:
        df (pd.DataFrame): Must have `x_col` and "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        n_bins (int): The largest number of bins.
        min_points_per_bin (int): The smallest average number of rows per bin.

    Outputs:
        (bin_centers, mean_recall): both np.ndarray, sorted by bin_centers. Fewer than n_bins
        points when there isn't enough data.
    """
    x = df[x_col].values
    detected = df["detected"].values
    valid = x > 0
    x, detected = x[valid], detected[valid]
    if len(x) == 0:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x_sorted, detected_sorted = x[order], detected[order]
    n_bins = max(1, min(n_bins, len(x_sorted) // min_points_per_bin))

    centers, means = [], []
    for chunk_x, chunk_detected in zip(np.array_split(x_sorted, n_bins), np.array_split(detected_sorted, n_bins)):
        if len(chunk_x) == 0:
            continue
        centers.append(10 ** np.mean(np.log10(chunk_x)))
        means.append(chunk_detected.mean())

    return np.array(centers), np.array(means)


def find_boundary_crossings(df: pd.DataFrame, x_col: str, levels: list, n_bins: int = 15) -> dict:
    """
    Finds where the detection rate crosses each level in `levels`, by linear interpolation in
    log x, on the pooled curve and on each damage type's curve (_recall_curve). Also reports
    the highest recall reached and where, so a level that is never reached still gets an
    answer.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", x_col, "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        levels (list[float]): Recall levels (e.g. [0.5, 0.9]).
        n_bins (int): Forwarded to _recall_curve.

    Outputs:
        dict: {"overall": {"crossings": {level: x_value_or_None, ...},
                            "max_recall": float, "max_recall_x": float},
               perturbation_name: {same structure}, ...}
    """
    def crossings_for(group):
        xs, ys = _recall_curve(group, x_col, n_bins)
        crossings = {}
        for level in levels:
            crossing = None
            for i in range(len(ys) - 1):
                if (ys[i] < level) != (ys[i + 1] < level):  # sign change around `level`
                    log_x0, log_x1 = np.log10(xs[i]), np.log10(xs[i + 1])
                    frac = (level - ys[i]) / (ys[i + 1] - ys[i])
                    crossing = float(10 ** (log_x0 + frac * (log_x1 - log_x0)))
                    break
            crossings[level] = crossing
        if len(ys) == 0:
            return {"crossings": crossings, "max_recall": None, "max_recall_x": None}
        max_idx = int(np.argmax(ys))
        return {"crossings": crossings, "max_recall": float(ys[max_idx]), "max_recall_x": float(xs[max_idx])}

    boundaries = {"overall": crossings_for(df)}
    for perturbation_name, group in df.groupby("perturbation_type"):
        boundaries[perturbation_name] = crossings_for(group)
    return boundaries


def _recall_curve(df: pd.DataFrame, x_col: str, n_bins: int = 15) -> tuple:
    """
    The recall curve of x_col: for "epsilon", the detection rate at each exact epsilon (every
    row at one epsilon is a replicate of the same strength); for "relative_error", the binned
    curve of _binned_recall.

    Inputs:
        df (pd.DataFrame): Must have `x_col` and "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        n_bins (int): Forwarded to _binned_recall when x_col is "relative_error".

    Outputs:
        (x_values, mean_recall): both np.ndarray, sorted by x_values.
    """
    if x_col == "epsilon":
        by_x = df.groupby(x_col)["detected"].mean().sort_index()
        return by_x.index.values, by_x.values
    return _binned_recall(df, x_col, n_bins)


def plot_recall_vs_x(df: pd.DataFrame, x_col: str, x_label: str, output_path: Path, title: str,
                      n_bins: int = 15):
    """
    Plots detection rate vs. x_col, one line per perturbation type plus an
    overall pooled line, on a log-x axis. See _recall_curve for how "epsilon"
    and "relative_error" are computed differently and why.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", x_col, "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        x_label (str): Axis label.
        output_path (Path): Where to save the figure.
        title (str): Plot title.
        n_bins (int): Forwarded to _recall_curve (only used when x_col is
            "relative_error"). Per-perturbation-type curves use fewer
            effective points than "overall" (each type has 1/5th the
            rows), so bins may be sparser for those lines.

    Outputs:
        None. Saves a PNG to output_path.
    """
    perturbation_types = sorted(df["perturbation_type"].unique())

    plt.figure(figsize=(8, 5.5))
    overall_x, overall_y = _recall_curve(df, x_col, n_bins)
    plt.plot(overall_x, overall_y, color="black", linewidth=2.5, marker="o",
              markersize=5, label="overall (pooled)", zorder=10)

    for i, perturbation_name in enumerate(perturbation_types):
        group = df[df["perturbation_type"] == perturbation_name]
        by_x, by_y = _recall_curve(group, x_col, n_bins)
        linestyle, marker = _LINE_STYLES[i % len(_LINE_STYLES)]
        plt.plot(by_x, by_y, linestyle=linestyle, marker=marker, markersize=5,
                  alpha=0.75, label=perturbation_name)

    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(0.9, color="gray", linestyle=":", alpha=0.5)
    plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Detection rate (recall, at the existing tau)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """
    Entry point. Loads an existing calibration, probes every damage type at every strength of
    EPSILON_VALUES for the requested cases, scores them against that calibration, finds where
    recall crosses 50% and 90% against epsilon and against the relative change, and writes
    the outputs listed in this module's docstring.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes files and prints a summary to stdout.
    """
    args = parse_args()

    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else project_root / "plots" / "phs_evaluation"
    calibration_path = calibration_dir / "normalizers_and_thresholds.json"
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Cannot find {calibration_path}. Run src/detection/evaluate_phs.py first -- this script probes "
            "an EXISTING calibration rather than fitting its own."
        )
    with open(calibration_path, "r") as f:
        calibration = json.load(f)
    normalizers = calibration["normalizers"]
    tau = calibration["thresholds"]["Score3_PHS_full"]

    metadata_path = project_root / "data" / "cases_metadata.json"
    case_meta_by_id = load_case_metadata(metadata_path)

    if args.case_id:
        case_ids = args.case_id
    else:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        case_ids = [c["case_id"] for c in metadata["test"]]

    models_dir = project_root / "models"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots" / "phs_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "phs_scores"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔎 DETECTION SENSITIVITY ANALYSIS")
    print(f"Using calibration from: {calibration_path.relative_to(project_root)}")
    print(f"tau = {tau:.4f}")
    print(f"Cases: {case_ids}")
    print(f"Epsilon grid: {EPSILON_VALUES}")
    print("=" * 60)

    df = probe_sensitivity(case_ids, case_meta_by_id, models_dir, args)
    if df.empty:
        raise RuntimeError("No fields were probed -- check that models/ contains the matching *.pth files.")

    df = score_against_existing_calibration(df, normalizers, tau)

    raw_csv_path = data_dir / "sensitivity_probe_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    print(f"\n💾 Wrote {raw_csv_path.relative_to(project_root)}")

    boundaries_eps = find_boundary_crossings(df, "epsilon", BOUNDARY_LEVELS)
    boundaries_err = find_boundary_crossings(df, "relative_error", BOUNDARY_LEVELS)
    summary_rows = []
    for key in boundaries_eps:
        summary_rows.append({
            "perturbation_type": key,
            **{f"epsilon_at_{int(lvl*100)}pct_recall": boundaries_eps[key]["crossings"][lvl] for lvl in BOUNDARY_LEVELS},
            **{f"relative_error_at_{int(lvl*100)}pct_recall": boundaries_err[key]["crossings"][lvl] for lvl in BOUNDARY_LEVELS},
            "max_recall_reached": boundaries_eps[key]["max_recall"],
            "max_recall_at_epsilon": boundaries_eps[key]["max_recall_x"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "sensitivity_boundary_summary.csv", index=False)
    with open(output_dir / "sensitivity_boundary_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"💾 Wrote {(output_dir / 'sensitivity_boundary_summary.csv').relative_to(project_root)}")

    plot_recall_vs_x(df, "epsilon", "Epsilon (log scale)",
                      output_dir / "sensitivity_recall_vs_epsilon.png",
                      "Detection Rate vs. Epsilon (fine-grained, below canonical range)")
    print(f"🖼️  Wrote sensitivity_recall_vs_epsilon.png to {output_dir.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print("✅ Detection boundary summary (overall, pooled across perturbation types):")
    overall_eps, overall_err = boundaries_eps["overall"], boundaries_err["overall"]
    for lvl in BOUNDARY_LEVELS:
        eps_b = overall_eps["crossings"][lvl]
        err_b = overall_err["crossings"][lvl]
        if eps_b is not None:
            eps_str = f"{eps_b:.5f}"
        else:
            eps_str = (f"not reached in this range (max recall = {overall_eps['max_recall']:.2f} "
                       f"at epsilon={overall_eps['max_recall_x']:.5f})")
        if err_b is not None:
            err_str = f"{err_b*100:.3f}%"
        else:
            err_str = (f"not reached in this range (max recall = {overall_err['max_recall']:.2f} "
                       f"at relative error={overall_err['max_recall_x']*100:.3f}%)")
        print(f"  {int(lvl*100)}% recall: epsilon ≈ {eps_str}   |   relative L2 error ≈ {err_str}")
    print("=" * 60)


if __name__ == "__main__":
    enable_utf8_output()
    main()