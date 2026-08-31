"""
Detection Sensitivity Analysis (extends below the canonical epsilon floor)

HISTORY: this script originally existed because the canonical epsilon sweep
(then [0.005, 0.01, 0.02, 0.05, 0.1]) gave a flat AUC=1.000 that didn't say
where detection actually became unreliable. That finding led directly to
EPSILON_VALUES itself being replaced (see perturbations.py) with a range
centered on the real detection boundary (roughly eps=0.0003-0.0016, per the
50%/90% recall crossings this script found). So the canonical
evaluate_phs.py run NOW produces the sensitivity curve directly, as part of
its standard output (recall_by_type.png, scores_vs_epsilon.png, etc. all
already span the boundary) -- this script is no longer the only place that
information exists.

What it's still for: probing EVEN LOWER than the new canonical floor
(0.0001), or checking specific epsilon values that aren't part of the
standard sweep, using an EXISTING calibration rather than recomputing one.
It never recalibrates anything -- normalizers and tau are loaded from disk
(from a prior evaluate_phs.py run), so this never touches validation or
test data in a way that could leak into the numbers it reports.

Epsilon itself is not a fair cross-perturbation-type axis: it means "a
fraction of U0" for one perturbation type, "a fraction of a boundary bump
amplitude" for another, "a fraction of a decay timescale" for a third --
none of those are directly comparable magnitudes. Relative L2 error IS
comparable across types (it is always "how much did this change the
field, in the same physical units, relative to the field's own scale"),
so THAT axis is what gives a meaningful, portable answer to "what is this
method's precision" -- e.g. "PHS reliably detects hallucinations once they
change the field by more than X% in relative L2 terms," a claim that
means the same thing regardless of which perturbation produced them.

Outputs:
  data/phs_scores/sensitivity_probe_raw.csv
      One row per (case, perturbation, epsilon): raw components, the
      normalized Score4_PHS_full (using the LOADED calibration), relative
      L2 error, and whether it was detected.
  plots/phs_evaluation/sensitivity_recall_vs_epsilon.png
      Recall vs. epsilon, one line per perturbation type, for whatever
      grid this script was run with.
  plots/phs_evaluation/sensitivity_boundary_summary.csv / .json
      The epsilon and relative-error values where recall crosses 50% and
      90% (linear interpolation in log-space, over QUANTILE bins -- see
      _binned_recall's docstring for why equal-width log bins produced
      visibly "jerky" curves with occasional single-point bins reading a
      hard 0% or 100%, and why equal-COUNT bins fix that), overall and
      per perturbation type. NOTE: the relative-error columns/crossings
      here are computed the same way as before, but the corresponding
      PLOT (sensitivity_recall_vs_relative_error.png) was removed --
      even after both binning fixes, per-perturbation-type curves still
      showed real 0%/100% jumps from small sample size (~35 rows/type),
      which read as more informative than they actually were. The single
      interpolated crossing NUMBER per type in this file is less
      misleading than the jumpy curve was, since it's one summary value
      rather than a chart inviting over-interpretation of every wiggle --
      but treat it with the same small-sample caution.

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
from src.hallucinations.perturbations import PERTURBATION_NAMES
from src.detection.phs import compute_phs_components, compute_relative_error, PHS_COMPONENT_NAMES

# Extends BELOW the canonical EPSILON_VALUES floor (0.0001, per perturbations.py) rather than
# overlapping it -- the canonical sweep already covers 0.0001-0.01, so this picks up from there
# downward, for anyone curious whether the boundary moves further once epsilon gets smaller still.
SENSITIVITY_EPSILON_VALUES = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]

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
    Loads a trained BaselinePINN checkpoint for a single case. Duplicated
    from evaluate_phs.py's load_model rather than imported, matching that
    file's own reasoning for not sharing it: keeping each script's I/O
    self-contained.

    Inputs:
        case_id (str): The case identifier (e.g. "case_00").
        k (float): The case's wavenumber, required by the BaselinePINN constructor.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        model (nn.Module): The loaded, float64, eval-mode model on `device`.
    """
    model_path = project_root / "models" / f"{case_id}_best.pth"
    model = BaselinePINN(k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.to(torch.float64)
    model.eval()
    return model


def probe_sensitivity(case_ids: list, case_meta_by_id: dict, models_dir: Path, args) -> pd.DataFrame:
    """
    For every case in `case_ids`, computes PHS components AND relative L2
    error at every SENSITIVITY_EPSILON_VALUES x PERTURBATION_NAMES
    combination.

    Inputs:
        case_ids (list[str]): Which cases to probe.
        case_meta_by_id (dict): Output of load_case_metadata().
        models_dir (Path): Directory containing {case_id}_best.pth files.
        args (argparse.Namespace): Parsed CLI arguments.

    Outputs:
        pd.DataFrame: One row per (case_id, perturbation_type, epsilon),
            with raw components, "relative_error", "case_id",
            "perturbation_type", "epsilon".
    """
    rows = []
    for case_id in case_ids:
        model_path = models_dir / f"{case_id}_best.pth"
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
              f"{len(SENSITIVITY_EPSILON_VALUES)} epsilons...")
        for perturbation_name in PERTURBATION_NAMES:
            for epsilon in SENSITIVITY_EPSILON_VALUES:
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
        tau (float): Score4_PHS_full's threshold, from the same file.

    Outputs:
        pd.DataFrame: `df` with "Score4_PHS_full" and "detected" columns
            appended (copy).
    """
    out = df.copy()
    for c in PHS_COMPONENT_NAMES:
        out[f"{c}_bar"] = out[c] / (normalizers[c] + 1e-12)
    out["Score4_PHS_full"] = sum(out[f"{c}_bar"] for c in PHS_COMPONENT_NAMES)
    out["detected"] = out["Score4_PHS_full"] > tau
    return out


def _binned_recall(df: pd.DataFrame, x_col: str, n_bins: int = 15, min_points_per_bin: int = 4) -> tuple:
    """
    Bins `df` by `x_col` into EQUAL-COUNT (quantile) bins and returns each
    bin's mean detection rate against its own mean x-value, sorted by x.

    WHY THIS EXISTS: "epsilon" is a small, shared, discrete grid (the same
    SENSITIVITY_EPSILON_VALUES for every perturbation type), so grouping by
    its exact value already pools multiple (case, perturbation) rows
    together meaningfully. "relative_error" is NOT shared like that --
    different perturbation types produce different relative_error values
    at the same epsilon (that's the whole point of using it as a
    cross-type-comparable axis), so relative_error values essentially
    never exactly match across rows. Grouping by exact value there doesn't
    pool anything -- it just returns one point per row, and adjacent
    points in sorted order can come from unrelated perturbation
    types/epsilons, making the "crossing" interpolation in
    find_boundary_crossings meaningless (confirmed: the first version of
    this analysis reported the same relative-error value for both 50% and
    90% recall, which should be impossible for a monotonic-ish curve --
    tracing it back showed this exact-groupby issue).

    WHY EQUAL-COUNT BINS, NOT EQUAL-WIDTH LOG BINS (a second, later fix):
    the first binned version used equal-width bins in log-x space, which
    fixed the grouping problem above but introduced a new, subtler one --
    equal-width bins have no knowledge of where the data actually
    clusters, so they occasionally slice through a tight cluster of
    near-identical x-values and isolate one point alone in its own bin.
    Confirmed directly: one such bin contained exactly n=1 point (case_26,
    "boundary", eps=0.0002) whose relative_error (4.5e-5) was almost
    identical to 4 OTHER rows (the same perturbation/epsilon for the other
    4 test cases, at 3.8e-5 to 4.1e-5) that landed in the adjacent bin
    purely because of where the fixed bin edge happened to fall -- with
    n=1, that bin could only ever read 0% or 100%, producing a visible
    "jerk" in the curve that had nothing to do with a real detection
    effect. Equal-COUNT bins (splitting the SORTED data into n_bins
    contiguous, similarly-sized chunks) make a lone-point bin possible
    only when there are fewer total rows than n_bins, not as a routine
    side effect of wherever bin edges happen to land.

    Inputs:
        df (pd.DataFrame): Must have `x_col` and "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        n_bins (int): Maximum number of equal-count bins. The actual
            number used is capped so each bin averages at least
            min_points_per_bin rows -- see min_points_per_bin.
        min_points_per_bin (int): Minimum average rows per bin. A group
            with few rows (e.g. one perturbation type's own subset, 1/5th
            of "overall"'s row count) automatically gets fewer, wider bins
            instead of reusing n_bins=15 regardless of how little data
            backs each one -- confirmed necessary in practice: even after
            switching to equal-count bins (see the docstring above), a
            single perturbation type's ~35 rows split into 15 bins still
            averaged ~2 rows/bin, which reads as a near-coin-flip 0%/33%/
            50%/67%/100% by chance alone and produced a visibly jagged
            per-type curve even though the pooled "overall" curve (5x the
            rows) was smooth. This ties bin width to how much data is
            actually available, per group, rather than a single fixed count.

    Outputs:
        (bin_centers, mean_recall): both np.ndarray, sorted by bin_centers.
        Returns fewer than n_bins points if there isn't enough data to
        fill them all.
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
    Finds where BINNED mean detection rate crosses each level in `levels`
    (see _binned_recall for why binning, not an exact-value groupby, is
    what makes this meaningful for a continuous x_col like
    "relative_error"), via linear interpolation between bin centers in
    log-x space -- computed on the pooled (all perturbation types, all
    cases) curve, plus the same per perturbation type.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", x_col, "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        levels (list[float]): Recall levels to find crossings for (e.g. [0.5, 0.9]).
        n_bins (int): Forwarded to _binned_recall.

    Outputs:
        dict: {"overall": {level: x_value_or_None, ...},
               perturbation_name: {level: x_value_or_None, ...}, ...}
    """
    def crossings_for(group):
        xs, ys = _binned_recall(group, x_col, n_bins)
        result = {}
        for level in levels:
            crossing = None
            for i in range(len(ys) - 1):
                if (ys[i] < level) != (ys[i + 1] < level):  # sign change around `level`
                    log_x0, log_x1 = np.log10(xs[i]), np.log10(xs[i + 1])
                    frac = (level - ys[i]) / (ys[i + 1] - ys[i])
                    crossing = float(10 ** (log_x0 + frac * (log_x1 - log_x0)))
                    break
            result[level] = crossing
        return result

    boundaries = {"overall": crossings_for(df)}
    for perturbation_name, group in df.groupby("perturbation_type"):
        boundaries[perturbation_name] = crossings_for(group)
    return boundaries


def plot_recall_vs_x(df: pd.DataFrame, x_col: str, x_label: str, output_path: Path, title: str,
                      n_bins: int = 15):
    """
    Plots BINNED mean detection rate vs. x_col (see _binned_recall), one
    line per perturbation type plus an overall pooled line, on a log-x axis.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", x_col, "detected" columns.
        x_col (str): Either "epsilon" or "relative_error".
        x_label (str): Axis label.
        output_path (Path): Where to save the figure.
        title (str): Plot title.
        n_bins (int): Forwarded to _binned_recall. Per-perturbation-type
            curves use fewer effective points than "overall" (each type
            has 1/5th the rows), so bins may be sparser for those lines.

    Outputs:
        None. Saves a PNG to output_path.
    """
    perturbation_types = sorted(df["perturbation_type"].unique())

    plt.figure(figsize=(8, 5.5))
    overall_x, overall_y = _binned_recall(df, x_col, n_bins)
    plt.plot(overall_x, overall_y, color="black", linewidth=2.5, marker="o",
              markersize=5, label="overall (pooled)", zorder=10)

    for i, perturbation_name in enumerate(perturbation_types):
        group = df[df["perturbation_type"] == perturbation_name]
        by_x, by_y = _binned_recall(group, x_col, n_bins)
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
    Entry point. Loads an existing calibration, probes the fine epsilon
    grid across every perturbation type for the requested case(s), scores
    against that calibration, finds where recall crosses 50%/90% in both
    epsilon and relative-error terms, and writes the outputs described in
    this module's docstring.

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
    tau = calibration["thresholds"]["Score4_PHS_full"]

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
    print(f"Epsilon grid: {SENSITIVITY_EPSILON_VALUES}")
    print("=" * 60)

    df = probe_sensitivity(case_ids, case_meta_by_id, models_dir, args)
    if df.empty:
        raise RuntimeError("No fields were probed -- check that models/ contains the matching *_best.pth files.")

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
            **{f"epsilon_at_{int(lvl*100)}pct_recall": boundaries_eps[key][lvl] for lvl in BOUNDARY_LEVELS},
            **{f"relative_error_at_{int(lvl*100)}pct_recall": boundaries_err[key][lvl] for lvl in BOUNDARY_LEVELS},
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
    for lvl in BOUNDARY_LEVELS:
        eps_b = boundaries_eps["overall"][lvl]
        err_b = boundaries_err["overall"][lvl]
        eps_str = f"{eps_b:.5f}" if eps_b is not None else "not reached in this range"
        err_str = f"{err_b*100:.3f}%" if err_b is not None else "not reached in this range"
        print(f"  {int(lvl*100)}% recall: epsilon ≈ {eps_str}   |   relative L2 error ≈ {err_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()