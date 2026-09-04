"""
PHS Detection Evaluation Script

For every (case_id, perturbation_type, epsilon) row listed in
data/hallucinations/hallucination_index.json (produced by
src/hallucinations/generate_hallucinations.py), this script:

  1. Loads that case's trained model once, then computes the 4 raw PHS
     components (Smom, Sdiv, Sbc, SE -- see src/detection/phs.py) for every
     one of its rows DIRECTLY from the model + perturbation functions, NOT
     from the saved data/hallucinations/*.pt bundles. Those bundles store
     detached, no-grad predictions (see generate_hallucinations.py's
     run_model_in_chunks); Smom/Sdiv need a live autograd graph for exact
     PDE residuals, so the field has to be recomputed here regardless. The
     index is still exactly what makes this script possible without
     re-deriving which (case, perturbation, epsilon, split, label) combos
     exist -- that enumeration is what generate_hallucinations.py's bundle
     is for.
     NOTE: "Sbc" here is computed via compute_boundary_localization_violation,
     NOT a literal periodicity-comparison formula -- see phs.py's module
     docstring for why that original formula was REPLACED (checked directly:
     it was essentially blind to every perturbation type in this benchmark,
     not just "boundary").
  2. Normalizes all 4 components using the VALIDATION split's clean-field
     means.
  3. Computes 3 detection scores per field -- 2 residual-only baselines plus
     PHS itself:
       Score1_momentum_only        = S_bar_mom
       Score2_momentum_divergence  = S_bar_mom + S_bar_div
       Score3_PHS_full             = S_bar_mom + S_bar_div + S_bar_bc + S_bar_E
                                     (= PHS, the official score)
  4. Selects each score's threshold tau from the 95th percentile of its
     VALIDATION-split clean-field distribution.
  5. Evaluates detection (ROC-AUC, Precision, Recall, F1 @ tau) for all 3
     scores on the held-out TEST split (target: AUC(PHS) > 0.90, ideally
     > AUC(Score2)).

Outputs:
  data/phs_scores/phs_components_raw.csv / .json
      One row per (case, perturbation, epsilon) field: raw components,
      normalized ("_bar") components, and all 3 score columns.
  plots/phs_evaluation/normalizers_and_thresholds.json
      The 4 component normalizers and the 3 scores' tau values.
  plots/phs_evaluation/detection_metrics_summary.csv / .json
      Per-score AUC / Precision / Recall / F1 on the test split.
  plots/phs_evaluation/roc_curves.png
      ROC curves for all 3 scores overlaid (test split).
  plots/phs_evaluation/score_distributions_comparison.png
      Clean vs. hallucinated scores for every score (Score1/2/3), as a
      strip plot with epsilon as categorical x-axis positions (clean, then
      each epsilon value, light-to-dark color gradient) rather than a
      pooled histogram -- shows the dose-response structure per epsilon
      directly, including exactly which epsilon's fields sit below tau.
  plots/phs_evaluation/normalized_components_vs_epsilon.png
      S_bar_mom, S_bar_div, S_bar_bc, S_bar_E (normalized -- the same
      quantities the scores are built from) vs. epsilon, one panel per
      perturbation type, on a single shared axis -- shows exactly which
      component(s) each perturbation type activates. Was raw_components_
      vs_epsilon.png (pre-normalization values) until Sbc became a
      near/far ratio and needed a different plotting approach; see
      plot_normalized_components_vs_epsilon's docstring for why normalized
      values were used instead of a secondary axis.
  plots/phs_evaluation/scores_vs_epsilon.png
      Score1 / Score2 / Score3 (PHS) vs. epsilon, one panel per
      perturbation type -- shows how adding each successive component
      changes the detection signal.

CAVEAT (historical): models committed before commit `be9b8ff` predated a phase-mismatch
bug identified during the hallucination-verification audit (see verify_hallucinations.py's
phase_amplitude_fit_diagnostic -- clean predictions matched the analytical TGV solution at
the WRONG phase). Retrained, phase-corrected models have since been committed. This script
was unaffected by that bug either way: Smom/Sdiv/Sbc are phase-invariant PDE/BC residuals,
and SE's reference curve is explicitly phase-invariant by construction (see phs.py's
compute_energy_violation) -- but Relative-L2-vs-analytical numbers from that era were not
representative and should not be cited.

Usage:
    python src/detection/evaluate_phs.py
    python src/detection/evaluate_phs.py --device cpu
    python src/detection/evaluate_phs.py --case_id case_00 --case_id case_25
    python src/detection/evaluate_phs.py --n_interior 4000 --n_time 8 --energy_res 16   # fast smoke test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

# 1. Define the project root (mirrors the convention used elsewhere in the repo)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.models.scaling import ResidualScaler
from src.physics.taylor_green import compute_nu, compute_T
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.detection.phs import (
    compute_phs_components,
    compute_normalizers,
    normalize_components,
    compute_baseline_scores,
    select_threshold,
    PHS_COMPONENT_NAMES,
    BASELINE_DEFINITIONS,
)

# Cycled through (by index, wrapping) for every multi-line plot in this module. Relying on color
# alone breaks down whenever two lines sit close together or land exactly on top of each other (e.g.
# several perturbation types legitimately tied at recall=1.0) -- distinct (linestyle, marker) pairs
# stay distinguishable even in that case, including in grayscale/colorblind-unfriendly renders.
_LINE_STYLES = [
    ("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ("-", "v"), ("--", "P"), ("-.", "X"),
]


def _styled_line(ax, x, y, index: int, label: str, **kwargs):
    """
    Plots one line using this module's shared (linestyle, marker) cycle
    (see _LINE_STYLES) plus a semi-transparent, slightly thick default
    style, so that lines which are close together -- or, for bounded
    metrics like recall, land exactly on top of each other -- stay visually
    distinguishable instead of collapsing into a single color blob.

    Inputs:
        ax: A matplotlib Axes (or the `plt` module itself) with a .plot method.
        x, y (array-like): Data to plot.
        index (int): Which entry of _LINE_STYLES to use (wraps via modulo).
        label (str): Legend label.
        **kwargs: Forwarded to .plot(), overriding the defaults below.

    Outputs:
        The Line2D object matplotlib's ax.plot() creates (useful for
        combining legends across two axes, e.g. a twinx() secondary axis --
        see plot_normalized_components_vs_epsilon). Existing callers that ignore
        the return value are unaffected.
    """
    linestyle, marker = _LINE_STYLES[index % len(_LINE_STYLES)]
    style = dict(linestyle=linestyle, marker=marker, markersize=6, linewidth=2, alpha=0.85)
    style.update(kwargs)
    line, = ax.plot(x, y, label=label, **style)
    return line


def parse_args():
    """
    Parses command-line arguments controlling which case(s) to score, the
    resolution/sampling density of each PHS component, and where outputs
    are written.

    Inputs:
        None (reads directly from sys.argv).

    Outputs:
        args (argparse.Namespace): Parsed arguments with fields
            case_id (list[str] | None), device (str), n_interior (int),
            n_bc (int), n_time (int), energy_res (int), chunk_size (int),
            percentile (float), output_dir (str).
    """
    parser = argparse.ArgumentParser(description="Evaluate the Physical Hallucination Score.")
    parser.add_argument("--case_id", action="append", default=None,
                        help="Restrict to this case_id. Repeatable (--case_id case_00 --case_id case_25). "
                             "Defaults to every case with a trained model AND an entry in the hallucination "
                             "index. NOTE: normalizer/threshold calibration needs at least one clean "
                             "VALIDATION-split field, so an arbitrary subset may fail calibration.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_interior", type=int, default=20000,
                        help="Interior collocation points sampled per field for Smom/Sdiv.")
    parser.add_argument("--n_bc", type=int, default=1000,
                        help="Paired boundary points sampled per axis (x, y) per field for Sbc.")
    parser.add_argument("--n_time", type=int, default=20,
                        help="Time slices spanning [0, T] per field for SE (matches the project's "
                             "64x64x20 evaluation-grid convention).")
    parser.add_argument("--energy_res", type=int, default=32,
                        help="Spatial grid resolution (energy_res x energy_res) per time slice for SE.")
    parser.add_argument("--chunk_size", type=int, default=8000,
                        help="Chunk size for the interior Smom/Sdiv pass (VRAM safety).")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Threshold percentile tau is drawn from.")
    parser.add_argument("--no_seed_points", action="store_true",
                        help="Disable per-case point-sampling seeding (default: ON -- every field of a "
                             "case, clean and perturbed, is evaluated at identical random points, "
                             "removing an unnecessary noise source; see phs.py's _seeded() docstring). "
                             "Only useful for reproducing pre-seeding behavior/results.")
    parser.add_argument("--bc_local_band_width_fraction", type=float, default=0.05,
                        help="Distance from an edge, as a fraction of the domain length (2*pi), "
                             "counted as 'near-boundary' for computing Sbc (see phs.py's "
                             "compute_boundary_localization_violation, which Sbc is now computed from). "
                             "Default 0.05 (5%%).")
    parser.add_argument("--bc_local_n_points", type=int, default=20000,
                        help="Interior points sampled before the near/far split for computing Sbc.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory for plots. Defaults to <project_root>/plots/phs_evaluation.")
    return parser.parse_args()


def load_model(case_id: str, k: float, device: str):
    """
    Loads a trained BaselinePINN checkpoint for a single case. Identical to
    verify_hallucinations.py's load_model, duplicated here rather than
    imported to keep src/detection/ decoupled from src/hallucinations/'s
    internal (non-library) helpers -- only phs.py's PURE formula functions
    and generate_hallucinations.py's load_case_metadata (a genuine shared
    utility) are imported across subpackages.

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


def score_all_fields(index_rows: list, case_meta_by_id: dict, models_dir: Path, args) -> pd.DataFrame:
    """
    Loads each case's model once and computes the 4 raw PHS components for
    every one of its rows in `index_rows`.

    Inputs:
        index_rows (list[dict]): Rows from hallucination_index.json,
            already filtered to the case(s) being processed.
        case_meta_by_id (dict): Output of load_case_metadata().
        models_dir (Path): Directory containing {case_id}_best.pth files.
        args (argparse.Namespace): Parsed CLI arguments (resolution knobs, device).

    Outputs:
        pd.DataFrame: One row per input index row, with the original
            columns plus "mom", "div", "bc", "E".
    """
    rows_by_case = {}
    for row in index_rows:
        rows_by_case.setdefault(row["case_id"], []).append(row)

    results = []
    for case_id, rows in sorted(rows_by_case.items()):
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

        # Same seed for every field of this case (clean + all perturbed variants) so they are
        # compared at IDENTICAL random (x, y, t) points -- see phs.py's _seeded() docstring for why
        # this removes an unnecessary source of noise between a field and its clean baseline.
        # Derived from the numeric part of case_id (e.g. "case_07" -> 1007) so it's deterministic
        # and unique per case without depending on cases_metadata.json's own (shared) "seed" field.
        case_seed = 1000 + int(case_id.split("_")[1]) if not args.no_seed_points else None

        print(f"[{case_id}] scoring {len(rows)} fields ({case_meta.get('split', 'unknown')} split)...")
        for row in rows:
            components = compute_phs_components(
                model, case_meta, nu, T, scaler,
                row["perturbation_type"], row["epsilon"],
                n_interior=args.n_interior, n_bc_per_axis=args.n_bc,
                n_time=args.n_time, energy_res=args.energy_res,
                chunk_size=args.chunk_size, device=args.device,
                bc_local_band_width_fraction=args.bc_local_band_width_fraction,
                bc_local_n_points=args.bc_local_n_points,
                seed=case_seed,
            )
            results.append({**row, **components})

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


def evaluate_detection(df: pd.DataFrame, percentile: float) -> tuple[dict, dict, list, pd.DataFrame]:
    """
    Runs the full normalize -> score -> threshold -> evaluate pipeline on
    an already-scored DataFrame.

    Inputs:
        df (pd.DataFrame): Output of score_all_fields(); must contain
            "split", "label", "mom", "div", "bc", "E" columns.
        percentile (float): Threshold percentile (95.0).

    Outputs:
        normalizers (dict): {component_name: normalizer (float)}.
        thresholds (dict): {score_name: tau (float)}.
        metrics_rows (list[dict]): One dict per score, with AUC/Precision/
            Recall/F1 on the test split.
        df (pd.DataFrame): The input df with normalized ("_bar") and score
            columns appended.
    """
    valid_val_mask = (df["split"] == "validation") & (df["label"] == "clean")
    if valid_val_mask.sum() == 0:
        raise RuntimeError(
            "No clean validation-split fields found -- cannot calibrate normalizers/threshold. "
            "If you passed --case_id, make sure at least one validation-split case is included, "
            "or omit --case_id to process every case."
        )

    normalizers = compute_normalizers(df.loc[valid_val_mask])
    df = normalize_components(df, normalizers)
    df = compute_baseline_scores(df)

    thresholds = {
        score_name: select_threshold(df.loc[valid_val_mask, score_name].values, percentile)
        for score_name in BASELINE_DEFINITIONS
    }

    test_mask = df["split"] == "test"
    if test_mask.sum() == 0:
        raise RuntimeError(
            "No test-split fields found -- cannot evaluate detection on a held-out test split. "
            "If you passed --case_id, make sure at least one test-split case is included."
        )
    y_true = (df.loc[test_mask, "label"] == "hallucinated").astype(int).values

    metrics_rows = []
    for score_name in BASELINE_DEFINITIONS:
        y_score = df.loc[test_mask, score_name].values
        y_pred = (y_score > thresholds[score_name]).astype(int)

        if len(np.unique(y_true)) < 2:
            # ROC-AUC is undefined with only one class present in the test split
            # (e.g. when running on a single --case_id). Still report P/R/F1.
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_score))

        metrics_rows.append({
            "score_name": score_name,
            "components": "+".join(BASELINE_DEFINITIONS[score_name]),
            "threshold_tau": thresholds[score_name],
            "roc_auc": auc,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "n_test_fields": int(test_mask.sum()),
            "n_test_hallucinated": int(y_true.sum()),
        })

    return normalizers, thresholds, metrics_rows, df


def plot_roc_curves(df: pd.DataFrame, output_dir: Path):
    """
    Plots ROC curves for all 3 baseline scores on the test split, overlaid
    for direct visual comparison (the AUC(PHS) > AUC(Score2) acceptance
    criterion is exactly what this figure is meant to show).

    Inputs:
        df (pd.DataFrame): Must have "split", "label", and score columns
            (post evaluate_detection()).
        output_dir (Path): Where to save roc_curves.png.

    Outputs:
        None. Saves plots/phs_evaluation/roc_curves.png.
    """
    test_mask = df["split"] == "test"
    y_true = (df.loc[test_mask, "label"] == "hallucinated").astype(int).values
    if len(np.unique(y_true)) < 2:
        print("⏭️  Skipping roc_curves.png: test split has only one class present.")
        return

    plt.figure(figsize=(6.5, 6))
    score_names = list(BASELINE_DEFINITIONS)  # all 3, always computed
    for score_name in score_names:
        y_score = df.loc[test_mask, score_name].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{score_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PHS Detection: ROC Curves (test split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()


def diagnose_misclassifications(df: pd.DataFrame, thresholds: dict, score_name: str = "Score3_PHS_full") -> pd.DataFrame:
    """
    Builds a diagnostic table of every TEST-split field the given score got
    wrong at its calibrated threshold: hallucinated fields that "slipped
    through" (scored below tau -- false negatives) and clean fields that
    were flagged anyway (false positives). Sorted by margin-to-tau
    (closest call first) so the most-instructive borderline cases are at
    the top.

    Inputs:
        df (pd.DataFrame): Post evaluate_detection() -- must have "split",
            "label", "case_id", "perturbation_type", "epsilon", and the
            requested score column.
        thresholds (dict): Output of evaluate_detection() -- {score_name: tau}.
        score_name (str): Which score to diagnose. Defaults to Score3_PHS_full.

    Outputs:
        pd.DataFrame: Columns "case_id", "perturbation_type", "epsilon",
            "label", score_name, "tau", "margin" (score - tau; negative
            for false negatives, positive for false positives), and
            "error_type" ("false_negative" or "false_positive"), sorted by
            |margin| ascending.
    """
    tau = thresholds[score_name]
    test_df = df[df["split"] == "test"]

    fn = test_df[(test_df["label"] == "hallucinated") & (test_df[score_name] < tau)].copy()
    fn["error_type"] = "false_negative"
    fp = test_df[(test_df["label"] == "clean") & (test_df[score_name] > tau)].copy()
    fp["error_type"] = "false_positive"

    cols = ["case_id", "perturbation_type", "epsilon", "label", score_name, "error_type"]
    out = pd.concat([fn[cols], fp[cols]], ignore_index=True)
    out["tau"] = tau
    out["margin"] = out[score_name] - tau
    return out.reindex(out["margin"].abs().sort_values().index).reset_index(drop=True)


def plot_score_distributions_comparison(df: pd.DataFrame, thresholds: dict, output_dir: Path):
    """
    Plots clean vs. hallucinated scores for ALL baseline scores (Score1-3)
    side by side, one panel per score, as a STRIP PLOT (jittered individual
    points) with epsilon as categorical x-axis positions, rather than a
    pooled histogram.

    WHY THIS REPLACED A POOLED HISTOGRAM: the previous version pooled all
    epsilon values into one "hallucinated" bucket per panel, which hid
    exactly the structure that matters most once EPSILON_VALUES was
    rebuilt around the detection boundary (see the README's Findings
    section) -- a wide, spread-out mass spanning orders of magnitude, with
    no visual indication that "easy" (large epsilon) and "hard" (small
    epsilon) fields were mixed together, which is exactly what made the
    false-negative mass hard to interpret at a glance. Splitting by
    epsilon as separate x-axis categories, with a light-to-dark color
    gradient for increasing epsilon and a distinct color for the clean
    baseline, shows the dose-response structure directly: each epsilon's
    cluster of points relative to tau is a visual recall readout for that
    epsilon specifically, not just a number in recall_by_type.png.

    A strip plot (not a histogram, box plot, or violin) was chosen because
    the sample sizes here are small (n=5 clean, n=25 hallucinated per
    epsilon) -- a violin plot's smoothed density estimate can visually
    oversell how much data backs it at this size, and a histogram bins
    away exactly the individual-field detail that matters when n is this
    small. Showing every real point, jittered only to avoid exact overlap,
    plus a short median marker per group, is the most honest
    representation of data this size.

    Inputs:
        df (pd.DataFrame): Must have "split", "label", "epsilon", and
            every score column in BASELINE_DEFINITIONS.
        thresholds (dict): Output of evaluate_detection(); tau per score.
        output_dir (Path): Where to save score_distributions_comparison.png.

    Outputs:
        None. Saves plots/phs_evaluation/score_distributions_comparison.png.
    """
    test_df = df[df["split"] == "test"]
    score_names = list(BASELINE_DEFINITIONS)  # all 3, always computed
    if not score_names:
        print("⏭️  Skipping score_distributions_comparison.png: no score columns found.")
        return

    epsilon_values = sorted(test_df.loc[test_df["label"] == "hallucinated", "epsilon"].unique())
    categories = ["clean"] + [f"{eps:g}" for eps in epsilon_values]
    n_categories = len(categories)

    # Light-to-dark gradient for increasing epsilon; clean gets a completely separate, fixed color
    # (steelblue/tab:blue) so it is never confusable with any hallucinated shade regardless of
    # colormap choice or how many epsilon values there are.
    cmap = plt.cm.Reds
    n_eps = max(len(epsilon_values), 1)
    epsilon_colors = [cmap(0.35 + 0.55 * i / max(n_eps - 1, 1)) for i in range(n_eps)]
    category_colors = ["tab:blue"] + epsilon_colors

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(score_names), figsize=(5.5 * len(score_names), 5.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, score_name in zip(axes, score_names):
        clean_scores = test_df.loc[test_df["label"] == "clean", score_name].values
        all_positive = test_df.loc[test_df[score_name] > 0, score_name].values
        floor = all_positive.min() if len(all_positive) else 1e-6

        # x=0 is "clean"; x=1..N are the epsilon categories in increasing order
        jitter = rng.uniform(-0.18, 0.18, size=len(clean_scores))
        ax.scatter(jitter, np.clip(clean_scores, floor, None),
                   color=category_colors[0], alpha=0.8, s=35, edgecolor="none", zorder=3)
        if len(clean_scores) > 0:
            ax.hlines(np.median(clean_scores), -0.3, 0.3, color=category_colors[0], linewidth=2.5, zorder=4)

        for i, eps in enumerate(epsilon_values):
            eps_scores = test_df.loc[(test_df["label"] == "hallucinated") & (test_df["epsilon"] == eps),
                                       score_name].values
            x_pos = i + 1
            jitter = rng.uniform(-0.18, 0.18, size=len(eps_scores))
            ax.scatter(x_pos + jitter, np.clip(eps_scores, floor, None),
                       color=category_colors[i + 1], alpha=0.7, s=25, edgecolor="none", zorder=2)
            if len(eps_scores) > 0:
                ax.hlines(np.median(eps_scores), x_pos - 0.3, x_pos + 0.3,
                          color=category_colors[i + 1], linewidth=2.5, zorder=4)

        ax.axhline(thresholds[score_name], color="black", linestyle="--", linewidth=1.3,
                   label=f"tau={thresholds[score_name]:.2f}", zorder=5)
        ax.set_yscale("log")
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_xlim(-0.6, n_categories - 1 + 0.6)
        ax.set_title(score_name, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, axis="y", which="both", alpha=0.25)

    axes[0].set_ylabel("Score value (log scale)")
    fig.suptitle("Score Distributions by Epsilon: Clean vs. Hallucinated, All Baselines Compared (test split)")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions_comparison.png", dpi=150)
    plt.close()


def evaluate_detection_by_perturbation_type(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Breaks recall down by (perturbation_type, epsilon) on the test split,
    at the already-calibrated thresholds -- i.e. "of the hallucinated
    fields of THIS type at THIS epsilon, what fraction did each score
    correctly flag as positive?" Unlike the pooled AUC/Precision/Recall/F1
    in evaluate_detection(), this exposes whether one perturbation type is
    systematically harder to detect than the others, rather than averaging
    that difference away.

    Inputs:
        df (pd.DataFrame): Post evaluate_detection() -- must have "split",
            "label", "perturbation_type", "epsilon", and the 3 score columns.
        thresholds (dict): Output of evaluate_detection() -- {score_name: tau}.

    Outputs:
        pd.DataFrame: One row per (perturbation_type, epsilon, score_name),
            with columns "n", "n_detected", "recall".
    """
    test_halluc = df[(df["split"] == "test") & (df["label"] == "hallucinated")]

    rows = []
    for (perturbation_name, epsilon), group in test_halluc.groupby(["perturbation_type", "epsilon"]):
        for score_name, tau in thresholds.items():
            n_detected = int((group[score_name] > tau).sum())
            rows.append({
                "perturbation_type": perturbation_name,
                "epsilon": epsilon,
                "score_name": score_name,
                "n": len(group),
                "n_detected": n_detected,
                "recall": n_detected / len(group),
            })
    return pd.DataFrame(rows)


def plot_recall_by_type(recall_df: pd.DataFrame, output_dir: Path):
    """
    Plots per-perturbation-type recall (Score3_PHS_full only, at tau) vs.
    epsilon, so any systematically under-detected perturbation type is
    immediately visible as a curve sitting below the others rather than
    hidden inside a single pooled recall number.

    Lines are plotted at their TRUE, unmodified recall values -- no
    vertical offset -- with a standard legend, matching every other
    multi-line plot in this module (_styled_line's distinct linestyle/
    marker cycle is what keeps types identifiable when several are tied
    at recall=1.0 and overlap exactly). An earlier version used
    per-type colored callout labels with leader lines instead of a
    legend specifically to guarantee zero label-to-label overlap; reverted
    in favor of a plain legend for consistency with the rest of the
    module's plots.

    Inputs:
        recall_df (pd.DataFrame): Output of evaluate_detection_by_perturbation_type().
        output_dir (Path): Where to save recall_by_type.png.

    Outputs:
        None. Saves plots/phs_evaluation/recall_by_type.png.
    """
    phs_recall = recall_df[recall_df["score_name"] == "Score3_PHS_full"]
    perturbation_types = sorted(phs_recall["perturbation_type"].unique())

    plt.figure(figsize=(8, 5.5))
    for i, perturbation_name in enumerate(perturbation_types):
        group = phs_recall[phs_recall["perturbation_type"] == perturbation_name].sort_values("epsilon")
        _styled_line(plt, group["epsilon"], group["recall"], i, perturbation_name)

    plt.xlabel("Epsilon")
    plt.ylabel("Recall (PHS, at tau)")
    plt.title("Detection Recall by Perturbation Type and Epsilon (test split)")
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_by_type.png", dpi=150)
    plt.close()



def plot_normalized_components_vs_epsilon(df: pd.DataFrame, output_dir: Path):
    """
    Plots all 4 NORMALIZED components (S_bar_mom, S_bar_div, S_bar_bc,
    S_bar_E -- the "_bar" quantities that actually get summed into the
    scores, S_bar_j = Sj / mean(Sj_clean_validation)) vs.
    epsilon, one subplot per perturbation type, on a SINGLE shared axis --
    so you can see exactly which component(s) each perturbation type
    activates, in the same units the scoring itself uses.

    HISTORY: an earlier version plotted the RAW (pre-normalization)
    components instead, which needed a secondary y-axis for "bc"
    specifically once it became a near/far RATIO -- a ratio of two
    similarly-tiny residuals is naturally of order 1, roughly six orders
    of magnitude larger than Smom/Sdiv/SE's raw MSE-style values (checked
    directly: bc's clean-baseline normalizer is ~1.39 vs. ~1e-6 to 1e-7
    for the other three). Switched to normalized values instead of
    maintaining two axes: every "_bar" component is, by construction,
    ~1.0 at its own clean baseline (S_bar = S / mean(S_clean)), so all 4
    become directly comparable on one shared axis without needing any
    special-casing for "bc" -- this is a more direct fix than the
    secondary-axis version, and ties the plot directly to what the actual
    scores are built from, rather than an intermediate raw quantity.

    NOTE on a related, separately-investigated finding this plot makes
    visible: for perturbation types OTHER than "boundary", S_bar_bc often
    starts ELEVATED at the smallest epsilon and DECREASES as epsilon
    grows, rather than the monotonic rise the other components show. This
    is not noise -- checked directly, the smallest-epsilon value is
    essentially identical to that SAME case's own clean-field bc value
    (e.g. one case: clean=3.190, eps=0.0001 gives 3.189), meaning at tiny
    epsilon this reflects each TRAINED MODEL's own intrinsic near/far
    residual imbalance (which varies substantially case-to-case, from
    below 1 to above 3) rather than a perturbation effect. As epsilon
    grows, a globally-uniform perturbation's own contribution starts
    dominating both the near- and far-boundary residual comparably,
    diluting that baseline imbalance toward the perturbation's own
    (typically closer-to-1) near/far ratio. For "boundary" specifically,
    the perturbation's effect is concentrated enough to quickly overwhelm
    this baseline-imbalance effect instead, producing the clear rise seen
    there rather than a dip.

    Inputs:
        df (pd.DataFrame): Post evaluate_detection() -- must have
            "perturbation_type", "epsilon", "label", and the 4 normalized
            ("mom_bar", "div_bar", "bc_bar", "E_bar") columns.
        output_dir (Path): Where to save normalized_components_vs_epsilon.png.

    Outputs:
        None. Saves plots/phs_evaluation/normalized_components_vs_epsilon.png.
    """
    halluc_df = df[df["label"] == "hallucinated"]
    perturbation_types = sorted(halluc_df["perturbation_type"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, perturbation_name in zip(axes, perturbation_types):
        group = halluc_df[halluc_df["perturbation_type"] == perturbation_name]
        for i, component in enumerate(PHS_COMPONENT_NAMES):
            by_eps = group.groupby("epsilon")[f"{component}_bar"].mean().sort_index()
            _styled_line(ax, by_eps.index, by_eps.values, i, f"S̄_{component}")
        ax.set_yscale("log")
        ax.set_ylabel("Normalized component (S̄_j, log scale)")
        ax.set_title(perturbation_name)
        ax.set_xlabel("Epsilon")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(perturbation_types):]:
        ax.axis("off")

    fig.suptitle("Normalized PHS Components vs. Epsilon, by Perturbation Type (all splits)")
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_components_vs_epsilon.png", dpi=150)
    plt.close()


def plot_all_scores_vs_epsilon(df: pd.DataFrame, output_dir: Path):
    """
    Plots Score1 (momentum-only), Score2 (+divergence), and Score3/PHS
    (+boundary +energy) vs. epsilon, one subplot per perturbation type, so
    you can see how adding each successive component changes the
    detection signal's shape and magnitude for each perturbation type.
    Semi-log axis (linear epsilon,
    log value) -- the project's standard convention for epsilon-response plots.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", "epsilon",
            "label", and the 3 score columns (post evaluate_detection()).
        output_dir (Path): Where to save scores_vs_epsilon.png.

    Outputs:
        None. Saves plots/phs_evaluation/scores_vs_epsilon.png.
    """
    halluc_df = df[df["label"] == "hallucinated"]
    perturbation_types = sorted(halluc_df["perturbation_type"].unique())
    score_names = list(BASELINE_DEFINITIONS)  # all 3, always computed

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, perturbation_name in zip(axes, perturbation_types):
        group = halluc_df[halluc_df["perturbation_type"] == perturbation_name]
        for i, score_name in enumerate(score_names):
            by_eps = group.groupby("epsilon")[score_name].mean().sort_index()
            _styled_line(ax, by_eps.index, by_eps.values, i, score_name)
        ax.set_yscale("log")
        ax.set_title(perturbation_name)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Mean score")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[len(perturbation_types):]:
        ax.axis("off")

    fig.suptitle("Score1 / Score2 / Score3 (PHS) vs. Epsilon, by Perturbation Type (all splits)")
    plt.tight_layout()
    plt.savefig(output_dir / "scores_vs_epsilon.png", dpi=150)
    plt.close()


def main():
    """
    Entry point for the PHS detection evaluation. Loads the hallucination
    index, scores every field, calibrates normalizers/thresholds from the
    validation split, evaluates detection on the test split, and writes
    all tables and plots described in this module's docstring.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes data/phs_scores/phs_components_raw.csv/.json and
        plots/phs_evaluation/* (see module docstring), and prints a
        summary to stdout.
    """
    args = parse_args()

    metadata_path = project_root / "data" / "cases_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata at {metadata_path}. Run src/data/sampler.py first.")
    case_meta_by_id = load_case_metadata(metadata_path)

    index_path = project_root / "data" / "hallucinations" / "hallucination_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Cannot find {index_path}. Run src/hallucinations/generate_hallucinations.py "
            "first to produce the dataset manifest this script scores."
        )
    with open(index_path, "r") as f:
        index_rows = json.load(f)

    if args.case_id:
        index_rows = [r for r in index_rows if r["case_id"] in set(args.case_id)]
        if not index_rows:
            raise ValueError(f"No hallucination_index rows found for case_id(s): {args.case_id}")

    models_dir = project_root / "models"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots" / "phs_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "phs_scores"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔍 EVALUATING PHYSICAL HALLUCINATION SCORE")
    print(f"Fields to score: {len(index_rows)}")
    print("=" * 60)

    df = score_all_fields(index_rows, case_meta_by_id, models_dir, args)
    if df.empty:
        raise RuntimeError("No fields were scored -- check that models/ contains the matching *_best.pth files.")

    normalizers, thresholds, metrics_rows, df = evaluate_detection(df, args.percentile)

    # --- Persist raw + normalized + scored table ---
    raw_csv_path = data_dir / "phs_components_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    raw_json_path = data_dir / "phs_components_raw.json"
    df.to_json(raw_json_path, orient="records", indent=2)
    print(f"\n💾 Wrote {raw_csv_path.relative_to(project_root)}")
    print(f"💾 Wrote {raw_json_path.relative_to(project_root)}")

    # --- Persist normalizers/thresholds ---
    calibration = {"normalizers": normalizers, "thresholds": thresholds, "percentile": args.percentile}
    with open(output_dir / "normalizers_and_thresholds.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"💾 Wrote {(output_dir / 'normalizers_and_thresholds.json').relative_to(project_root)}")

    # --- Persist detection metrics ---
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "detection_metrics_summary.csv", index=False)
    with open(output_dir / "detection_metrics_summary.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)
    print(f"💾 Wrote {(output_dir / 'detection_metrics_summary.csv').relative_to(project_root)}")

    # --- Detection metrics broken down by perturbation type ---
    recall_by_type_df = evaluate_detection_by_perturbation_type(df, thresholds)
    recall_by_type_df.to_csv(output_dir / "recall_by_perturbation_type.csv", index=False)
    print(f"💾 Wrote {(output_dir / 'recall_by_perturbation_type.csv').relative_to(project_root)}")

    # --- Diagnostic: which specific fields did PHS get wrong, and by how much? ---
    misclass_df = diagnose_misclassifications(df, thresholds)
    misclass_df.to_csv(output_dir / "misclassified_fields.csv", index=False)
    print(f"💾 Wrote {(output_dir / 'misclassified_fields.csv').relative_to(project_root)}")
    if len(misclass_df) > 0:
        print(f"\n🔎 {len(misclass_df)} misclassified test field(s) at tau={thresholds['Score3_PHS_full']:.3f} "
              f"(closest calls first):")
        print(misclass_df.to_string(index=False))
    else:
        print("\n🔎 No misclassified test fields at the calibrated threshold.")

    # --- Plots ---
    plot_roc_curves(df, output_dir)
    plot_score_distributions_comparison(df, thresholds, output_dir)
    plot_normalized_components_vs_epsilon(df, output_dir)
    plot_all_scores_vs_epsilon(df, output_dir)
    plot_recall_by_type(recall_by_type_df, output_dir)
    print(f"🖼️  Wrote roc_curves.png, score_distributions_comparison.png, "
          f"normalized_components_vs_epsilon.png, scores_vs_epsilon.png, recall_by_type.png to "
          f"{output_dir.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print("✅ Detection summary (test split):")
    for m in metrics_rows:
        auc_str = f"{m['roc_auc']:.3f}" if not np.isnan(m["roc_auc"]) else "N/A (single class)"
        print(f"  {m['score_name']:28s} AUC={auc_str:>18s}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()