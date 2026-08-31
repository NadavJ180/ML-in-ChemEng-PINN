"""
PHS Detection Evaluation Script (Section 8, WP5)

For every (case_id, perturbation_type, epsilon) row listed in
data/hallucinations/hallucination_index.json (produced by Issue #8's
src/hallucinations/generate_hallucinations.py), this script:

  1. Loads that case's trained model once, then computes the 5 raw PHS
     components (Smom, Sdiv, Sbc, S_bc_local, SE -- see src/detection/phs.py)
     for every one of its rows DIRECTLY from the model + perturbation
     functions, NOT from the saved data/hallucinations/*.pt bundles. Those
     bundles store detached, no-grad predictions (see generate_hallucinations.py's
     run_model_in_chunks); Smom/Sdiv need a live autograd graph for exact
     PDE residuals, so the field has to be recomputed here regardless. The
     index is still exactly what makes this script possible without
     re-deriving which (case, perturbation, epsilon, split, label) combos
     exist -- that enumeration is the part Issue #8's bundle was for.
  2. Normalizes all 5 components using the VALIDATION split's clean-field
     means [Section 8].
  3. Computes 4 detection scores per field -- WP5's baselines plus an
     ablation for the 5th component:
       Score1_momentum_only        = S_bar_mom
       Score2_momentum_divergence  = S_bar_mom + S_bar_div
       Score3_without_bc_local     = S_bar_mom + S_bar_div + S_bar_bc + S_bar_E
                                     (Section 8's original literal 4-term formula)
       Score4_PHS_full             = S_bar_mom + S_bar_div + S_bar_bc + S_bar_bc_local + S_bar_E
                                     (= PHS, the current official score -- see
                                     phs.py's module docstring for why bc_local
                                     was promoted from an optional add-on to a
                                     permanent 5th term)
  4. Selects each score's threshold tau from the 95th percentile of its
     VALIDATION-split clean-field distribution [Section 8].
  5. Evaluates detection (ROC-AUC, Precision, Recall, F1 @ tau) for all 4
     scores on the held-out TEST split, per WP5's acceptance criterion
     (AUC(PHS) > 0.90, ideally > AUC(Score2)) -- Score3 additionally shows
     what including bc_local specifically changes versus not.

Outputs:
  data/phs_scores/phs_components_raw.csv / .json
      One row per (case, perturbation, epsilon) field: raw components,
      normalized ("_bar") components, and all 4 score columns.
  plots/phs_evaluation/normalizers_and_thresholds.json
      The 5 component normalizers and the 4 scores' tau values.
  plots/phs_evaluation/detection_metrics_summary.csv / .json
      Per-score AUC / Precision / Recall / F1 on the test split.
  plots/phs_evaluation/roc_curves.png
      ROC curves for all 4 scores overlaid (test split).
  plots/phs_evaluation/score_distributions_comparison.png
      Clean vs. hallucinated distribution for every score (Score1/2/3/4),
      side by side with a shared x-axis, so separation quality can be
      compared directly across baselines.
  plots/phs_evaluation/phs_vs_epsilon.png
      Mean PHS vs. epsilon per perturbation type (all splits pooled) --
      sanity check that PHS increases with perturbation strength.
  plots/phs_evaluation/raw_components_vs_epsilon.png
      Smom, Sdiv, Sbc, S_bc_local, SE (raw, pre-normalization) vs. epsilon,
      one panel per perturbation type -- shows exactly which component(s)
      each perturbation type activates (e.g. makes it visible directly that
      Sbc itself stays flat for "boundary" while S_bc_local rises).
  plots/phs_evaluation/scores_vs_epsilon.png
      Score1 / Score2 / Score3 / Score4 (PHS) vs. epsilon, one panel per
      perturbation type -- shows how adding each successive component
      changes the detection signal.

CAVEAT (historical): models committed before commit `be9b8ff` predated a phase-mismatch
bug identified during Issue #9's audit (see verify_hallucinations.py's
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
        None (draws on `ax`).
    """
    linestyle, marker = _LINE_STYLES[index % len(_LINE_STYLES)]
    style = dict(linestyle=linestyle, marker=marker, markersize=6, linewidth=2, alpha=0.85)
    style.update(kwargs)
    ax.plot(x, y, label=label, **style)


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
    parser = argparse.ArgumentParser(description="Evaluate the Physical Hallucination Score (Section 8, WP5).")
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
                        help="Threshold percentile tau is drawn from, per Section 8.")
    parser.add_argument("--no_seed_points", action="store_true",
                        help="Disable per-case point-sampling seeding (default: ON -- every field of a "
                             "case, clean and perturbed, is evaluated at identical random points, "
                             "removing an unnecessary noise source; see phs.py's _seeded() docstring). "
                             "Only useful for reproducing pre-seeding behavior/results.")
    parser.add_argument("--bc_local_band_width", type=float, default=0.3,
                        help="Distance from an edge counted as 'near-boundary' for S_bc_local "
                             "(always computed -- see phs.py's compute_boundary_localization_violation).")
    parser.add_argument("--bc_local_n_points", type=int, default=20000,
                        help="Interior points sampled before the near/far split for S_bc_local.")
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
    Loads each case's model once and computes the 5 raw PHS components for
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
                bc_local_band_width=args.bc_local_band_width,
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
            "split", "label", "mom", "div", "bc", "bc_local", "E" columns.
        percentile (float): Threshold percentile, per Section 8 (95.0).

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
            "No clean validation-split fields found -- cannot calibrate normalizers/threshold "
            "(Section 8 requires them). If you passed --case_id, make sure at least one "
            "validation-split case is included, or omit --case_id to process every case."
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
            "No test-split fields found -- cannot evaluate detection (Section 9.5 requires a held-out "
            "test split). If you passed --case_id, make sure at least one test-split case is included."
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
    Plots ROC curves for all 4 baseline scores on the test split, overlaid
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
    score_names = list(BASELINE_DEFINITIONS)  # all 4, always computed
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


def diagnose_misclassifications(df: pd.DataFrame, thresholds: dict, score_name: str = "Score4_PHS_full") -> pd.DataFrame:
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
        score_name (str): Which score to diagnose. Defaults to Score4_PHS_full.

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
    Plots clean vs. hallucinated score distributions for ALL baseline
    scores (Score1, Score2, Score3, Score4/PHS) side by
    side in one figure, sharing a common log-x axis range across every
    panel so the DEGREE of separation can be compared directly panel to
    panel, not just inferred from the AUC numbers.

    This intentionally replaces having one near-identical full-size
    distribution plot per score (three or four separate files would be
    largely redundant with each other, and with the ROC curve, which
    already summarizes comparative rank-order separation numerically) --
    one compact side-by-side figure shows the same "does PHS separate
    better than the simpler baselines" story as an actual score gap you
    can see, which is complementary to the ROC curve's abstract
    TPR/FPR view rather than a duplicate of it.

    Inputs:
        df (pd.DataFrame): Must have "split", "label", and every score
            column in BASELINE_DEFINITIONS that is present.
        thresholds (dict): Output of evaluate_detection(); tau per score.
        output_dir (Path): Where to save score_distributions_comparison.png.

    Outputs:
        None. Saves plots/phs_evaluation/score_distributions_comparison.png.
    """
    test_df = df[df["split"] == "test"]
    score_names = list(BASELINE_DEFINITIONS)  # all 4, always computed
    if not score_names:
        print("⏭️  Skipping score_distributions_comparison.png: no score columns found.")
        return

    all_scores = np.concatenate([test_df[s].values for s in score_names])
    all_scores_positive = all_scores[all_scores > 0]
    if len(all_scores_positive) == 0:
        print("⏭️  Skipping score_distributions_comparison.png: all scores are zero.")
        return
    log_min = np.log10(max(all_scores_positive.min(), 1e-6))
    log_max = np.log10(max(all_scores_positive.max(), 10 ** (log_min + 1)))
    bins = np.logspace(log_min, log_max, 30)

    fig, axes = plt.subplots(1, len(score_names), figsize=(5.5 * len(score_names), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, score_name in zip(axes, score_names):
        clean_scores = test_df.loc[test_df["label"] == "clean", score_name].values
        halluc_scores = test_df.loc[test_df["label"] == "hallucinated", score_name].values

        ax.hist(np.clip(clean_scores, all_scores_positive.min(), None), bins=bins, alpha=0.6,
                label=f"Clean (n={len(clean_scores)})", color="tab:blue")
        ax.hist(np.clip(halluc_scores, all_scores_positive.min(), None), bins=bins, alpha=0.6,
                label=f"Hallucinated (n={len(halluc_scores)})", color="tab:red")
        ax.axvline(max(thresholds[score_name], all_scores_positive.min()), color="black", linestyle="--",
                   label=f"tau={thresholds[score_name]:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel(f"{score_name} (log scale)")
        ax.set_title(score_name, fontsize=10)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Score Distributions: Clean vs. Hallucinated, All Baselines Compared (test split)")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions_comparison.png", dpi=150)
    plt.close()
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
            "label", "perturbation_type", "epsilon", and the 4 score columns.
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
    Plots per-perturbation-type recall (Score4_PHS_full only, at tau) vs.
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
    phs_recall = recall_df[recall_df["score_name"] == "Score4_PHS_full"]
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


def plot_phs_vs_epsilon(df: pd.DataFrame, output_dir: Path):
    """
    Plots mean PHS vs. epsilon, one line per perturbation type (all splits
    pooled, since this is a sanity/robustness check rather than a formal
    evaluation) -- confirms the "PHS increases with perturbation strength"
    success criterion.

    Uses a semi-log axis (linear epsilon, log PHS) per project preference.
    Note EPSILON_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1] is itself an
    approximately geometric sequence (consecutive ratios of ~2-2.5), so
    the 5 points still land closer together at the low-epsilon end on a
    linear x-axis than a log-log reader might expect -- that clustering is
    inherent to the sweep design, not a plotting artifact.

    Epsilon=0 (the clean baseline) is NOT plotted as a sixth x-position:
    it isn't a value in EPSILON_VALUES for any perturbation type -- it's a
    separate label="clean" row with no perturbation_type at all. Instead,
    the clean-split mean PHS is drawn as a horizontal reference line.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", "epsilon",
            "label", "Score4_PHS_full" columns.
        output_dir (Path): Where to save phs_vs_epsilon.png.

    Outputs:
        None. Saves plots/phs_evaluation/phs_vs_epsilon.png.
    """
    halluc_df = df[df["label"] == "hallucinated"]
    clean_mean = df.loc[df["label"] == "clean", "Score4_PHS_full"].mean()
    perturbation_types = sorted(halluc_df["perturbation_type"].unique())

    plt.figure(figsize=(7, 5))
    for i, perturbation_name in enumerate(perturbation_types):
        group = halluc_df[halluc_df["perturbation_type"] == perturbation_name]
        by_eps = group.groupby("epsilon")["Score4_PHS_full"].mean().sort_index()
        _styled_line(plt, by_eps.index, by_eps.values, i, perturbation_name)

    plt.axhline(clean_mean, color="gray", linestyle="--", alpha=0.7,
                label=f"clean baseline (mean={clean_mean:.2f})")
    plt.yscale("log")
    plt.xlabel("Epsilon (perturbation strength, linear scale)")
    plt.ylabel("Mean PHS (log scale)")
    plt.title("PHS vs. Epsilon by Perturbation Type (all splits)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "phs_vs_epsilon.png", dpi=150)
    plt.close()


def plot_raw_components_vs_epsilon(df: pd.DataFrame, output_dir: Path):
    """
    Plots all 4 RAW components (Smom, Sdiv, Sbc, SE -- before normalization)
    vs. epsilon, one subplot per perturbation type, so you can see exactly
    which component(s) each perturbation type activates. This is the
    figure that makes the Sbc "boundary" blind spot visible directly: the
    "boundary" panel's bc line should sit flat while its mom/div lines
    climb. Semi-log axis (linear epsilon, log value), matching phs_vs_epsilon.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", "epsilon",
            "label", and the 5 raw component columns ("mom", "div", "bc", "bc_local", "E").
        output_dir (Path): Where to save raw_components_vs_epsilon.png.

    Outputs:
        None. Saves plots/phs_evaluation/raw_components_vs_epsilon.png.
    """
    halluc_df = df[df["label"] == "hallucinated"]
    perturbation_types = sorted(halluc_df["perturbation_type"].unique())
    component_names = PHS_COMPONENT_NAMES  # all 5, always computed

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, perturbation_name in zip(axes, perturbation_types):
        group = halluc_df[halluc_df["perturbation_type"] == perturbation_name]
        for i, component in enumerate(component_names):
            by_eps = group.groupby("epsilon")[component].mean().sort_index()
            _styled_line(ax, by_eps.index, by_eps.values, i, f"S_{component}")
        ax.set_yscale("log")
        ax.set_title(perturbation_name)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Raw component value")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(perturbation_types):]:
        ax.axis("off")

    fig.suptitle("Raw PHS Components vs. Epsilon, by Perturbation Type (all splits)")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_components_vs_epsilon.png", dpi=150)
    plt.close()


def plot_all_scores_vs_epsilon(df: pd.DataFrame, output_dir: Path):
    """
    Plots Score1 (momentum-only), Score2 (+divergence), Score3 (+boundary
    +energy, without bc_local), and Score4/PHS (+bc_local) vs. epsilon, one
    subplot per perturbation type, so you can see how adding each
    successive component changes the detection signal's shape and
    magnitude for each perturbation type. Semi-log axis (linear epsilon,
    log value), matching phs_vs_epsilon.

    Inputs:
        df (pd.DataFrame): Must have "perturbation_type", "epsilon",
            "label", and the 4 score columns (post evaluate_detection()).
        output_dir (Path): Where to save scores_vs_epsilon.png.

    Outputs:
        None. Saves plots/phs_evaluation/scores_vs_epsilon.png.
    """
    halluc_df = df[df["label"] == "hallucinated"]
    perturbation_types = sorted(halluc_df["perturbation_type"].unique())
    score_names = list(BASELINE_DEFINITIONS)  # all 4, always computed

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

    fig.suptitle("Score1 / Score2 / Score3 / Score4 (PHS) vs. Epsilon, by Perturbation Type (all splits)")
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
            "first (Issue #8) to produce the dataset manifest this script scores."
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
    print("🔍 EVALUATING PHYSICAL HALLUCINATION SCORE (Section 8, WP5)")
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
        print(f"\n🔎 {len(misclass_df)} misclassified test field(s) at tau={thresholds['Score4_PHS_full']:.3f} "
              f"(closest calls first):")
        print(misclass_df.to_string(index=False))
    else:
        print("\n🔎 No misclassified test fields at the calibrated threshold.")

    # --- Plots ---
    plot_roc_curves(df, output_dir)
    plot_score_distributions_comparison(df, thresholds, output_dir)
    plot_phs_vs_epsilon(df, output_dir)
    plot_raw_components_vs_epsilon(df, output_dir)
    plot_all_scores_vs_epsilon(df, output_dir)
    plot_recall_by_type(recall_by_type_df, output_dir)
    print(f"🖼️  Wrote roc_curves.png, score_distributions_comparison.png, phs_vs_epsilon.png, "
          f"raw_components_vs_epsilon.png, scores_vs_epsilon.png, recall_by_type.png to "
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