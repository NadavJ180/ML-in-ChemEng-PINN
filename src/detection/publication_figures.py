"""
Publication Figures & Tables (Issue #12: Figures 1-3, Issue #13: Tables 1-2)

Generates the exact deliverables WP6 asks for, per the write-up's Section
9.6:

    Figure 1: Valid field, hallucinated field, difference map, and
              residual heatmap.
    Figure 2: Histogram of log10(PHS) and ROC curve.
    Figure 3: Violation signature heatmap showing perturbation types
              versus score components.
    Table 1:  Experimental setup.
    Table 2:  Detection results comparing momentum-only, momentum plus
              divergence, and full PHS.

WHY A SEPARATE SCRIPT rather than extending verify_hallucinations.py or
evaluate_phs.py: those two produce EXHAUSTIVE diagnostic material (every
perturbation type, every epsilon, multiple views per case) for verifying
and debugging the pipeline -- exactly what Issues #9/#10 needed. WP6 asks
for a small, CURATED set of illustrative figures for a paper, sized and
styled for print rather than on-screen inspection. Keeping this separate
means the diagnostic scripts stay focused on exhaustive verification and
this one stays focused on "what actually goes in the PDF," without either
job compromising the other's defaults.

IEEE double-column sizing conventions used throughout (see IEEE_* constants
below): IEEE_COL_WIDTH (3.5in) is a single-column figure width;
IEEE_PAGE_WIDTH (7.16in) spans both columns. Font sizes are set explicitly
(8-9pt) rather than left at matplotlib defaults (10-12pt), which look
oversized once a figure is scaled down to print at these physical
dimensions. Figure 1 (4 side-by-side panels) and Figure 3 (a heatmap with
long perturbation-type labels) both use the double-column width; Figure 2
(2 panels) fits comfortably at double-column width too, at a shorter height.

Deliberately reuses rather than duplicates: Table 2 and Figure 2's ROC
panel read directly from evaluate_phs.py's own outputs
(detection_metrics_summary.csv, phs_components_raw.csv) rather than
recomputing anything, so they can never drift from the numbers
evaluate_phs.py itself reports. Figure 1's clean/hallucinated/diff panels
and residual computation reuse the exact same grid convention as
verify_hallucinations.py's build_interior_grid (res=96, time_frac=0.5) and
the same compute_residuals/apply_perturbation functions everything else in
this project uses, so "the residual" means the same thing here as
everywhere else.

Outputs (all under plots/paper_figures/):
  figure1_valid_hallucinated_diff_residual.png
  figure2_phs_histogram_and_roc.png
  figure3_violation_signature_heatmap.png
  table1_experimental_setup.csv / .tex
  table2_detection_results.csv / .tex

Usage:
    python src/detection/publication_figures.py
    python src/detection/publication_figures.py --case_id case_00 --perturbation_type velocity_divergence --epsilon 0.001
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, roc_auc_score

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.pinn import BaselinePINN
from src.models.scaling import ResidualScaler
from src.physics.taylor_green import compute_nu, compute_T
from src.physics.navier_stokes import compute_residuals
from src.hallucinations.generate_hallucinations import load_case_metadata
from src.hallucinations.perturbations import apply_perturbation, PERTURBATION_NAMES, EPSILON_VALUES
from src.detection.phs import PHS_COMPONENT_NAMES, BASELINE_DEFINITIONS

# IEEE double-column page conventions (inches). A single column is ~3.5in wide; the full text
# width spanning both columns is ~7.16in. Font sizes are set explicitly (see FIGURE_FONT_SIZE)
# since matplotlib's defaults look oversized once scaled down to these physical dimensions.
IEEE_COL_WIDTH = 3.5
IEEE_PAGE_WIDTH = 7.16
FIGURE_FONT_SIZE = 8
FIGURE_DPI = 300  # print-quality; matches typical IEEE submission requirements


def parse_args():
    """
    Parses command-line arguments selecting the case/perturbation/epsilon
    used for Figure 1's illustrative example, and where prior evaluate_phs.py
    outputs (needed for Figures 2-3 and Table 2) are read from.

    Inputs:
        None (reads directly from sys.argv).

    Outputs:
        args (argparse.Namespace).
    """
    parser = argparse.ArgumentParser(description="Generate WP6's publication figures and tables.")
    parser.add_argument("--case_id", type=str, default="case_00",
                        help="Case used for Figure 1's illustrative example.")
    parser.add_argument("--perturbation_type", type=str, default="velocity_divergence",
                        choices=PERTURBATION_NAMES,
                        help="Perturbation type used for Figure 1's illustrative example.")
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="Epsilon used for Figure 1's illustrative example -- chosen by default "
                             "to sit in the 'climbing' region of the detection curve (visually subtle, "
                             "but clearly detectable), rather than the extremes.")
    parser.add_argument("--res", type=int, default=96,
                        help="Grid resolution for Figure 1, matching verify_hallucinations.py's default.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phs_scores_dir", type=str, default=None,
                        help="Directory containing evaluate_phs.py's phs_components_raw.csv and "
                             "detection_metrics_summary.csv. Defaults to <project_root>/data/phs_scores "
                             "and <project_root>/plots/phs_evaluation respectively.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write figures/tables. Defaults to <project_root>/plots/paper_figures.")
    return parser.parse_args()


def load_model(case_id: str, k: float, device: str):
    """
    Loads a trained BaselinePINN checkpoint for a single case. Duplicated
    from evaluate_phs.py's load_model rather than imported, matching that
    file's own reasoning for keeping each script's I/O self-contained.

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


def build_grid(T: float, res: int, time_frac: float, device: str):
    """
    Builds a square, gradient-tracked (x, y) grid at a single fixed time
    slice. Identical convention to verify_hallucinations.py's
    build_interior_grid (same defaults: res=96, time_frac=0.5), duplicated
    here rather than imported to keep this script's I/O self-contained,
    matching the project's established convention for these small,
    frequently-reused grid builders.

    Inputs:
        T (float): The case's final simulation time.
        res (int): Grid resolution (res x res points).
        time_frac (float): Fraction of T at which the snapshot is taken.
        device (str): Target hardware device ('cuda' or 'cpu').

    Outputs:
        x, y, t (torch.Tensor): Flattened leaf tensors of shape (res*res, 1),
                                 each with requires_grad=True, dtype float64.
    """
    t_val = time_frac * T
    x_lin = torch.linspace(0, 2 * torch.pi, res, dtype=torch.float64)
    y_lin = torch.linspace(0, 2 * torch.pi, res, dtype=torch.float64)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")

    x = X.reshape(-1, 1).to(device).clone().requires_grad_(True)
    y = Y.reshape(-1, 1).to(device).clone().requires_grad_(True)
    t = torch.full_like(x, t_val).clone().requires_grad_(True)
    return x, y, t


def make_figure1(args, case_meta_by_id, output_dir):
    """
    Figure 1: Valid field, hallucinated field, difference map, and residual
    heatmap -- one representative (case, perturbation, epsilon) example.

    Uses the u-velocity component for the valid/hallucinated/difference
    panels (matching what verify_hallucinations.py's contour checks already
    show), and the momentum residual magnitude sqrt(Ru^2 + Rv^2) for the
    4th panel -- the piece that was missing entirely from every existing
    plot in the repo (residual STATISTICS were already computed everywhere,
    but never rendered as a spatial map).

    Inputs:
        args (argparse.Namespace): Must have case_id, perturbation_type,
            epsilon, res, device.
        case_meta_by_id (dict): Output of load_case_metadata().
        output_dir (Path): Where to save the figure.

    Outputs:
        None. Saves figure1_valid_hallucinated_diff_residual.png.
    """
    case_meta = case_meta_by_id[args.case_id]
    U0, Re, k = case_meta["U0"], case_meta["Re"], case_meta["k"]
    nu, T = compute_nu(U0, Re, k), compute_T(U0, Re, k)
    model = load_model(args.case_id, k, args.device)

    x, y, t = build_grid(T, args.res, 0.5, args.device)
    coords = {"x": x, "y": y, "t": t}
    params = {"U0": U0, "k": k, "T": T}

    preds = model(torch.cat([x, y, t], dim=1))
    clean = {"u": preds[:, 0:1], "v": preds[:, 1:2], "p": preds[:, 2:3]}
    perturbed = apply_perturbation(args.perturbation_type, clean, coords, params, args.epsilon,
                                     model=model, no_grad=False)

    Ru, Rv, _ = compute_residuals(perturbed["u"], perturbed["v"], perturbed["p"], x, y, t, nu)
    residual_magnitude = torch.sqrt(Ru ** 2 + Rv ** 2).detach().cpu().numpy().reshape(args.res, args.res)

    u_clean = clean["u"].detach().cpu().numpy().reshape(args.res, args.res)
    u_halluc = perturbed["u"].detach().cpu().numpy().reshape(args.res, args.res)
    u_diff = np.abs(u_halluc - u_clean)

    plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})
    fig, axes = plt.subplots(1, 4, figsize=(IEEE_PAGE_WIDTH, IEEE_PAGE_WIDTH / 4 + 0.5))

    panels = [
        (u_clean, "(a) Valid field (u)", "RdBu_r", None),
        (u_halluc, "(b) Hallucinated field (u)", "RdBu_r", None),
        (u_diff, "(c) |Difference|", "inferno", None),
        (residual_magnitude, "(d) Residual |R|", "inferno", LogNorm()),
    ]
    for ax, (data, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(data.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                        cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(title, fontsize=FIGURE_FONT_SIZE)
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{args.case_id}, {args.perturbation_type}, \u03b5={args.epsilon:g}", fontsize=FIGURE_FONT_SIZE + 1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_valid_hallucinated_diff_residual.png", dpi=FIGURE_DPI)
    plt.close()


def make_figure2(phs_df, output_dir):
    """
    Figure 2: Histogram of log10(PHS) and ROC curve -- both restricted to
    Score4_PHS_full specifically (the official PHS), matching the
    write-up's singular "the PHS" framing. The fuller multi-score ROC
    comparison (Score1-4 overlaid) remains available separately in
    evaluate_phs.py's own roc_curves.png for the ablation discussion --
    this figure is the clean, single-score headline version for the paper.

    Inputs:
        phs_df (pd.DataFrame): evaluate_phs.py's phs_components_raw.csv,
            already scored (must have "split", "label", "Score4_PHS_full").
        output_dir (Path): Where to save the figure.

    Outputs:
        None. Saves figure2_phs_histogram_and_roc.png.
    """
    test_df = phs_df[phs_df["split"] == "test"]
    clean_scores = test_df.loc[test_df["label"] == "clean", "Score4_PHS_full"].values
    halluc_scores = test_df.loc[test_df["label"] == "hallucinated", "Score4_PHS_full"].values
    y_true = (test_df["label"] == "hallucinated").astype(int).values
    y_score = test_df["Score4_PHS_full"].values

    plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, IEEE_PAGE_WIDTH / 2.4))

    # Panel (a): histogram of log10(PHS)
    all_positive = np.concatenate([clean_scores, halluc_scores])
    all_positive = all_positive[all_positive > 0]
    log_clean = np.log10(np.clip(clean_scores, all_positive.min(), None))
    log_halluc = np.log10(np.clip(halluc_scores, all_positive.min(), None))
    bins = np.linspace(min(log_clean.min(), log_halluc.min()), max(log_clean.max(), log_halluc.max()), 25)
    axes[0].hist(log_clean, bins=bins, alpha=0.65, label=f"Clean (n={len(clean_scores)})", color="tab:blue")
    axes[0].hist(log_halluc, bins=bins, alpha=0.65, label=f"Hallucinated (n={len(halluc_scores)})", color="tab:red")
    axes[0].set_xlabel("log$_{10}$(PHS)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) PHS distribution")
    axes[0].legend(fontsize=FIGURE_FONT_SIZE - 1)

    # Panel (b): ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    axes[1].plot(fpr, tpr, color="tab:red", linewidth=1.8, label=f"PHS (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("(b) ROC curve")
    axes[1].legend(fontsize=FIGURE_FONT_SIZE - 1, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "figure2_phs_histogram_and_roc.png", dpi=FIGURE_DPI)
    plt.close()


def make_figure3(phs_df, output_dir):
    """
    Figure 3: Violation signature heatmap -- rows are perturbation types,
    columns are the 5 raw PHS components, cell value is that component's
    NORMALIZED value (the same S_bar quantity evaluate_phs.py sums into
    scores), averaged over test-split cases at the LARGEST epsilon in
    EPSILON_VALUES (the strongest, clearest signal for each cell).

    S_bar IS correctly normalized in the sense this project defines it --
    each raw Sj is divided by ITS OWN mean over clean validation fields, so
    a clean field reads ~1.0 for every component regardless of scale. That
    is a per-component, relative-to-clean normalization; it does NOT (and
    isn't meant to) equalize how STRONGLY different perturbation types
    activate different components -- a perturbation genuinely can push one
    component to ~800x its clean level while barely moving another, and
    that contrast IS the finding this figure exists to show (e.g. Sbc
    staying near 1 even for "boundary" is the blind spot; S_bar_local
    responding broadly rather than boundary-only). So the wide spread
    across cells (~1 up to ~800+) is real, not evidence of missing
    normalization to fix.

    What IS worth fixing is display: annotating cells with the raw S_bar
    number makes that same correct, wide dynamic range hard to read at a
    glance (single-digit and 3-digit numbers side by side). The color
    mapping already uses a log scale for exactly this reason; the
    annotated numbers now do too (log10(S_bar), shown in the colorbar
    label) so the ON-SCREEN numbers are visually comparable across cells,
    while the underlying data and PHS's actual score computation elsewhere
    in this project still use the untransformed S_bar -- only this
    figure's displayed digits are log-scaled, not any stored or scored value.

    Inputs:
        phs_df (pd.DataFrame): evaluate_phs.py's phs_components_raw.csv,
            already scored (must have "perturbation_type", "epsilon",
            "label", and every "{component}_bar" column).
        output_dir (Path): Where to save the figure.

    Outputs:
        None. Saves figure3_violation_signature_heatmap.png.
    """
    max_epsilon = max(EPSILON_VALUES)
    subset = phs_df[(phs_df["label"] == "hallucinated") & (phs_df["epsilon"] == max_epsilon)]

    perturbation_types = sorted(subset["perturbation_type"].unique())
    component_cols = [f"{c}_bar" for c in PHS_COMPONENT_NAMES]
    matrix = np.zeros((len(perturbation_types), len(PHS_COMPONENT_NAMES)))
    for i, pert in enumerate(perturbation_types):
        rows = subset[subset["perturbation_type"] == pert]
        for j, col in enumerate(component_cols):
            matrix[i, j] = rows[col].mean()

    plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH * 0.6, IEEE_PAGE_WIDTH * 0.42))
    im = ax.imshow(matrix, cmap="viridis", norm=LogNorm(vmin=max(matrix[matrix > 0].min(), 1e-3),
                                                          vmax=matrix.max()), aspect="auto")
    ax.set_xticks(range(len(PHS_COMPONENT_NAMES)))
    # Underscores inside an already-subscripted mathtext label (e.g. "bc_local") trigger a second,
    # nested subscript and render as a stray vertical bar -- replace with a comma for display only.
    ax.set_xticklabels([f"$\\bar{{S}}_{{{c.replace('_', ',')}}}$" for c in PHS_COMPONENT_NAMES])
    ax.set_yticks(range(len(perturbation_types)))
    ax.set_yticklabels(perturbation_types)
    ax.set_title(f"Violation signature at \u03b5={max_epsilon:g} (test split, mean $\\bar{{S}}_j$)")
    for i in range(len(perturbation_types)):
        for j in range(len(PHS_COMPONENT_NAMES)):
            # Cell text shows log10(S_bar) for readability across a ~3-order-of-magnitude range
            # (see docstring); the underlying value/color mapping is unaffected.
            log_val = np.log10(max(matrix[i, j], 1e-12))
            ax.text(j, i, f"{log_val:.1f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.max() ** 0.5 else "black", fontsize=FIGURE_FONT_SIZE - 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log$_{10}(\\bar{S}_j)$")
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_violation_signature_heatmap.png", dpi=FIGURE_DPI)
    plt.close()


def _escape_latex(text: str) -> str:
    """
    Escapes underscores for safe inclusion in a raw LaTeX table cell.
    Needed because several values written into these tables come directly
    from Python identifiers (PERTURBATION_NAMES, PHS_COMPONENT_NAMES) that
    contain underscores -- e.g. "velocity_divergence", "bc_local" -- and an
    unescaped underscore in LaTeX text mode is a reserved math-mode
    character that would either fail to compile or render as an
    unintended subscript. The CSV outputs are left with plain underscores
    (correct there; CSVs have no such reserved character), only the .tex
    outputs need this.

    Inputs:
        text (str): Raw text that may contain underscores.

    Outputs:
        str: The same text with every "_" replaced by "\\_".
    """
    return text.replace("_", "\\_")


def make_table1(case_meta_by_id, output_dir):
    """
    Table 1: Experimental setup -- summarizes the case ensemble,
    perturbation sweep, and PHS calibration in one small reference table,
    computed directly from cases_metadata.json and the project's own
    constants (PERTURBATION_NAMES, EPSILON_VALUES, PHS_COMPONENT_NAMES)
    rather than hand-typed, so it can't silently drift out of sync with
    the actual pipeline.

    Inputs:
        case_meta_by_id (dict): Output of load_case_metadata().
        output_dir (Path): Where to save the table.

    Outputs:
        None. Saves table1_experimental_setup.csv and .tex.
    """
    splits = {"train": 0, "validation": 0, "test": 0}
    re_vals, u0_vals, k_vals = [], [], []
    for meta in case_meta_by_id.values():
        splits[meta["split"]] = splits.get(meta["split"], 0) + 1
        re_vals.append(meta["Re"])
        u0_vals.append(meta["U0"])
        k_vals.append(meta["k"])

    rows = [
        ("Total cases", str(len(case_meta_by_id))),
        ("Train / Validation / Test split", f"{splits.get('train', 0)} / {splits.get('validation', 0)} / {splits.get('test', 0)}"),
        ("Reynolds number Re range", f"{min(re_vals):.1f} - {max(re_vals):.1f}"),
        ("Velocity scale U0 range", f"{min(u0_vals):.2f} - {max(u0_vals):.2f}"),
        ("Wavenumbers k used", ", ".join(str(v) for v in sorted(set(k_vals)))),
        ("Perturbation types", str(len(PERTURBATION_NAMES))),
        ("Perturbation type names", ", ".join(PERTURBATION_NAMES)),
        ("Epsilon values", ", ".join(f"{e:g}" for e in EPSILON_VALUES)),
        ("PHS components", ", ".join(PHS_COMPONENT_NAMES)),
        ("Threshold percentile", "95th (of clean validation-split PHS)"),
    ]
    df = pd.DataFrame(rows, columns=["Setting", "Value"])
    df.to_csv(output_dir / "table1_experimental_setup.csv", index=False)

    with open(output_dir / "table1_experimental_setup.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Experimental setup.}\n\\label{tab:setup}\n")
        f.write("\\begin{tabular}{ll}\n\\hline\n")
        for setting, value in rows:
            f.write(f"{_escape_latex(setting)} & {_escape_latex(value)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def make_table2(metrics_summary_path, output_dir):
    """
    Table 2: Detection results comparing momentum-only, momentum plus
    divergence, and full PHS -- read directly from evaluate_phs.py's own
    detection_metrics_summary.csv rather than recomputed, so this table
    can never disagree with the numbers evaluate_phs.py itself reports.
    Includes Score3 (without bc_local) as a 4th row beyond the write-up's
    literal 3-way ask, since it's a meaningful ablation already computed
    as a side effect of the Score4-as-PHS decision (see the README).

    Inputs:
        metrics_summary_path (Path): evaluate_phs.py's detection_metrics_summary.csv.
        output_dir (Path): Where to save the table.

    Outputs:
        None. Saves table2_detection_results.csv and .tex.
    """
    metrics = pd.read_csv(metrics_summary_path)
    display_names = {
        "Score1_momentum_only": "Momentum only",
        "Score2_momentum_divergence": "Momentum + divergence",
        "Score3_without_bc_local": "Full PHS, no bc_local (ablation)",
        "Score4_PHS_full": "Full PHS",
    }
    metrics = metrics.copy()
    metrics["Score"] = metrics["score_name"].map(display_names)
    out = metrics[["Score", "roc_auc", "precision", "recall", "f1"]].rename(
        columns={"roc_auc": "AUC", "precision": "Precision", "recall": "Recall", "f1": "F1"})
    out.to_csv(output_dir / "table2_detection_results.csv", index=False)

    with open(output_dir / "table2_detection_results.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Detection results (test split).}\n\\label{tab:detection}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Score & AUC & Precision & Recall & F1 \\\\\n\\hline\n")
        for _, row in out.iterrows():
            f.write(f"{_escape_latex(row['Score'])} & {row['AUC']:.3f} & {row['Precision']:.3f} & "
                    f"{row['Recall']:.3f} & {row['F1']:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def main():
    """
    Entry point. Loads case metadata and evaluate_phs.py's prior outputs,
    then generates all 3 figures and 2 tables described in this module's
    docstring.

    Inputs:
        None (reads parsed command-line arguments via parse_args()).

    Outputs:
        None. Writes files under plots/paper_figures/ and prints a summary.
    """
    args = parse_args()

    metadata_path = project_root / "data" / "cases_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find {metadata_path}. Run src/data/sampler.py first.")
    case_meta_by_id = load_case_metadata(metadata_path)

    phs_scores_dir = Path(args.phs_scores_dir) if args.phs_scores_dir else project_root / "data" / "phs_scores"
    phs_raw_path = phs_scores_dir / "phs_components_raw.csv"
    if not phs_raw_path.exists():
        raise FileNotFoundError(f"Cannot find {phs_raw_path}. Run src/detection/evaluate_phs.py first.")
    phs_df = pd.read_csv(phs_raw_path)

    metrics_summary_path = project_root / "plots" / "phs_evaluation" / "detection_metrics_summary.csv"
    if not metrics_summary_path.exists():
        raise FileNotFoundError(f"Cannot find {metrics_summary_path}. Run src/detection/evaluate_phs.py first.")

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📄 GENERATING PUBLICATION FIGURES & TABLES (Issues #12, #13)")
    print("=" * 60)

    make_figure1(args, case_meta_by_id, output_dir)
    print(f"🖼️  Wrote figure1_valid_hallucinated_diff_residual.png")

    make_figure2(phs_df, output_dir)
    print(f"🖼️  Wrote figure2_phs_histogram_and_roc.png")

    make_figure3(phs_df, output_dir)
    print(f"🖼️  Wrote figure3_violation_signature_heatmap.png")

    make_table1(case_meta_by_id, output_dir)
    print(f"📊 Wrote table1_experimental_setup.csv / .tex")

    make_table2(metrics_summary_path, output_dir)
    print(f"📊 Wrote table2_detection_results.csv / .tex")

    print("=" * 60)
    print(f"✅ All outputs written to {output_dir.relative_to(project_root)}")
    print("=" * 60)


if __name__ == "__main__":
    main()