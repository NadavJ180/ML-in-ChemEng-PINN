"""
End-to-End Pipeline Runner

Runs every stage of a pipeline in dependency order by invoking each stage's
own script exactly as documented in the README's Usage section and in
PIPELINE.md (same module, same flags) -- this file sequences the pipeline, it
does not duplicate any of its logic.

There are two pipelines, chosen with --preset:

original (the default): the 30 old cases, from sampling and training to the
    first paper's figures and tables. See PIPELINE.md for what each stage
    produces, how expensive it is, and when it is safe to skip: sampling and
    dataset generation are deterministic (safe to skip once
    data/cases_metadata.json and data/tensors/ exist); training already skips
    any case_id that has a models/{case_id}.pth checkpoint; every stage after
    training regenerates its outputs in full each run (no skip-if-exists
    logic), which PIPELINE.md flags per stage.

final: every result of the final iteration (#46), rebuilt from the saved
    models. No stage trains: the models are read from models/ (the 30 old
    ones) and models/confirmation/ (the 210 checkpoints of the 30 new cases).
    The stages remake the new cases' tensors, the damaged fields, their scores
    with the frozen calibration and the benchmark tables (#35), the true
    errors of all 240 models (#37), the natural errors of the 210 checkpoints
    (#39) and the final figures and tables (#36). They overwrite the files in
    results/ with the rebuilt ones, so in a git checkout `git status` shows
    any file that came out different. About 80 min on a CPU.

Usage:
    python run_pipeline.py                         # run every stage of the original pipeline, in order
    python run_pipeline.py --preset final --device cpu   # rebuild every result of the final iteration
    python run_pipeline.py --preset final --list   # print the final stages and their commands
    python run_pipeline.py --from train            # resume from training onward
    python run_pipeline.py --stages train,phs      # run only these stages
    python run_pipeline.py --skip sample,datasets  # run everything except these
    python run_pipeline.py --device cpu            # forwarded to every GPU-capable stage
    python run_pipeline.py --list                  # print the stage list and exit
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.utils.console import enable_utf8_output

PROJECT_ROOT = Path(__file__).parent


def _device(a):
    """--device as given, or nothing: then each stage uses its own default (cuda if available, else cpu)."""
    return ["--device", a.device] if a.device else []


# Each stage: (name, module, description, extra_args(args) -> list[str]).
# Order encodes the real data dependency between stages -- see PIPELINE.md.
ORIGINAL_STAGES = [
    ("sample", "src.data.sampler",
     "Generate data/cases_metadata.json (30 randomized TGV cases).",
     lambda a: []),
    ("datasets", "src.data.generate_datasets",
     "Generate data/tensors/{case_id}.pt point clouds for every case.",
     lambda a: []),
    ("train", "src.models.train_model",
     "Train a PINN per case -> models/{case_id}.pth (skips cases already trained).",
     lambda a: (["--device", a.device] if a.device else [])
               + (["--case_id", a.case_id] if a.case_id else [])),
    ("verify_model", "src.models.verify_model",
     "Sanity-check trained model(s) against the analytical TGV solution.",
     lambda a: ["--all_cases", "--no_show"] if a.case_id is None else ["--case_id", a.case_id, "--no_show"]),
    ("hallucinate", "src.hallucinations.generate_hallucinations",
     "Generate the hallucinated-field dataset + data/hallucinations/hallucination_index.json.",
     lambda a: (["--device", a.device] if a.device else [])),
    ("verify_hallucinations", "src.hallucinations.verify_hallucinations",
     "Sanity-check the hallucination perturbations (visual + quantitative).",
     lambda a: (["--device", a.device] if a.device else []) + ["--all_cases"]),
    ("phs", "src.detection.evaluate_phs",
     "Compute PHS scores, calibrate tau, evaluate detection on the test split.",
     lambda a: (["--device", a.device] if a.device else [])),
    ("sensitivity", "src.detection.detection_sensitivity",
     "Probe the detection boundary against evaluate_phs.py's calibration.",
     lambda a: (["--device", a.device] if a.device else [])),
    ("figures", "src.detection.publication_figures",
     "Generate the paper's figures and tables from evaluate_phs.py's outputs.",
     lambda a: (["--device", a.device] if a.device else [])),
]

# The final iteration's inputs, all in git: the case file (tag cases-v1), the
# 210 checkpoints (models-v1) and the frozen calibration (protocol-v1).
CONFIRMATION_CASES = "cases/confirmation_cases.json"
CONFIRMATION_MODELS = "models/confirmation"
FROZEN_CALIBRATION = "calibration/frozen_calibration.json"
# Its rebuilt intermediate files, which are not in git (data/ is gitignored).
CONFIRMATION_TENSORS = "data/confirmation/tensors"
CONFIRMATION_DAMAGE = "data/confirmation/hallucinations"
CONFIRMATION_SCORES = "data/confirmation/phs_scores"
OLD_CASES = "data/cases_metadata.json"
OLD_ERRORS = "results/accuracy/reference_errors_existing.csv"
NEW_ERRORS = "results/accuracy/reference_errors_confirmation.csv"

# benchmark_metrics.py, natural_errors.py and final_figures.py are run with no
# path flags: their defaults are the final iteration's files, and
# tests/test_run_pipeline.py checks that they match the paths above.
FINAL_STAGES = [
    ("sample", "src.data.sampler",
     f"Write {OLD_CASES}: the 30 old cases (seed 42), whose models' true errors are measured below.",
     lambda a: []),
    ("tensors", "src.data.generate_datasets",
     f"Write the points of the 30 new cases to {CONFIRMATION_TENSORS}/ (the damage stage reuses their "
     "evaluation grid).",
     lambda a: ["--metadata", CONFIRMATION_CASES, "--output_dir", CONFIRMATION_TENSORS]),
    ("damage", "src.hallucinations.generate_hallucinations",
     "Damage the final model of each new case (5 types x 6 strengths) and list the 930 fields in "
     f"{CONFIRMATION_DAMAGE}/hallucination_index.json. It also writes about 1.8 GB of damaged fields "
     "that nothing reads afterwards.",
     lambda a: _device(a) + ["--metadata", CONFIRMATION_CASES, "--models_dir", CONFIRMATION_MODELS,
                             "--tensors_dir", CONFIRMATION_TENSORS, "--output_dir", CONFIRMATION_DAMAGE]),
    ("scores", "src.detection.evaluate_phs",
     f"Score the 930 fields with the frozen calibration -> {CONFIRMATION_SCORES}/phs_components_frozen.csv "
     "(the slow step, about 1 h on a CPU).",
     lambda a: _device(a) + ["--calibration", FROZEN_CALIBRATION, "--metadata", CONFIRMATION_CASES,
                             "--models_dir", CONFIRMATION_MODELS, "--hallucinations_dir", CONFIRMATION_DAMAGE,
                             "--scores_dir", CONFIRMATION_SCORES]),
    ("metrics", "src.detection.benchmark_metrics",
     "Write the benchmark's tables, false alarms and missed detections -> results/benchmark/ (#35).",
     lambda a: []),
    ("errors_old", "src.models.accuracy_audit",
     f"Measure the true error of the 30 old models -> {OLD_ERRORS} (#37).",
     lambda a: ["--metadata", OLD_CASES, "--models_dir", "models", "--out", OLD_ERRORS] + _device(a)),
    ("errors_new", "src.models.accuracy_audit",
     f"Measure the true error of the 210 new checkpoints -> {NEW_ERRORS} (#37).",
     lambda a: ["--metadata", CONFIRMATION_CASES, "--models_dir", CONFIRMATION_MODELS, "--out", NEW_ERRORS]
               + _device(a)),
    ("natural_errors", "src.detection.natural_errors",
     "Score the clean field of the 210 checkpoints and hold the scores against their true errors "
     "-> results/natural_errors/ (#39).",
     lambda a: _device(a)),
    ("final_figures", "src.detection.final_figures",
     "Draw the final figures and tables -> results/figures/ (#36).",
     lambda a: []),
]

PRESETS = {"original": ORIGINAL_STAGES, "final": FINAL_STAGES}
STAGES = ORIGINAL_STAGES                    # the default preset
STAGE_NAMES = [s[0] for s in STAGES]


def parse_args(argv=None):
    """
    Parses the command line.

    Inputs:
        argv (list[str] | None): Arguments; None reads sys.argv.

    Outputs:
        argparse.Namespace: preset, device, case_id, stages, from_stage, skip (str | None) and list (bool).
    """
    parser = argparse.ArgumentParser(
        description="Run the full PINN pipeline end to end. See PIPELINE.md for what each "
                     "stage does, how expensive it is, and when it's safe to skip.",
    )
    parser.add_argument("--preset", type=str, default="original", choices=sorted(PRESETS),
                        help="Which pipeline to run: 'original' (the default), the 30 old cases from sampling "
                             "and training to the first paper's figures; or 'final', every result of the final "
                             "iteration rebuilt from the saved models, with no training (about 80 min on a CPU). "
                             "--stages, --from, --skip and --list then refer to that preset's stages.")
    parser.add_argument("--device", type=str, default=None,
                        help="Forwarded to every stage that accepts --device. "
                             "Defaults to each stage's own default (cuda if available, else cpu).")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Restrict the 'train' and 'verify_model' stages to this one case_id "
                             "(useful for testing the pipeline on a single case). Original preset only.")
    parser.add_argument("--stages", type=str, default=None,
                        help=f"Comma-separated list of stages to run, from: {', '.join(STAGE_NAMES)} "
                             "(see --list for the final preset's). Defaults to all, in the order above.")
    parser.add_argument("--from", dest="from_stage", type=str, default=None,
                        help="Run this stage through the end, skipping every stage before it.")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated list of stage names to skip.")
    parser.add_argument("--list", action="store_true",
                        help="Print the stage list (name, module, what it does, command) and exit without "
                             "running anything.")
    args = parser.parse_args(argv)
    if args.case_id is not None and args.preset != "original":
        parser.error(f"--case_id works only with the original preset: every stage of --preset {args.preset} "
                     "covers all of its cases.")
    return args


def select_stages(args, stages=None):
    """
    Picks the stages to run, in pipeline order.

    Inputs:
        args (argparse.Namespace): Parsed arguments (stages, from_stage, skip; preset when stages is None).
        stages (list | None): The preset's stage list; None takes PRESETS[args.preset].

    Outputs:
        list: The selected stage tuples. Raises ValueError for a stage name the preset doesn't have.
    """
    if stages is None:
        stages = PRESETS[getattr(args, "preset", "original")]
    names = [s[0] for s in stages]

    if args.stages:
        requested = args.stages.split(",")
        unknown = [s for s in requested if s not in names]
        if unknown:
            raise ValueError(f"Unknown stage name(s): {unknown}. Valid stages: {names}")
        selected = [s for s in stages if s[0] in requested]
    elif args.from_stage:
        if args.from_stage not in names:
            raise ValueError(f"Unknown stage name in --from: {args.from_stage!r}. Valid stages: {names}")
        selected = stages[names.index(args.from_stage):]
    else:
        selected = list(stages)

    if args.skip:
        skip_set = set(args.skip.split(","))
        unknown = skip_set - set(names)
        if unknown:
            raise ValueError(f"Unknown stage name(s) in --skip: {unknown}. Valid stages: {names}")
        selected = [s for s in selected if s[0] not in skip_set]

    return selected


def stage_command(stage, args, python=None):
    """
    The command that runs one stage.

    Inputs:
        stage (tuple): (name, module, description, extra_args_fn).
        args (argparse.Namespace): Parsed arguments, passed to extra_args_fn.
        python (str | None): The interpreter to name; None uses this one (sys.executable).

    Outputs:
        list[str]: [python, "-m", module, *flags], run from the project root.
    """
    _, module, _, extra_args_fn = stage
    return [python or sys.executable, "-m", module] + extra_args_fn(args)


def main(argv=None):
    args = parse_args(argv)
    stages = PRESETS[args.preset]

    if args.list:
        print(f"Stages of --preset {args.preset}, in order (each runs from the project root):")
        for stage in stages:
            name, module, desc, _ = stage
            print(f"  {name:24s} ({module})\n      {desc}\n      > {' '.join(stage_command(stage, args, 'python'))}")
        return

    try:
        selected = select_stages(args, stages)
    except ValueError as err:
        sys.exit(f"run_pipeline.py: error: {err}")
    if not selected:
        print("No stages selected -- nothing to run (check --stages/--skip/--from).")
        return

    print("=" * 70, flush=True)
    print(f"PINN PIPELINE (--preset {args.preset}): running {len(selected)} stage(s): "
          f"{', '.join(s[0] for s in selected)}", flush=True)
    print(f"Command: python run_pipeline.py {' '.join(sys.argv[1:] if argv is None else argv)}", flush=True)
    if args.preset == "final":
        print("No stage trains. The rebuilt tables and figures overwrite the files in results/.", flush=True)
    print("=" * 70, flush=True)

    pipeline_start = time.monotonic()
    for stage in selected:
        name, _, desc, _ = stage
        cmd = stage_command(stage, args)
        print(f"\n{'-' * 70}\n[{name}] {desc}\n> {' '.join(cmd)}\n{'-' * 70}", flush=True)
        stage_start = time.monotonic()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        minutes = (time.monotonic() - stage_start) / 60
        if result.returncode != 0:
            print(f"\n❌ Stage '{name}' failed (exit code {result.returncode}) after {minutes:.1f} min. "
                  "Stopping pipeline.", flush=True)
            sys.exit(result.returncode)
        print(f"[{name}] done in {minutes:.1f} min", flush=True)

    total = (time.monotonic() - pipeline_start) / 60
    print("\n" + "=" * 70, flush=True)
    print(f"✅ Pipeline complete (--preset {args.preset}): {', '.join(s[0] for s in selected)} "
          f"in {total:.1f} min", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    enable_utf8_output()
    main()
