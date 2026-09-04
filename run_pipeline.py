"""
End-to-End Pipeline Runner

Runs every stage of the PINN pipeline in dependency order by invoking each
stage's own script exactly as documented in the README's Usage section and
in PIPELINE.md (same module, same flags) -- this file sequences the
pipeline, it does not duplicate any of its logic.

See PIPELINE.md for what each stage produces, how expensive it is, and
when it is safe to skip: sampling and dataset generation are deterministic
(safe to skip once data/cases_metadata.json and data/tensors/ exist);
training already skips any case_id that has a models/{case_id}_best.pth
checkpoint; every stage after training regenerates its outputs in full
each run (no skip-if-exists logic), which PIPELINE.md flags per stage.

Usage:
    python run_pipeline.py                        # run every stage, in order
    python run_pipeline.py --from train            # resume from training onward
    python run_pipeline.py --stages train,phs      # run only these stages
    python run_pipeline.py --skip sample,datasets  # run everything except these
    python run_pipeline.py --device cpu            # forwarded to every GPU-capable stage
    python run_pipeline.py --list                  # print the stage list and exit
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Each stage: (name, module, description, extra_args(args) -> list[str]).
# Order encodes the real data dependency between stages -- see PIPELINE.md.
STAGES = [
    ("sample", "src.data.sampler",
     "Generate data/cases_metadata.json (30 randomized TGV cases).",
     lambda a: []),
    ("datasets", "src.data.generate_datasets",
     "Generate data/tensors/{case_id}.pt point clouds for every case.",
     lambda a: []),
    ("train", "src.models.train_model",
     "Train a PINN per case -> models/{case_id}_best.pth (skips cases already trained).",
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
STAGE_NAMES = [s[0] for s in STAGES]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full PINN pipeline end to end. See PIPELINE.md for what each "
                     "stage does, how expensive it is, and when it's safe to skip.",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Forwarded to every stage that accepts --device. "
                             "Defaults to each stage's own default (cuda if available, else cpu).")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Restrict the 'train' and 'verify_model' stages to this one case_id "
                             "(useful for testing the pipeline on a single case).")
    parser.add_argument("--stages", type=str, default=None,
                        help=f"Comma-separated list of stages to run, from: {', '.join(STAGE_NAMES)}. "
                             "Defaults to all, in the order above.")
    parser.add_argument("--from", dest="from_stage", type=str, default=None, choices=STAGE_NAMES,
                        help="Run this stage through the end, skipping every stage before it.")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated list of stage names to skip.")
    parser.add_argument("--list", action="store_true",
                        help="Print the stage list (name, module, what it does) and exit without running anything.")
    return parser.parse_args()


def select_stages(args):
    if args.stages:
        requested = args.stages.split(",")
        unknown = [s for s in requested if s not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"Unknown stage name(s): {unknown}. Valid stages: {STAGE_NAMES}")
        selected = [s for s in STAGES if s[0] in requested]
    elif args.from_stage:
        start = STAGE_NAMES.index(args.from_stage)
        selected = STAGES[start:]
    else:
        selected = list(STAGES)

    if args.skip:
        skip_set = set(args.skip.split(","))
        unknown = skip_set - set(STAGE_NAMES)
        if unknown:
            raise ValueError(f"Unknown stage name(s) in --skip: {unknown}. Valid stages: {STAGE_NAMES}")
        selected = [s for s in selected if s[0] not in skip_set]

    return selected


def main():
    args = parse_args()

    if args.list:
        for name, module, desc, _ in STAGES:
            print(f"  {name:24s} ({module})\n      {desc}")
        return

    stages = select_stages(args)
    if not stages:
        print("No stages selected -- nothing to run (check --stages/--skip/--from).")
        return

    print("=" * 70, flush=True)
    print(f"PINN PIPELINE: running {len(stages)} stage(s): {', '.join(s[0] for s in stages)}", flush=True)
    print("=" * 70, flush=True)

    for name, module, desc, extra_args_fn in stages:
        cmd = [sys.executable, "-m", module] + extra_args_fn(args)
        print(f"\n{'-' * 70}\n[{name}] {desc}\n> {' '.join(cmd)}\n{'-' * 70}", flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"\n❌ Stage '{name}' failed (exit code {result.returncode}). Stopping pipeline.", flush=True)
            sys.exit(result.returncode)

    print("\n" + "=" * 70, flush=True)
    print(f"✅ Pipeline complete: {', '.join(s[0] for s in stages)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
