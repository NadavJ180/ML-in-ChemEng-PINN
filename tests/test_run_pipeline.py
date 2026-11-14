"""
Tests for run_pipeline.py: the two presets, stage selection, and the final
preset's wiring (#46). Nothing here runs a stage: subprocess.run is replaced
by a stub wherever main() is called.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import run_pipeline as rp

ROOT = Path(rp.__file__).resolve().parent

ORIGINAL = [
    ("sample", "src.data.sampler"),
    ("datasets", "src.data.generate_datasets"),
    ("train", "src.models.train_model"),
    ("verify_model", "src.models.verify_model"),
    ("hallucinate", "src.hallucinations.generate_hallucinations"),
    ("verify_hallucinations", "src.hallucinations.verify_hallucinations"),
    ("phs", "src.detection.evaluate_phs"),
    ("sensitivity", "src.detection.detection_sensitivity"),
    ("figures", "src.detection.publication_figures"),
]
FINAL = [
    ("sample", "src.data.sampler"),
    ("tensors", "src.data.generate_datasets"),
    ("damage", "src.hallucinations.generate_hallucinations"),
    ("scores", "src.detection.evaluate_phs"),
    ("metrics", "src.detection.benchmark_metrics"),
    ("errors_old", "src.models.accuracy_audit"),
    ("errors_new", "src.models.accuracy_audit"),
    ("natural_errors", "src.detection.natural_errors"),
    ("final_figures", "src.detection.final_figures"),
]
# The stages whose scripts take --device; the others would refuse it
FINAL_GPU_STAGES = {"damage", "scores", "errors_old", "errors_new", "natural_errors"}


def args_for(argv):
    return rp.parse_args(argv)


def flags_of(stage_name, argv=("--preset", "final")):
    """The flags the final preset passes to one stage."""
    args = args_for(list(argv))
    (stage,) = [s for s in rp.FINAL_STAGES if s[0] == stage_name]
    return stage[3](args)


def flag_value(stage_name, flag):
    flags = flags_of(stage_name)
    return flags[flags.index(flag) + 1]


def same_path(a, b):
    return (ROOT / a).resolve() == Path(b).resolve()


# --- the presets ---

def test_the_original_preset_is_unchanged_and_the_default():
    assert [(s[0], s[1]) for s in rp.ORIGINAL_STAGES] == ORIGINAL
    assert rp.STAGES is rp.ORIGINAL_STAGES
    assert args_for([]).preset == "original"
    assert [s[0] for s in rp.select_stages(args_for([]))] == [name for name, _ in ORIGINAL]


def test_the_final_preset_runs_these_stages_in_this_order():
    assert [(s[0], s[1]) for s in rp.FINAL_STAGES] == FINAL
    assert rp.PRESETS == {"original": rp.ORIGINAL_STAGES, "final": rp.FINAL_STAGES}


def test_no_final_stage_trains():
    modules = {s[1] for s in rp.FINAL_STAGES}
    assert "src.models.train_model" not in modules
    assert not any("train" in name for name, *_ in rp.FINAL_STAGES)


@pytest.mark.parametrize("stage", [s for s in rp.ORIGINAL_STAGES + rp.FINAL_STAGES], ids=lambda s: s[0])
def test_every_stage_module_exists(stage):
    assert (ROOT / (stage[1].replace(".", "/") + ".py")).is_file()


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("name", [s[0] for s in rp.FINAL_STAGES])
def test_every_final_stage_script_accepts_its_flags(name, device):
    """Parses each stage's flags with that script's own parser, so a renamed or missing flag fails here."""
    argv = ["--preset", "final"] + (["--device", device] if device else [])
    flags = flags_of(name, argv)
    assert ("--device" in flags) == (device is not None and name in FINAL_GPU_STAGES)
    if not flags:
        return
    module = importlib.import_module(dict(FINAL)[name])
    with mock.patch.object(sys, "argv", [module.__file__] + flags):
        parsed = module.parse_args()
    if "--device" in flags:
        assert parsed.device == device


def test_the_final_stages_hand_their_files_on():
    """Each stage reads what an earlier stage writes, or a committed input."""
    benchmark_metrics = importlib.import_module("src.detection.benchmark_metrics")
    evaluate_phs = importlib.import_module("src.detection.evaluate_phs")
    natural_errors = importlib.import_module("src.detection.natural_errors")
    final_figures = importlib.import_module("src.detection.final_figures")

    assert flag_value("tensors", "--output_dir") == flag_value("damage", "--tensors_dir")
    assert flag_value("damage", "--output_dir") == flag_value("scores", "--hallucinations_dir")
    for stage in ("tensors", "damage", "scores", "errors_new"):
        assert flag_value(stage, "--metadata") == rp.CONFIRMATION_CASES
    for stage in ("damage", "scores", "errors_new"):
        assert flag_value(stage, "--models_dir") == rp.CONFIRMATION_MODELS
    assert flag_value("scores", "--calibration") == rp.FROZEN_CALIBRATION

    # metrics: evaluate_phs.py --calibration writes {scores_dir}/phs_components_frozen.csv
    scores_csv = Path(flag_value("scores", "--scores_dir")) / evaluate_phs.FROZEN_SCORES_FILE
    assert same_path(scores_csv, benchmark_metrics.DEFAULT_SCORES_CSV)
    assert same_path(rp.FROZEN_CALIBRATION, benchmark_metrics.DEFAULT_CALIBRATION_JSON)
    assert same_path(rp.CONFIRMATION_CASES, benchmark_metrics.DEFAULT_METADATA)
    assert same_path(rp.CONFIRMATION_MODELS, benchmark_metrics.DEFAULT_MODELS_DIR)
    assert same_path("results/benchmark", benchmark_metrics.DEFAULT_OUTPUT_DIR)

    # errors: the old cases come from the sample stage, the new ones from the case file
    assert flag_value("errors_old", "--metadata") == rp.OLD_CASES == "data/cases_metadata.json"
    assert flag_value("errors_old", "--models_dir") == "models"
    assert flag_value("errors_old", "--out") == rp.OLD_ERRORS
    assert flag_value("errors_new", "--out") == rp.NEW_ERRORS

    # natural_errors reads the true errors and the benchmark's per-field table
    assert same_path(rp.NEW_ERRORS, natural_errors.DEFAULT_ERRORS_CSV)
    assert same_path("results/benchmark/confirmation_scores_raw.csv", natural_errors.DEFAULT_BENCHMARK_CSV)
    assert same_path(rp.CONFIRMATION_CASES, natural_errors.DEFAULT_METADATA)
    assert same_path(rp.CONFIRMATION_MODELS, natural_errors.DEFAULT_MODELS_DIR)
    assert same_path(rp.FROZEN_CALIBRATION, natural_errors.DEFAULT_CALIBRATION_JSON)
    assert same_path("results/natural_errors", natural_errors.DEFAULT_OUTPUT_DIR)

    # final_figures reads everything above
    assert final_figures.DEFAULT_BENCHMARK_DIR.resolve() == benchmark_metrics.DEFAULT_OUTPUT_DIR.resolve()
    assert final_figures.DEFAULT_NATURAL_DIR.resolve() == natural_errors.DEFAULT_OUTPUT_DIR.resolve()
    assert same_path(rp.NEW_ERRORS, final_figures.DEFAULT_ERRORS_CSV)
    assert same_path(rp.CONFIRMATION_CASES, final_figures.DEFAULT_METADATA)
    assert same_path("models/confirmation/loss_summary.csv", final_figures.DEFAULT_LOSS_SUMMARY)
    assert same_path(rp.FROZEN_CALIBRATION, final_figures.DEFAULT_CALIBRATION_JSON)
    assert same_path("results/figures", final_figures.DEFAULT_OUTPUT_DIR)


def test_the_committed_inputs_of_the_final_preset_exist():
    for path in (rp.CONFIRMATION_CASES, rp.FROZEN_CALIBRATION, "calibration/source_phs_components_raw.csv",
                 "models/confirmation/loss_summary.csv"):
        assert (ROOT / path).is_file(), path
    assert len(list((ROOT / rp.CONFIRMATION_MODELS).glob("case_*/lbfgs_final.pth"))) == 30
    assert len(list((ROOT / "models").glob("case_*.pth"))) == 30


# --- selecting stages ---

@pytest.mark.parametrize("argv, expected", [
    (["--preset", "final", "--from", "scores"],
     ["scores", "metrics", "errors_old", "errors_new", "natural_errors", "final_figures"]),
    (["--preset", "final", "--stages", "final_figures,metrics"], ["metrics", "final_figures"]),
    (["--preset", "final", "--skip", "sample,tensors,damage,scores"],
     ["metrics", "errors_old", "errors_new", "natural_errors", "final_figures"]),
    (["--from", "phs"], ["phs", "sensitivity", "figures"]),
    (["--stages", "sample,hallucinate,phs,figures"], ["sample", "hallucinate", "phs", "figures"]),
])
def test_select_stages_picks_within_the_preset(argv, expected):
    assert [s[0] for s in rp.select_stages(args_for(argv))] == expected


@pytest.mark.parametrize("argv", [
    ["--preset", "final", "--stages", "phs"],        # an original stage
    ["--preset", "final", "--from", "train"],
    ["--preset", "final", "--skip", "figures"],
    ["--stages", "damage"],                           # a final stage
    ["--from", "natural_errors"],
])
def test_a_stage_of_the_other_preset_is_refused(argv):
    with pytest.raises(ValueError, match="Valid stages"):
        rp.select_stages(args_for(argv))


def test_case_id_is_refused_with_the_final_preset():
    with pytest.raises(SystemExit):
        args_for(["--preset", "final", "--case_id", "case_30"])
    assert args_for(["--case_id", "case_00"]).case_id == "case_00"


# --- running ---

def run_main(argv, returncodes=None):
    """Calls main(argv) with subprocess.run stubbed; returns the commands it ran."""
    calls = []
    codes = list(returncodes or [])

    def fake_run(cmd, cwd):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, codes.pop(0) if codes else 0)

    with mock.patch.object(rp.subprocess, "run", side_effect=fake_run):
        rp.main(argv)
    return calls


def test_main_runs_every_final_stage_once_in_order_from_the_project_root(capsys):
    calls = run_main(["--preset", "final", "--device", "cpu"])
    assert [cmd[2] for cmd, _ in calls] == [module for _, module in FINAL]
    assert all(cmd[:2] == [sys.executable, "-m"] for cmd, _ in calls)
    assert all(Path(cwd).resolve() == ROOT for _, cwd in calls)
    out = capsys.readouterr().out
    assert "No stage trains" in out
    assert "Pipeline complete (--preset final)" in out


def test_main_stops_at_the_first_failing_stage(capsys):
    with pytest.raises(SystemExit) as exc:
        run_main(["--preset", "final"], returncodes=[0, 0, 3])
    assert exc.value.code == 3
    assert "Stage 'damage' failed (exit code 3)" in capsys.readouterr().out


def test_list_prints_every_stage_and_runs_nothing(capsys):
    calls = run_main(["--preset", "final", "--list", "--device", "cpu"])
    assert calls == []
    out = capsys.readouterr().out
    for name, module in FINAL:
        assert name in out and module in out
    assert ("> python -m src.detection.evaluate_phs --device cpu --calibration "
            "calibration/frozen_calibration.json") in out


def test_an_unknown_stage_exits_with_a_message():
    with pytest.raises(SystemExit) as exc:
        run_main(["--preset", "final", "--stages", "phs"])
    assert "Unknown stage name" in str(exc.value.code)
