"""
Tests for release/build_final_package.py (#46): the comparison of rebuilt
outputs, the package checks, the log, the report and the ZIP. They build small
packages in tmp_path; only the export and checks marked "real" read this
repository.
"""

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import release.build_final_package as b

ROOT = b.ROOT


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)
    return path


def pair(tmp_path, name, packaged, rebuilt):
    return write(tmp_path / "package" / name, packaged), write(tmp_path / "rebuilt" / name, rebuilt)


CSV = ("case_id,k,flagged_Score1,n_flagged,score,low\n"
       "case_30,1,True,3,1.2345678901234567,NaN\n"
       "case_31,all,False,0,2.5e-07,0.5\n")


# --- compare_file ---

def test_identical_bytes_and_line_endings_aside(tmp_path):
    assert b.compare_file(*pair(tmp_path, "a.csv", CSV, CSV)) == (b.IDENTICAL, "")
    status, detail = b.compare_file(*pair(tmp_path, "b.tex", "x\ny\n", "x\r\ny\r\n"))
    assert status == b.IDENTICAL and "line endings" in detail


def test_a_csv_that_differs_in_the_last_digits_has_the_same_numbers(tmp_path):
    rebuilt = CSV.replace("1.2345678901234567", "1.2345678901234569").replace("2.5e-07", "2.5000000000001e-07")
    status, detail = b.compare_file(*pair(tmp_path, "a.csv", CSV, rebuilt))
    assert status == b.SAME_NUMBERS
    assert "e-1" in detail


@pytest.mark.parametrize("old, new, why", [
    ("True,3", "False,3", "flagged_Score1"),        # a flag
    ("True,3", "True,4", "n_flagged"),              # a count
    (",NaN\n", ",0.1\n", "NaN"),                    # an empty cell
    ("case_31,all", "case_31,1", "k"),              # a text cell
    ("1.2345678901234567", "1.2345679", "score"),   # a number beyond 1e-9
])
def test_a_csv_with_another_flag_count_or_number_is_different(tmp_path, old, new, why):
    status, detail = b.compare_file(*pair(tmp_path, "a.csv", CSV, CSV.replace(old, new)))
    assert status == b.DIFFERENT
    assert why in detail


def test_a_csv_with_other_rows_or_columns_is_different(tmp_path):
    assert b.compare_file(*pair(tmp_path, "a.csv", CSV, CSV.rsplit("case_31", 1)[0]))[0] == b.DIFFERENT
    assert b.compare_file(*pair(tmp_path, "b.csv", CSV, CSV.replace("score,low", "score,high")))[0] == b.DIFFERENT


def test_tex_must_be_equal_byte_for_byte(tmp_path):
    status, _ = b.compare_file(*pair(tmp_path, "a.tex", "0.872 [0.868, 0.880]\n", "0.872 [0.868, 0.881]\n"))
    assert status == b.DIFFERENT


def test_json_numbers_are_compared_with_a_tolerance(tmp_path):
    old = {"train": [{"case_id": "case_00", "Re": 113.10000000000001, "k": 1}]}
    close = {"train": [{"case_id": "case_00", "Re": 113.10000000000004, "k": 1}]}
    far = {"train": [{"case_id": "case_00", "Re": 113.2, "k": 1}]}
    text = {"train": [{"case_id": "case_01", "Re": 113.10000000000001, "k": 1}]}
    assert b.compare_file(*pair(tmp_path, "a.json", json.dumps(old), json.dumps(close)))[0] == b.SAME_NUMBERS
    assert b.compare_file(*pair(tmp_path, "b.json", json.dumps(old), json.dumps(far)))[0] == b.DIFFERENT
    assert b.compare_file(*pair(tmp_path, "c.json", json.dumps(old), json.dumps(text)))[0] == b.DIFFERENT


def test_figures_whose_bytes_differ_are_drawn_differently(tmp_path):
    import matplotlib.image as mpimg

    image = np.zeros((10, 20, 4))
    image[..., 3] = 1.0
    packaged, rebuilt = tmp_path / "a.png", tmp_path / "b.png"
    mpimg.imsave(packaged, image)
    image[0, 0, 0] = 1.0
    mpimg.imsave(rebuilt, image)
    status, detail = b.compare_file(packaged, rebuilt)
    assert status == b.DRAWN_DIFFERENTLY and "0.50% of the pixels" in detail

    assert b.compare_file(*pair(tmp_path, "c.pdf", b"%PDF-1.4 a", b"%PDF-1.4 b"))[0] == b.DRAWN_DIFFERENTLY
    assert b.compare_file(*pair(tmp_path, "d.png", b"not a png", b"also not"))[0] == b.DIFFERENT


# --- compare_outputs ---

def make_results(folder: Path):
    write(folder / b.OLD_CASES, '{"train": []}\n')
    for result_dir in b.RESULT_DIRS:
        write(folder / result_dir / "table.csv", CSV)
        write(folder / result_dir / "table.tex", "t\n")


def test_compare_outputs_lists_missing_and_new_files(tmp_path):
    package, rebuilt = tmp_path / "package", tmp_path / "rebuilt"
    make_results(package)
    make_results(rebuilt)
    (rebuilt / "results/figures/table.tex").unlink()
    write(rebuilt / "results/benchmark/metrics_passed_only.csv", CSV)
    rows = {path: status for status, path, _ in b.compare_outputs(package, rebuilt)}
    assert rows["results/figures/table.tex"] == b.MISSING
    assert rows["results/benchmark/metrics_passed_only.csv"] == b.NEW
    assert rows[b.OLD_CASES] == b.IDENTICAL
    assert len(rows) == 1 + 2 * len(b.RESULT_DIRS) + 1


# --- the checks of the package ---

def make_package(folder: Path, n_new=2):
    """A small package that passes every check: 30 old models, n_new new cases, the README and so on."""
    new_ids = [f"case_{30 + i}" for i in range(n_new)]
    write(folder / b.NEW_CASES, json.dumps({"confirmation": [{"case_id": c} for c in new_ids]}))
    for n in range(b.N_OLD_CASES):
        write(folder / f"models/case_{n:02d}.pth", f"old {n}")
    for c in new_ids:
        for stage in b.STAGES:
            write(folder / f"models/confirmation/{c}/{stage}.pth", f"{c} {stage}")
            write(folder / f"models/confirmation/{c}/{stage}.json", "{}")
        write(folder / f"models/confirmation/{c}/loss_history.json", "{}")
    for name in ("loss_summary.csv", "training_summary.log", "RESTARTS.md"):
        write(folder / "models/confirmation" / name, name)
    write_manifest(folder)
    for _, paths in b.expected_files(folder):
        for path in paths:
            if not (folder / path).exists():
                write(folder / path, "x\n")
    write(folder / "results/figures/fig_roc.pdf", "pdf")
    write(folder / "results/figures/fig_roc.png", "png")
    for result_dir in b.TABLE_DIRS:
        for name in os.listdir(folder / result_dir):
            if name.endswith(".csv"):
                write(folder / result_dir / name.replace(".csv", ".tex"), "t\n")


def write_manifest(folder: Path):
    models = folder / "models/confirmation"
    lines = [f"{b.sha256(models / name)}  ./{name}" for name in b.files_under(models)
             if name != "MANIFEST.sha256" and not name.endswith((".png", ".tmp"))]
    write(models / "MANIFEST.sha256", "\n".join(lines) + "\n")


def test_a_complete_package_passes_its_checks(tmp_path):
    make_package(tmp_path)
    assert b.check_contents(tmp_path) == []
    n_listed, problems = b.check_manifest(tmp_path)
    assert problems == [] and n_listed == 2 * (2 * len(b.STAGES) + 1) + 3


def test_check_contents_reports_missing_items_and_incomplete_pairs(tmp_path):
    make_package(tmp_path)
    (tmp_path / "models/case_07.pth").unlink()
    (tmp_path / "results/natural_errors/missed_failures.csv").unlink()
    (tmp_path / "results/benchmark/metrics_overall.csv").write_text("x")
    (tmp_path / "results/figures/fig_roc.png").unlink()
    problems = "\n".join(b.check_contents(tmp_path))
    assert "Models: missing models/case_07.pth" in problems
    assert "Every false alarm and every miss: missing results/natural_errors/missed_failures.csv" in problems
    assert "results/benchmark/metrics_overall.csv has no .tex" in problems
    assert "results/figures/fig_roc has only .pdf" in problems


def test_raw_errors_need_no_tex_but_a_result_folder_needs_files(tmp_path):
    make_package(tmp_path)
    assert not (tmp_path / "results/accuracy/reference_errors_existing.tex").exists()
    for name in os.listdir(tmp_path / "results/figures"):
        (tmp_path / "results/figures" / name).unlink()
    assert b.check_contents(tmp_path) == ["Figures and tables: results/figures/ has no files"]


def test_check_manifest_catches_a_changed_a_missing_and_an_unlisted_file(tmp_path):
    make_package(tmp_path)
    models = tmp_path / "models/confirmation"
    (models / "case_30/adam_00100.pth").write_text("changed")
    (models / "case_31/lbfgs_final.json").unlink()
    write(models / "case_31/extra.pth", "new")
    write(models / "case_31/plot.png", "png files are left out of the manifest")
    problems = "\n".join(b.check_manifest(tmp_path)[1])
    assert "case_30/adam_00100.pth does not match" in problems
    assert "case_31/lbfgs_final.json, which is missing" in problems
    assert "does not list case_31/extra.pth" in problems
    assert "plot.png" not in problems


def test_protocol_hashes_are_read_from_the_line_that_names_the_file():
    text = ("Their fingerprint is `models/confirmation/MANIFEST.sha256` (455 files; the SHA-256 of it is "
            f"`{'a' * 64}`).\n"
            f"| `calibration/source_phs_components_raw.csv` | `{'b' * 64}` |\n"
            f"| `calibration/frozen_calibration.json` | `{'c' * 64}` |\n")
    assert b.protocol_hashes(text) == {"models/confirmation/MANIFEST.sha256": "a" * 64,
                                       "calibration/source_phs_components_raw.csv": "b" * 64,
                                       "calibration/frozen_calibration.json": "c" * 64}


def test_check_protocol_hashes_reports_a_changed_file(tmp_path):
    for path in b.PROTOCOL_HASHED:
        write(tmp_path / path, path)
    rows = [f"| `{p}` | `{b.sha256(tmp_path / p)}` |" for p in b.PROTOCOL_HASHED]
    write(tmp_path / b.PROTOCOL, "\n".join(rows) + "\n")
    assert b.check_protocol_hashes(tmp_path) == []
    (tmp_path / "calibration/frozen_calibration.json").write_text("refit")
    (problem,) = b.check_protocol_hashes(tmp_path)
    assert problem.startswith("calibration/frozen_calibration.json does not have the SHA-256 that PROTOCOL.md records")
    write(tmp_path / b.PROTOCOL, "\n".join(rows[:2]) + "\n")
    assert b.check_protocol_hashes(tmp_path) == ["PROTOCOL.md records no SHA-256 for calibration/frozen_calibration.json"]


def test_the_real_checkpoints_and_frozen_files_pass_their_checks():
    """real: the committed models/confirmation/ against its MANIFEST, and the hashes in PROTOCOL.md."""
    n_listed, problems = b.check_manifest(ROOT)
    assert problems == [] and n_listed == 455
    assert b.check_protocol_hashes(ROOT) == []


def test_export_tree_writes_files_as_git_stores_them(tmp_path):
    """real: text files come out with LF on every OS, and -text files with their exact bytes."""
    b.export_tree("HEAD", tmp_path, ["run_pipeline.py", "calibration/source_phs_components_raw.csv"])
    assert b"\r\n" not in (tmp_path / "run_pipeline.py").read_bytes()
    assert (b.sha256(tmp_path / "calibration/source_phs_components_raw.csv")
            == b.protocol_hashes((ROOT / b.PROTOCOL).read_text(encoding="utf-8"))[
                "calibration/source_phs_components_raw.csv"])


def test_export_tree_of_an_unknown_ref_fails_cleanly(tmp_path):
    with pytest.raises(b.BuildError):
        b.export_tree("no-such-ref-46", tmp_path)
    with pytest.raises(b.BuildError, match="no tag, branch or commit"):
        b.resolve_ref("no-such-ref-46")
    assert b.resolve_ref("HEAD") == b.git("rev-parse", "HEAD")


def test_the_checkpoint_stages_are_the_training_code_s():
    from src.models.checkpoints import STAGES
    assert b.STAGES == STAGES


def test_the_rebuild_must_be_a_checkout_of_the_package_s_commit_changed_only_in_results(tmp_path):
    def git(*args):
        b.git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args, cwd=tmp_path)
    write(tmp_path / "results/a.csv", "1\n")
    write(tmp_path / "src/x.py", "x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "c")
    commit = b.git("rev-parse", "HEAD", cwd=tmp_path)

    (tmp_path / "results/a.csv").write_text("2\n")
    note, problems = b.check_rebuilt_checkout(tmp_path, commit)
    assert problems == [] and "1 files in results/" in note
    assert b.check_rebuilt_checkout(tmp_path, "0" * 40)[1] == [
        f"{tmp_path} is at commit {commit[:7]}, not at the package's 0000000"]
    (tmp_path / "src/x.py").write_text("x = 2\n")
    assert b.check_rebuilt_checkout(tmp_path, commit)[1] == [
        "the rebuild changed tracked files outside results/: src/x.py"]


# --- the log, the report and the ZIP ---

def test_read_log_hides_folder_names_and_reads_the_run(tmp_path):
    rebuilt = tmp_path / "pinn-final-check"
    rebuilt.mkdir()
    python = r"C:\Users\someone\venv\Scripts\python.exe"
    log = write(tmp_path / "log.txt",
                "PINN PIPELINE (--preset final): running 9 stage(s)\r\n"
                "Command: python run_pipeline.py --preset final --device cpu\r\n"
                f"> {python} -m src.data.sampler\r\n"
                f"Successfully exported dataset to: {rebuilt}\\data\\cases_metadata.json\r\n"
                f"Wrote {str(rebuilt).replace(os.sep, '/')}/results/benchmark\r\n"
                f"Home: {Path.home()}\\x\r\n"
                "✅ Pipeline complete (--preset final): sample, tensors in 97.3 min\r\n")
    text, command, minutes = b.read_log(log, rebuilt)
    assert command == "python run_pipeline.py --preset final --device cpu"
    assert minutes == "97.3"
    assert "> python -m src.data.sampler" in text
    assert "<repo>\\data\\cases_metadata.json" in text and "<repo>/results/benchmark" in text
    assert str(rebuilt) not in text and str(Path.home()) not in text and "\r" not in text


def test_read_log_refuses_an_unfinished_run(tmp_path):
    log = write(tmp_path / "log.txt", "Command: python run_pipeline.py --preset final\n❌ Stage 'scores' failed\n")
    with pytest.raises(b.BuildError, match="not the log of a finished"):
        b.read_log(log, tmp_path)


def test_write_report_passes_only_when_every_output_matches(tmp_path):
    rows = [(b.IDENTICAL, "results/a.csv", ""), (b.SAME_NUMBERS, "results/b.csv", "largest 1e-13"),
            (b.DRAWN_DIFFERENTLY, "results/c.png", "same size")]
    args = ("final-v1", "a" * 40)
    assert b.write_report(tmp_path, *args, rows, ["checked"], [], "python run_pipeline.py --preset final", "97.3",
                          "note")
    report = (tmp_path / b.CHECK_NAME).read_text(encoding="utf-8")
    assert "RESULT: every rebuilt output matches" in report
    assert "Outputs compared: 3: 1 identical, 1 same numbers, 1 drawn differently." in report
    row_lines = [line for line in report.splitlines() if line.startswith(b.STATUS_ORDER)]
    assert [line.split("results/")[1].split()[0] for line in row_lines] == ["c.png", "b.csv", "a.csv"]
    assert "- checked" in report

    assert not b.write_report(tmp_path, *args, rows + [(b.MISSING, "results/d.tex", "not rebuilt")], [], [],
                              "cmd", "1", "note")
    assert "NOT REPRODUCED" in (tmp_path / b.CHECK_NAME).read_text(encoding="utf-8")
    assert not b.write_report(tmp_path, *args, rows, [], ["a problem"], "cmd", "1", "note")


def test_make_zip_writes_one_top_folder_with_forward_slashes(tmp_path):
    package = tmp_path / b.PACKAGE_NAME
    write(package / "a/b/c.txt", "c")
    write(package / "z.txt", "z")
    write(package / "src/__pycache__/x.pyc", "cache")
    zip_path = tmp_path / "p.zip"
    assert b.make_zip(package, zip_path) == 2
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [f"{b.PACKAGE_NAME}/a/b/c.txt", f"{b.PACKAGE_NAME}/z.txt"]
        assert archive.read(f"{b.PACKAGE_NAME}/a/b/c.txt") == b"c"
    assert not (tmp_path / "p.zip.partial").exists()


# --- the command line and the whole build ---

def test_the_final_package_needs_the_check_and_a_trial_must_not_have_one():
    with pytest.raises(SystemExit):
        b.parse_args([])
    with pytest.raises(SystemExit):
        b.parse_args(["--rebuilt", "x"])
    with pytest.raises(SystemExit):
        b.parse_args(["--trial", "--rebuilt", "x", "--log", "y"])
    assert b.parse_args(["--trial"]).ref == b.DEFAULT_REF
    assert b.parse_args(["--rebuilt", "x", "--log", "y"]).trial is False


def fake_build_steps(monkeypatch, n_new=2):
    """Replaces the git export, the sampler and pip freeze, so build() runs on a small package."""
    def export(ref, dest, paths=()):
        make_package(dest, n_new)
        write(dest / b.README_SOURCE, "# The package\n")
        for added in (b.README_NAME, b.FREEZE_NAME, b.OLD_CASES):      # made by build() itself
            (dest / added).unlink()
    monkeypatch.setattr(b, "resolve_ref", lambda ref: "c" * 40)
    monkeypatch.setattr(b, "export_tree", export)
    monkeypatch.setattr(b, "rebuild_old_cases", lambda d: write(d / b.OLD_CASES, '{"train": []}\n'))
    monkeypatch.setattr(b, "write_pip_freeze", lambda d: write(d / b.FREEZE_NAME, "numpy==2.5.1\n"))
    monkeypatch.setattr(b, "check_protocol_hashes", lambda d: [])


def test_a_trial_build_zips_a_checked_package(tmp_path, monkeypatch):
    fake_build_steps(monkeypatch)
    zip_path = b.build(b.parse_args(["--trial", "--out", str(tmp_path)]))
    assert zip_path == tmp_path / f"{b.PACKAGE_NAME}{b.TRIAL_SUFFIX}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    top = f"{b.PACKAGE_NAME}{b.TRIAL_SUFFIX}/"
    assert f"{top}{b.README_NAME}" in names and f"{top}{b.FREEZE_NAME}" in names
    assert f"{top}{b.CHECK_NAME}" not in names

    with pytest.raises(b.BuildError, match="already exists"):
        b.build(b.parse_args(["--trial", "--out", str(tmp_path)]))
    assert b.build(b.parse_args(["--trial", "--out", str(tmp_path), "--overwrite"])) == zip_path


def test_a_trial_build_with_a_missing_item_is_not_zipped(tmp_path, monkeypatch):
    fake_build_steps(monkeypatch)
    monkeypatch.setattr(b, "write_pip_freeze", lambda d: None)
    with pytest.raises(b.BuildError, match="pip_freeze.txt"):
        b.build(b.parse_args(["--trial", "--out", str(tmp_path)]))
    assert not (tmp_path / f"{b.PACKAGE_NAME}{b.TRIAL_SUFFIX}.zip").exists()


def final_build(tmp_path, monkeypatch, change_rebuilt=None):
    fake_build_steps(monkeypatch)
    rebuilt = tmp_path / "rebuilt"
    make_package(rebuilt)
    write(rebuilt / b.OLD_CASES, '{"train": []}\r\n')
    if change_rebuilt:
        change_rebuilt(rebuilt)
    log = write(tmp_path / "log.txt", "Command: python run_pipeline.py --preset final --device cpu\n"
                                      "✅ Pipeline complete (--preset final): sample in 1.0 min\n")
    out = tmp_path / "out"
    args = b.parse_args(["--rebuilt", str(rebuilt), "--log", str(log), "--out", str(out)])
    return args, out / b.PACKAGE_NAME


def test_the_final_build_compares_the_rebuild_reports_and_zips(tmp_path, monkeypatch):
    args, package = final_build(tmp_path, monkeypatch)
    zip_path = b.build(args)
    report = (package / b.CHECK_NAME).read_text(encoding="utf-8")
    assert "RESULT: every rebuilt output matches" in report
    assert "not a git checkout" in report
    with zipfile.ZipFile(zip_path) as archive:
        assert f"{b.PACKAGE_NAME}/{b.CHECK_NAME}" in archive.namelist()
        assert f"{b.PACKAGE_NAME}/{b.LOG_NAME}" in archive.namelist()


def test_the_final_build_stops_before_zipping_when_an_output_differs(tmp_path, monkeypatch):
    def change(rebuilt):
        (rebuilt / "results/benchmark/false_alarms.tex").write_text("another table\n")
    args, package = final_build(tmp_path, monkeypatch, change)
    with pytest.raises(b.BuildError, match="not zipped"):
        b.build(args)
    report = (package / b.CHECK_NAME).read_text(encoding="utf-8")
    assert "NOT REPRODUCED" in report
    assert any(line.startswith("DIFFERENT ") and "results/benchmark/false_alarms.tex" in line
               for line in report.splitlines())
    assert not (package.parent / f"{b.PACKAGE_NAME}.zip").exists()
