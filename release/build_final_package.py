"""
Build the final package for Barak (#46): barak_package_final/ and barak_package_final.zip.

Barak asked for "one self-contained ZIP with all models/checkpoints, metadata
and seeds, raw scores and reference errors, frozen calibration, training logs,
complete code, figures in PDF/PNG, tables in CSV/LaTeX, and instructions to
regenerate everything without retraining", and to "check completeness and
reproducibility before handoff".

The package is the repository at a tag (final-v1 by default), exported with
`git archive` exactly as git stores it (LF line endings on every OS): the code,
the models, the case files, the frozen calibration, PROTOCOL.md and every
result. So the regeneration command runs inside the unpacked package. The
script adds:

    00_README.md              release/PACKAGE_README.md: what is where, how each figure
                              and table is made, and what the results show
    data/cases_metadata.json  the 30 old cases, rebuilt by the package's own
                              src/data/sampler.py (data/ is not in git)
    pip_freeze.txt            the Python packages of the machine that ran the check
    reproduction_check.txt    each output rebuilt from scratch (--rebuilt), compared
                              with its copy in the package
    repro_log.txt             the console log of that run (--log), with the machine's
                              folder names replaced by <repo>, python and ~

Before zipping it checks that every item of #46's list is there, that the 210
checkpoints match models/confirmation/MANIFEST.sha256, that the files whose
SHA-256 PROTOCOL.md records still have it, and that the package's own code refits
the frozen calibration from its source scores. If a check fails, or a rebuilt
output differs, it writes the report and stops without zipping. The ZIP is written with
Python's zipfile (forward-slash paths, which open on macOS, Linux and Windows).
The damaged fields (data/confirmation/hallucinations, about 1.8 GB) are left
out, since the command rebuilds them.

Comparing a rebuilt output with its copy (compare_file): line endings are
ignored. A .csv or .json file whose bytes differ is compared as numbers: every
count, flag and text cell must be equal, and every other number within 1e-9 of
it (relative). A machine with another OS or library build can differ in the
last digits (about 2e-13 for the scores). A .tex file must be equal byte for
byte. A figure (.png, .pdf) whose bytes differ is listed as "DRAWN
DIFFERENTLY", for a look by eye: matplotlib on another OS can place text a
pixel or two apart.

Usage:
    # 1. Rebuild every result in a clean checkout of the tag (about 80 min on a CPU)
    git worktree add ../pinn-final-check final-v1
    cd ../pinn-final-check
    python run_pipeline.py --preset final --device cpu > ../repro_log.txt 2>&1
    cd -
    # 2. Build the package, compare the rebuilt outputs with it, and zip it
    python release/build_final_package.py --rebuilt ../pinn-final-check --log ../repro_log.txt

    # A trial build of the current commit, with no reproduction check
    python release/build_final_package.py --trial --ref HEAD --out <folder>
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PACKAGE_NAME = "barak_package_final"
TRIAL_SUFFIX = "_trial"
DEFAULT_REF = "final-v1"

README_SOURCE = "release/PACKAGE_README.md"
README_NAME = "00_README.md"
CHECK_NAME = "reproduction_check.txt"
LOG_NAME = "repro_log.txt"
FREEZE_NAME = "pip_freeze.txt"
OLD_CASES = "data/cases_metadata.json"
NEW_CASES = "cases/confirmation_cases.json"
MANIFEST = "models/confirmation/MANIFEST.sha256"
PROTOCOL = "PROTOCOL.md"
FROZEN_CALIBRATION = "calibration/frozen_calibration.json"
# The files whose SHA-256 PROTOCOL.md records (sections 2 and 8)
PROTOCOL_HASHED = ("models/confirmation/MANIFEST.sha256",
                   "calibration/source_phs_components_raw.csv",
                   "calibration/frozen_calibration.json")
# The 7 checkpoints of each new case: src/models/checkpoints.STAGES (not imported, to keep torch out of this script)
STAGES = ("adam_00100", "adam_00500", "adam_01000", "adam_02500", "adam_04000", "adam_05000", "lbfgs_final")
N_OLD_CASES = 30
# Rebuilt by the final preset, and so compared with the rebuild
RESULT_DIRS = ("results/accuracy", "results/benchmark", "results/natural_errors", "results/figures")
# Their figures and tables, each as .pdf + .png or .csv + .tex (results/accuracy holds raw errors only)
TABLE_DIRS = ("results/benchmark", "results/natural_errors", "results/figures")

RTOL = 1e-9
TEXT_SUFFIXES = {".csv", ".tex", ".json", ".txt", ".md", ".log"}
FIGURE_SUFFIXES = {".png", ".pdf"}

IDENTICAL = "IDENTICAL"
SAME_NUMBERS = "SAME NUMBERS"
DRAWN_DIFFERENTLY = "DRAWN DIFFERENTLY"
DIFFERENT = "DIFFERENT"
MISSING = "MISSING"
NEW = "NEW"
PASSING = (IDENTICAL, SAME_NUMBERS, DRAWN_DIFFERENTLY)
STATUS_ORDER = (DIFFERENT, MISSING, NEW, DRAWN_DIFFERENTLY, SAME_NUMBERS, IDENTICAL)


class BuildError(RuntimeError):
    """A problem that stops the build, with a message for the person running it."""


# --- git and files ---

def git(*args, cwd=ROOT) -> str:
    """Runs git and returns its stripped output; raises BuildError if git fails."""
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_ref(ref: str) -> str:
    """The full commit hash of a tag, branch or commit."""
    try:
        return git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except BuildError:
        raise BuildError(f"{ref!r} is no tag, branch or commit of this repository"
                         + (" yet: tag the merged PR first" if ref == DEFAULT_REF else "")) from None


def export_tree(ref: str, dest: Path, paths=()) -> None:
    """
    Writes the files of ref into dest as git stores them: git archive with
    core.autocrlf off, so text files keep LF on every OS and the files marked
    -text in .gitattributes keep their exact bytes.

    Inputs:
        ref (str): Tag, branch or commit.
        dest (Path): Folder to write into (created if needed).
        paths (iterable[str]): Only these paths of the tree; empty means all of it.
    """
    result = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", ref, *paths],
                            cwd=str(ROOT), capture_output=True)
    if result.returncode != 0:
        raise BuildError(f"git archive {ref} failed: {result.stderr.decode(errors='replace').strip()}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        tar.extractall(dest, filter="data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(folder: Path) -> list:
    """Every file below folder, as sorted POSIX paths relative to it."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def to_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


# --- the added files ---

def rebuild_old_cases(package_dir: Path) -> None:
    """
    Writes data/cases_metadata.json with the package's own sampler (seed 42,
    deterministic), then sets its line endings to LF, like the rest of the package.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "src.data.sampler"], cwd=str(package_dir), env=env,
                            capture_output=True, text=True)
    out = package_dir / OLD_CASES
    if result.returncode != 0 or not out.is_file():
        raise BuildError(f"src/data/sampler.py failed in {package_dir}:\n{result.stdout}\n{result.stderr}")
    out.write_bytes(to_lf(out.read_bytes()))


def write_pip_freeze(package_dir: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"pip freeze failed: {result.stderr.strip()}")
    (package_dir / FREEZE_NAME).write_bytes(to_lf(result.stdout.encode("utf-8")))


# --- checks of the package itself ---

def expected_files(package_dir: Path) -> list:
    """
    #46's list of what the ZIP must contain, as (item, [paths]). The new cases
    are read from the package's case file.
    """
    with open(package_dir / NEW_CASES, encoding="utf-8") as f:
        new_ids = [case["case_id"] for case in json.load(f)["confirmation"]]
    checkpoints = [f"models/confirmation/{c}/{stage}{ext}" for c in new_ids for stage in STAGES
                   for ext in (".pth", ".json")]
    return [
        ("Models", [f"models/case_{n:02d}.pth" for n in range(N_OLD_CASES)] + checkpoints + [MANIFEST]),
        ("Cases and seeds", [OLD_CASES, NEW_CASES]),
        ("Raw scores and true errors", ["calibration/source_phs_components_raw.csv",
                                        "results/benchmark/confirmation_scores_raw.csv",
                                        "results/natural_errors/checkpoint_scores.csv",
                                        "results/accuracy/reference_errors_existing.csv",
                                        "results/accuracy/reference_errors_confirmation.csv"]),
        ("Locked calibration and rules", ["calibration/frozen_calibration.json", PROTOCOL]),
        ("Training logs", [f"models/confirmation/{c}/loss_history.json" for c in new_ids]
                          + ["models/confirmation/loss_summary.csv", "models/confirmation/training_summary.log",
                             "models/confirmation/RESTARTS.md"]),
        ("Code", ["run_pipeline.py", "requirements.txt", "README.md", "PIPELINE.md",
                  "release/build_final_package.py", FREEZE_NAME]),
        ("Every false alarm and every miss", ["results/benchmark/false_alarms.csv",
                                              "results/benchmark/missed_detections.csv",
                                              "results/benchmark/clean_false_positives.csv",
                                              "results/benchmark/preflagged_cases.csv",
                                              "results/natural_errors/false_alarms.csv",
                                              "results/natural_errors/missed_failures.csv",
                                              "results/natural_errors/inaccurate_final_models.csv"]),
        ("The README", [README_NAME]),
    ]


def check_contents(package_dir: Path) -> list:
    """
    Problems with the package's contents (an empty list means none): a missing
    file of #46's list, a result folder with no files, a table without both its
    .csv and its .tex, or a figure without both its .pdf and its .png.
    """
    problems = []
    for item, paths in expected_files(package_dir):
        missing = [p for p in paths if not (package_dir / p).is_file()]
        if missing:
            shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            problems.append(f"{item}: missing {shown}")
    for folder in RESULT_DIRS:
        names = files_under(package_dir / folder) if (package_dir / folder).is_dir() else []
        if not names:
            problems.append(f"Figures and tables: {folder}/ has no files")
        if folder not in TABLE_DIRS:
            continue
        stems = {}
        for name in names:
            stem, suffix = os.path.splitext(name)
            stems.setdefault(stem, set()).add(suffix)
        for stem, suffixes in sorted(stems.items()):
            if ".csv" in suffixes and ".tex" not in suffixes:
                problems.append(f"Figures and tables: {folder}/{stem}.csv has no .tex")
            if suffixes & FIGURE_SUFFIXES and not FIGURE_SUFFIXES <= suffixes:
                problems.append(f"Figures and tables: {folder}/{stem} has only {', '.join(sorted(suffixes))}")
    return problems


def check_manifest(package_dir: Path) -> tuple:
    """
    Checks models/confirmation/ against MANIFEST.sha256, as `sha256sum -c` would,
    and that it lists every file there (except itself, *.png and *.tmp, which the
    #40 command leaves out).

    Outputs:
        (int, list[str]): the number of files listed, and the problems (empty if none).
    """
    folder = (package_dir / MANIFEST).parent
    listed, problems = {}, []
    for line in (package_dir / MANIFEST).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]\./(.+)", line)
        if not match:
            problems.append(f"{MANIFEST}: unreadable line {line!r}")
            continue
        listed[match.group(2)] = match.group(1)
    for name, digest in listed.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{MANIFEST} lists {name}, which is missing")
        elif sha256(path) != digest:
            problems.append(f"{name} does not match {MANIFEST}")
    present = [n for n in files_under(folder)
               if n != Path(MANIFEST).name and not n.endswith((".png", ".tmp"))]
    unlisted = sorted(set(present) - set(listed))
    if unlisted:
        problems.append(f"{MANIFEST} does not list {', '.join(unlisted)}")
    return len(listed), problems


def protocol_hashes(protocol_text: str) -> dict:
    """
    The SHA-256 that PROTOCOL.md records for each file of PROTOCOL_HASHED: on the
    line that names the file in backticks, the first backticked 64-digit hash after it.
    """
    found = {}
    for path in PROTOCOL_HASHED:
        for line in protocol_text.splitlines():
            at = line.find(f"`{path}`")
            if at >= 0:
                match = re.search(r"`([0-9a-f]{64})`", line[at:])
                if match:
                    found[path] = match.group(1)
                    break
    return found


def check_protocol_hashes(package_dir: Path) -> list:
    """Problems with the files whose SHA-256 PROTOCOL.md records (empty if none)."""
    recorded = protocol_hashes((package_dir / PROTOCOL).read_text(encoding="utf-8"))
    problems = [f"{PROTOCOL} records no SHA-256 for {p}" for p in PROTOCOL_HASHED if p not in recorded]
    for path, digest in recorded.items():
        if sha256(package_dir / path) != digest:
            problems.append(f"{path} does not have the SHA-256 that {PROTOCOL} records ({digest[:12]}...)")
    return problems


def check_calibration_refit(package_dir: Path) -> list:
    """
    Refits the calibration from calibration/source_phs_components_raw.csv with the
    package's own src/detection/frozen_calibration.py, into a temporary file, and
    checks that it equals calibration/frozen_calibration.json: every field but the
    commit it was written at (git_commit, git_dirty), numbers within RTOL.

    Outputs:
        list[str]: the problems (empty if none).
    """
    import tempfile

    frozen = package_dir / FROZEN_CALIBRATION
    with tempfile.TemporaryDirectory() as tmp:
        refit = Path(tmp) / "refit.json"
        result = subprocess.run([sys.executable, "-m", "src.detection.frozen_calibration", "--out", str(refit)],
                                cwd=str(package_dir), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                capture_output=True, text=True)
        if result.returncode != 0 or not refit.is_file():
            return [f"refitting {FROZEN_CALIBRATION} failed:\n{result.stdout}\n{result.stderr}"]
        refitted = json.loads(refit.read_text(encoding="utf-8"))
    stored = json.loads(frozen.read_text(encoding="utf-8"))
    for calib in (stored, refitted):
        for key in ("git_commit", "git_dirty"):
            calib.pop(key, None)
    d = _json_difference(stored, refitted)
    if isinstance(d, str) or d > RTOL:
        return [f"{FROZEN_CALIBRATION} is not what its source refits to: "
                + (d if isinstance(d, str) else f"a number differs by {d:.1e} (relative)")]
    return []


# --- comparing a rebuilt output with its copy in the package ---

def _relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(a), np.abs(b))
    return np.where(scale > 0, np.abs(a - b) / np.where(scale > 0, scale, 1.0), 0.0)


def compare_csv(packaged: Path, rebuilt: Path) -> tuple:
    a = pd.read_csv(packaged, float_precision="round_trip")
    b = pd.read_csv(rebuilt, float_precision="round_trip")
    if list(a.columns) != list(b.columns):
        return DIFFERENT, "the columns differ"
    if len(a) != len(b):
        return DIFFERENT, f"{len(a)} rows here, {len(b)} rebuilt"
    worst, worst_column = 0.0, None
    for column in a.columns:
        x, y = a[column], b[column]
        numeric = (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
                   and not pd.api.types.is_bool_dtype(x) and not pd.api.types.is_bool_dtype(y))
        if numeric and not (pd.api.types.is_integer_dtype(x) and pd.api.types.is_integer_dtype(y)):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            special = ~(np.isfinite(xv) & np.isfinite(yv))
            if not np.array_equal(xv[special], yv[special], equal_nan=True):
                return DIFFERENT, f"{column}: empty (NaN) or infinite in different rows"
            finite = ~special
            if finite.any():
                rel = float(_relative_difference(xv[finite], yv[finite]).max())
                if rel > worst:
                    worst, worst_column = rel, column
        else:
            # Empty cells read as NaN (pandas 3 keeps them NaN in text columns too), and NaN != NaN
            differs = ~((x == y) | (x.isna() & y.isna())).to_numpy(dtype=bool)
            if differs.any():
                row = int(np.flatnonzero(differs)[0])
                return DIFFERENT, f"{column} differs in row {row + 2}: {x.iloc[row]!r} here, {y.iloc[row]!r} rebuilt"
    if worst > RTOL:
        return DIFFERENT, f"{worst_column} differs by up to {worst:.1e} (relative)"
    return SAME_NUMBERS, f"largest relative difference {worst:.1e} ({worst_column})"


def _json_difference(x, y, where="the file"):
    """The largest relative difference between two parsed JSON values, or a string saying where they differ."""
    if isinstance(x, dict) and isinstance(y, dict):
        if list(x) != list(y):
            return f"the keys of {where} differ"
        worst = 0.0
        for key in x:
            d = _json_difference(x[key], y[key], f"{where}/{key}")
            if isinstance(d, str):
                return d
            worst = max(worst, d)
        return worst
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return f"the lengths of {where} differ"
        worst = 0.0
        for i, (xi, yi) in enumerate(zip(x, y)):
            d = _json_difference(xi, yi, f"{where}[{i}]")
            if isinstance(d, str):
                return d
            worst = max(worst, d)
        return worst
    numbers = (int, float)
    if (isinstance(x, numbers) and isinstance(y, numbers) and not isinstance(x, bool) and not isinstance(y, bool)
            and (isinstance(x, float) or isinstance(y, float))):
        return float(_relative_difference(np.array([float(x)]), np.array([float(y)]))[0])
    return 0.0 if (type(x) is type(y) and x == y) else f"{where} differs: {x!r} here, {y!r} rebuilt"


def compare_json(packaged: Path, rebuilt: Path) -> tuple:
    d = _json_difference(json.loads(packaged.read_text(encoding="utf-8")),
                         json.loads(rebuilt.read_text(encoding="utf-8")))
    if isinstance(d, str):
        return DIFFERENT, d
    if d > RTOL:
        return DIFFERENT, f"a number differs by {d:.1e} (relative)"
    return SAME_NUMBERS, f"largest relative difference {d:.1e}"


def compare_png(packaged: Path, rebuilt: Path) -> tuple:
    import matplotlib.image as mpimg

    a, b = mpimg.imread(packaged), mpimg.imread(rebuilt)
    if a.shape != b.shape:
        return DRAWN_DIFFERENTLY, f"{a.shape[1]}x{a.shape[0]} px here, {b.shape[1]}x{b.shape[0]} px rebuilt"
    changed = np.abs(a.astype(float) - b.astype(float))
    if changed.ndim == 3:
        changed = changed.max(axis=2)
    return DRAWN_DIFFERENTLY, f"same size, {100 * float((changed > 0).mean()):.2f}% of the pixels differ"


def compare_file(packaged: Path, rebuilt: Path) -> tuple:
    """
    Compares one rebuilt output with its copy in the package, by the rules in the
    module docstring.

    Outputs:
        (str, str): the status (IDENTICAL, SAME NUMBERS, DRAWN DIFFERENTLY or
                    DIFFERENT) and a short detail (may be empty).
    """
    a, b = packaged.read_bytes(), rebuilt.read_bytes()
    if a == b:
        return IDENTICAL, ""
    suffix = packaged.suffix.lower()
    if suffix in TEXT_SUFFIXES and to_lf(a) == to_lf(b):
        return IDENTICAL, "apart from line endings"
    try:
        if suffix == ".csv":
            return compare_csv(packaged, rebuilt)
        if suffix == ".json":
            return compare_json(packaged, rebuilt)
        if suffix == ".png":
            return compare_png(packaged, rebuilt)
    except Exception as err:          # an unreadable file is a difference, not a crash of the check
        return DIFFERENT, f"could not be read and compared ({type(err).__name__}: {err})"
    if suffix == ".pdf":
        return DRAWN_DIFFERENTLY, "the bytes differ"
    return DIFFERENT, "the bytes differ"


def compared_paths(package_dir: Path) -> list:
    """The package's files that the final preset rebuilds: everything in results/, and the old cases."""
    paths = [OLD_CASES]
    for folder in RESULT_DIRS:
        paths += [f"{folder}/{name}" for name in files_under(package_dir / folder)]
    return paths


def compare_outputs(package_dir: Path, rebuilt_dir: Path) -> list:
    """
    Compares every output the final preset rebuilds with its copy in the package.

    Outputs:
        list[(status, path, detail)]: one row per file, including MISSING (not
        rebuilt) and NEW (rebuilt, but not in the package).
    """
    rows = []
    paths = compared_paths(package_dir)
    for path in paths:
        rebuilt = rebuilt_dir / path
        if not rebuilt.is_file():
            rows.append((MISSING, path, "not rebuilt"))
        else:
            status, detail = compare_file(package_dir / path, rebuilt)
            rows.append((status, path, detail))
    known = set(paths)
    for folder in RESULT_DIRS:
        if (rebuilt_dir / folder).is_dir():
            rows += [(NEW, f"{folder}/{name}", "rebuilt, but not in the package")
                     for name in files_under(rebuilt_dir / folder) if f"{folder}/{name}" not in known]
    # A PDF stores its coordinates as text, so a last-digit change in the data can alter its bytes; its PNG says more
    status_of = {path: status for status, path, _ in rows}
    for i, (status, path, detail) in enumerate(rows):
        if status == DRAWN_DIFFERENTLY and path.endswith(".pdf") and status_of.get(path[:-4] + ".png") == IDENTICAL:
            rows[i] = (status, path, f"{detail}; its .png is identical")
    return rows


def check_rebuilt_checkout(rebuilt_dir: Path, commit: str) -> tuple:
    """
    Checks that rebuilt_dir is a checkout of the package's commit, and that the run
    changed no tracked file outside results/.

    Outputs:
        (str, list[str]): a sentence for the report, and the problems (empty if none).
    """
    try:
        head = git("rev-parse", "HEAD", cwd=rebuilt_dir)
    except BuildError:
        return "The rebuilt folder is not a git checkout, so its commit was not checked.", []
    if head != commit:
        return "", [f"{rebuilt_dir} is at commit {head[:7]}, not at the package's {commit[:7]}"]
    # git diff compares contents with line endings normalized; git status would also list files whose
    # line endings alone changed (the scripts write LF, a Windows checkout has CRLF)
    changed = git("diff", "--name-only", cwd=rebuilt_dir).splitlines()
    outside = [p for p in changed if not p.startswith("results/")]
    if outside:
        return "", [f"the rebuild changed tracked files outside results/: {', '.join(outside)}"]
    return (f"The run was made in a checkout of commit {commit[:7]}, and it changed the contents of no "
            f"tracked file outside results/. In results/, {len(changed)} rebuilt files differ from git in "
            "their bytes, line endings aside; they are compared below.", [])


# --- the console log ---

def read_log(log_path: Path, rebuilt_dir: Path) -> tuple:
    """
    Reads the console log of the rebuild and hides the machine's folder names.

    Outputs:
        (str, str, str): the log with the interpreter's path shown as "python", the
        rebuilt checkout as "<repo>" and the home folder as "~"; the command line;
        and the total time, from run_pipeline.py's last line.
    """
    text = log_path.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    command = re.search(r"^Command: (python run_pipeline\.py.*)$", text, re.M)
    done = re.search(r"Pipeline complete \(--preset final\).* in ([0-9.]+) min", text)
    if not command or not done:
        raise BuildError(f"{log_path} is not the log of a finished `run_pipeline.py --preset final` run")

    interpreters = set(re.findall(r"^> (.+?) -m src\.", text, re.M))
    replacements = [(path, "python") for path in sorted(interpreters, key=len, reverse=True)]
    replacements += [(str(rebuilt_dir.resolve()), "<repo>"), (str(Path.home()), "~")]
    for old, new in replacements:
        for variant in {old, old.replace("\\", "/")}:
            text = re.sub(re.escape(variant), lambda _: new, text, flags=re.I)
    return text, command.group(1).strip(), done.group(1)


# --- the report ---

def machine_line() -> str:
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"
    return (f"{platform.system()} {platform.release()} ({platform.machine()}), {os.cpu_count()} CPU threads; "
            f"Python {platform.python_version()}; torch {version('torch')}, numpy {version('numpy')} "
            f"(all packages: {FREEZE_NAME})")


def write_report(package_dir: Path, ref: str, commit: str, rows: list, package_checks: list, problems: list,
                 command: str, minutes: str, checkout_note: str) -> bool:
    """
    Writes reproduction_check.txt. Returns True when every rebuilt output matches
    (figures drawn differently are listed but allowed) and the package passed its checks.
    """
    counts = {status: sum(1 for row in rows if row[0] == status) for status in STATUS_ORDER}
    reproduced = not problems and all(row[0] in PASSING for row in rows)
    width = max(len(s) for s in STATUS_ORDER) + 2
    summary = ", ".join(f"{counts[s]} {s.lower()}" for s in reversed(STATUS_ORDER) if counts[s])
    lines = [
        f"REPRODUCTION CHECK ({datetime.date.today().isoformat()})",
        "",
        f"Code:     {ref} (commit {commit})",
        f"Machine:  {machine_line()}",
        f"Command:  {command}",
        f"Result:   finished in {minutes} min, with no training; console log: {LOG_NAME}",
        "",
        *([checkout_note, ""] if checkout_note else []),
        "How each rebuilt output was compared with its copy in this package: line endings are ignored. A .csv or",
        "a .json file whose bytes differ is compared as numbers: every count, flag and text cell must be equal,",
        f"and every other number within {RTOL:.0e} of it (relative). A .tex file must be equal byte for byte. A".replace("e-0", "e-"),
        "figure whose bytes differ is listed as DRAWN DIFFERENTLY, for a look by eye: matplotlib on another OS",
        "can place text a pixel or two apart.",
        "",
        f"Outputs compared: {len(rows)}: {summary}.",
        ("RESULT: every rebuilt output matches its copy in this package." if reproduced else
         "RESULT: NOT REPRODUCED. See the rows and problems below."),
        "",
    ]
    for status in STATUS_ORDER:
        for row_status, path, detail in rows:
            if row_status == status:
                lines.append(f"{status:<{width}}{path}" + (f"   ({detail})" if detail else ""))
    lines += ["", "Checks of the package itself:"] + [f"- {check}" for check in package_checks]
    if problems:
        lines += ["", "Problems:"] + [f"- {problem}" for problem in problems]
    (package_dir / CHECK_NAME).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return reproduced


# --- the ZIP ---

def make_zip(package_dir: Path, zip_path: Path) -> int:
    """
    Zips package_dir into zip_path, under one top folder named like package_dir,
    with forward-slash paths in sorted order, then reads it back to check every
    file's CRC. Returns the number of files.
    """
    names = [n for n in files_under(package_dir) if "__pycache__" not in n.split("/")]
    partial = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(package_dir / name, f"{package_dir.name}/{name}")
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad is not None or len(archive.namelist()) != len(names):
            raise BuildError(f"the ZIP did not read back correctly (first bad file: {bad})")
    os.replace(partial, zip_path)
    return len(names)


# --- main ---

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the final package for Barak (#46): the repository at a "
                                                 "tag, checked, compared with a rebuild from scratch, and zipped.")
    parser.add_argument("--ref", type=str, default=DEFAULT_REF,
                        help=f"The tag (or commit) to package (default: {DEFAULT_REF}).")
    parser.add_argument("--out", type=str, default=str(ROOT),
                        help=f"Folder to write {PACKAGE_NAME}/ and {PACKAGE_NAME}.zip into (default: the "
                             "repository root, where .gitignore keeps them out of git).")
    parser.add_argument("--rebuilt", type=str, default=None,
                        help="The checkout of the same commit where `python run_pipeline.py --preset final` ran; "
                             f"its outputs are compared with the package's copies into {CHECK_NAME}.")
    parser.add_argument("--log", type=str, default=None,
                        help=f"The console log of that run, copied into the package as {LOG_NAME}.")
    parser.add_argument("--trial", action="store_true",
                        help=f"Build without the reproduction check, as {PACKAGE_NAME}{TRIAL_SUFFIX}/ and .zip.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the package folder and ZIP if they exist.")
    args = parser.parse_args(argv)
    if args.trial and (args.rebuilt or args.log):
        parser.error("--trial builds without the check; leave out --rebuilt and --log")
    if not args.trial and not (args.rebuilt and args.log):
        parser.error("the final package needs its reproduction check: pass --rebuilt and --log "
                     "(see the module docstring), or --trial for a trial build")
    return args


def build(args) -> Path:
    """Builds, checks and zips the package. Returns the ZIP's path; raises BuildError on a failed check."""
    commit = resolve_ref(args.ref)
    name = PACKAGE_NAME + (TRIAL_SUFFIX if args.trial else "")
    out = Path(args.out).resolve()
    package_dir, zip_path = out / name, out / f"{name}.zip"
    for existing in (package_dir, zip_path):
        if existing.exists() and not args.overwrite:
            raise BuildError(f"{existing} already exists; pass --overwrite to replace it")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    zip_path.unlink(missing_ok=True)

    print(f"Exporting {args.ref} ({commit[:7]}) into {package_dir}", flush=True)
    export_tree(commit, package_dir)
    if not (package_dir / README_SOURCE).is_file():
        raise BuildError(f"{args.ref} has no {README_SOURCE}")
    shutil.copyfile(package_dir / README_SOURCE, package_dir / README_NAME)
    print(f"Rebuilding {OLD_CASES} and writing {FREEZE_NAME}", flush=True)
    rebuild_old_cases(package_dir)
    write_pip_freeze(package_dir)

    rows, command, minutes, checkout_note, checkout_problems = [], "", "", "", []
    if not args.trial:
        rebuilt_dir = Path(args.rebuilt).resolve()
        log_text, command, minutes = read_log(Path(args.log), rebuilt_dir)
        (package_dir / LOG_NAME).write_bytes(log_text.encode("utf-8"))
        checkout_note, checkout_problems = check_rebuilt_checkout(rebuilt_dir, commit)
        print(f"Comparing the outputs rebuilt in {rebuilt_dir} with the package", flush=True)
        rows = compare_outputs(package_dir, rebuilt_dir)

    print("Checking the package", flush=True)
    content_problems = check_contents(package_dir)
    n_listed, manifest_problems = check_manifest(package_dir)
    protocol_problems = check_protocol_hashes(package_dir)
    refit_problems = check_calibration_refit(package_dir)
    problems = content_problems + manifest_problems + protocol_problems + refit_problems + checkout_problems
    package_checks = [
        "every item of the list in #46 is present, and every table has its .csv and .tex and every figure "
        "its .pdf and .png" if not content_problems else "items are missing (see Problems)",
        f"{MANIFEST}: " + (f"all {n_listed} files listed in it match, and it lists every file"
                           if not manifest_problems else "mismatch (see Problems)"),
        f"the {len(PROTOCOL_HASHED)} files whose SHA-256 {PROTOCOL} records ("
        + ", ".join(PROTOCOL_HASHED) + "): " + ("all match" if not protocol_problems else "mismatch (see Problems)"),
        f"{FROZEN_CALIBRATION}, refitted from its source scores by src/detection/frozen_calibration.py: "
        + ("the same normalizers and tau" if not refit_problems else "different (see Problems)"),
    ]

    if not args.trial:
        reproduced = write_report(package_dir, args.ref, commit, rows, package_checks, problems, command,
                                  minutes, checkout_note)
        print((package_dir / CHECK_NAME).read_text(encoding="utf-8"), flush=True)
        if not reproduced:
            raise BuildError(f"not zipped: see {package_dir / CHECK_NAME}")
    elif problems:
        raise BuildError("not zipped:\n" + "\n".join(f"- {p}" for p in problems))

    print(f"Zipping into {zip_path}", flush=True)
    n_files = make_zip(package_dir, zip_path)
    print(f"Wrote {zip_path}: {n_files} files, {zip_path.stat().st_size / 2**20:.1f} MB", flush=True)
    return zip_path


def main(argv=None):
    args = parse_args(argv)
    try:
        build(args)
    except BuildError as err:
        sys.exit(f"build_final_package.py: {err}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.utils.console import enable_utf8_output
    enable_utf8_output()
    main()
