import sys

def enable_utf8_output() -> None:
    """
    Switches stdout/stderr to UTF-8 so the scripts' emoji and Greek-letter
    status messages cannot crash a run on Windows. When output there is
    redirected (to a log file, a pipe, or run_pipeline.py's stages under a
    redirected parent), Python encodes it with the locale code page (e.g.
    cp1252), and the first emoji raises UnicodeEncodeError. The call is
    effectively a no-op where the streams already use UTF-8 (Linux, macOS,
    interactive Windows consoles). Call it first in every entry point.
    """
    for stream in (sys.stdout, sys.stderr):
        # Streams replaced by Jupyter/IDEs (or None under pythonw) may lack reconfigure()
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
