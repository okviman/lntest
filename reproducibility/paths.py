"""Paths relative to the directory of each experiment, with inputs that fail by name.

Every entry point resolves its data and results from its own location, not from the working directory.
`require_input` exits naming a missing input and where it comes from, before anything is written.
"""

from __future__ import annotations

import os
import pathlib
import sys


def arm_dir(module_file) -> pathlib.Path:
    """The experiment directory containing the calling script."""
    return pathlib.Path(module_file).resolve().parent


def data_dir(module_file) -> pathlib.Path:
    return arm_dir(module_file) / "data"


def results_dir(module_file, *, create: bool = True) -> pathlib.Path:
    d = arm_dir(module_file) / "results"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def require_input(path, *, what: str, source: str) -> pathlib.Path:
    """Return ``path``, or exit naming what is missing and where it comes from."""
    p = pathlib.Path(path)
    if p.exists():
        return p
    raise SystemExit(
        f"\nMISSING INPUT\n"
        f"  path   : {p}\n"
        f"  what   : {what}\n"
        f"  source : {source}\n\n"
        f"This script cannot run without it. Nothing has been written.\n"
    )


FIG_EXT = os.environ.get("LNTEST_FIG_FORMAT", "pdf").lstrip(".")

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")


_FIG_SUFFIXES = {".png", ".pdf", ".eps", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}


def fig_name(stem) -> str:
    """Figure filename for ``stem``, replacing an image extension if it has one."""
    p = pathlib.PurePath(str(stem))
    if p.suffix.lower() in _FIG_SUFFIXES:
        p = p.with_suffix("")
    return f"{p}.{FIG_EXT}"
