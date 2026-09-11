"""Arm-relative paths, and inputs that fail by name.

Every entry point resolves its data and results from its own location rather
than from the working directory. That matters here because the arms disagreed
about what the working directory was -- `clustering/config.yaml` assumed one
thing and `celltype/config.yaml` another, and four R scripts assumed a third --
which is a large part of why only one arm reproduced.

`require_input` makes an entry point check its inputs exist and exit with the
missing path named. Three arms read data that lives on a colleague's machine; the most this repository can
do for them is fail immediately and say whose file is missing and where it comes
from, instead of dying part-way through a run.
"""

from __future__ import annotations

import pathlib
import sys


def arm_dir(module_file) -> pathlib.Path:
    """The arm folder containing the calling script."""
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
        f"This arm cannot run without it. Nothing has been written.\n"
    )
