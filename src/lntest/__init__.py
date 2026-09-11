"""LN's t-test: differential expression on an asymptotically unbiased log-fold-change estimator.

    from lntest import get_LN_lfcs, rank_genes_groups_ln

This module is the whole public surface. The two implementation modules are
private, following scanpy, which keeps ``rank_genes_groups`` in
``scanpy/tools/_rank_genes_groups.py`` and exposes only the name.

Everything here imports eagerly, and can, because the package depends on numpy,
scipy and statsmodels alone -- and statsmodels only inside the Benjamini-Hochberg
branch that needs it, which is again what scanpy does.
"""
from __future__ import annotations

from ._ln_test import (
    get_LN_lfcs,
    get_LN_lfcs_sparse,
    trigamma_diff,
)
from ._rank_genes_groups import CORR_METHODS, rank_genes_groups_ln

def _detect_version() -> str:
    # Scoped in a function so importlib.metadata's names do not land in the
    # package namespace; `dir(lntest)` should show the API and nothing else.
    try:  # pragma: no cover - fails only for a tree with no installed metadata
        from importlib.metadata import version

        # The distribution name, not this module's name. They differ: PyPI
        # rejected "lntest" as too close to the existing "intest".
        return version("ln-ttest")
    except Exception:
        return "0.0.0.dev0"


__version__ = _detect_version()
del _detect_version

__all__ = [
    "CORR_METHODS",
    "get_LN_lfcs",
    "get_LN_lfcs_sparse",
    "rank_genes_groups_ln",
    "trigamma_diff",
    "__version__",
]


def __dir__():
    return sorted(__all__)
