"""LN's t-test: differential expression on an asymptotically unbiased log-fold-change estimator.

    from lntest import get_LN_lfcs, rank_genes_groups_ln
"""
from __future__ import annotations

from ._ln_test import (
    get_LN_lfcs,
    get_LN_lfcs_sparse,
    trigamma_diff,
)
from ._rank_genes_groups import CORR_METHODS, rank_genes_groups_ln

def _detect_version() -> str:
    # In a function, so that importlib.metadata's names stay out of dir(lntest).
    try:  # pragma: no cover - fails only for a tree with no installed metadata
        from importlib.metadata import version

        # The distribution name, which differs from the module name.
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
