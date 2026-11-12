"""The multiple-testing correction of ``rank_genes_groups_ln`` matches scanpy's.

The accepted methods are scanpy's, read from scanpy at test time. scanpy's own
p-values, passed through the correction of this package, give scanpy's adjusted
p-values for both methods.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

import anndata  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

from lntest import CORR_METHODS  # noqa: E402
from lntest._rank_genes_groups import _adjust_pvalues  # noqa: E402


@pytest.fixture(scope="module")
def scanpy_run():
    """scanpy's own DE output: p-values and the adjusted p-values it derived."""
    rng = np.random.default_rng(0)
    X = rng.poisson(rng.gamma(1.0, 5.0, size=(120, 60))).astype(np.float32)
    ad = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"g": pd.Categorical(["a"] * 60 + ["b"] * 60)},
                         index=[f"c{i}" for i in range(120)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(60)]),
    )
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def scanpy_de(ad, corr_method):
    out = ad.copy()
    sc.tl.rank_genes_groups(out, "g", method="t-test", corr_method=corr_method,
                            key_added="ref")
    p = np.asarray(out.uns["ref"]["pvals"]["a"], dtype=float)
    q = np.asarray(out.uns["ref"]["pvals_adj"]["a"], dtype=float)
    return p, q


class TestVocabularyMatchesScanpy:
    def test_we_accept_exactly_what_scanpy_accepts(self, scanpy_run):
        """scanpy's accepted methods, read from the error it raises, since ``_CorrMethod`` is not importable."""
        import re

        with pytest.raises(ValueError) as excinfo:
            sc.tl.rank_genes_groups(scanpy_run.copy(), "g",
                                    corr_method="definitely-not-a-method")
        scanpy_accepts = set(re.findall(r"'([a-z-]+)'", str(excinfo.value)))
        assert scanpy_accepts, "could not parse scanpy's accepted methods"
        assert set(CORR_METHODS) == scanpy_accepts

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="corr_method"):
            _adjust_pvalues(np.array([0.1, 0.2]), "fdr_bh", 2)

    def test_statsmodels_spelling_is_not_silently_accepted(self):
        """`fdr_bh`, statsmodels' name, is not accepted."""
        assert "fdr_bh" not in CORR_METHODS

    def test_default_is_scanpys(self):
        """Switching ``method=`` must not also switch the correction."""
        import inspect

        from lntest import rank_genes_groups_ln

        def default(f):
            return inspect.signature(f).parameters["corr_method"].default

        assert default(rank_genes_groups_ln) == default(sc.tl.rank_genes_groups)


class TestReproducesScanpysAdjustment:
    @pytest.mark.parametrize("method", CORR_METHODS)
    def test_our_correction_lands_on_scanpys_numbers(self, scanpy_run, method):
        p, q_scanpy = scanpy_de(scanpy_run, method)
        assert not np.any(np.isnan(p)), "fixture should not produce NaN p-values"
        q_ours = _adjust_pvalues(p, method, n_genes=p.size)
        np.testing.assert_allclose(q_ours, q_scanpy, rtol=1e-6, atol=1e-12)

    def test_bonferroni_uses_the_gene_count_not_the_array_length(self):
        """A truncated result is corrected over all genes, as in scanpy."""
        p = np.array([0.01, 0.02])
        np.testing.assert_allclose(_adjust_pvalues(p, "bonferroni", 100),
                                   [1.0, 1.0])
        np.testing.assert_allclose(_adjust_pvalues(p, "bonferroni", 2),
                                   [0.02, 0.04])

    def test_nan_pvalues_are_treated_as_one_before_correcting(self):
        """NaN p-values count as 1 before the correction, as ``pvals[np.isnan(pvals)] = 1`` does in scanpy."""
        p = np.array([0.001, np.nan, 0.5])
        q = _adjust_pvalues(p, "benjamini-hochberg", 3)
        assert np.all(np.isfinite(q))
        assert q[1] == pytest.approx(1.0)

    def test_input_array_is_not_modified(self):
        p = np.array([0.001, np.nan, 0.5])
        before = p.copy()
        _adjust_pvalues(p, "benjamini-hochberg", 3)
        np.testing.assert_array_equal(p, before, err_msg="corrected in place")
