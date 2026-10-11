"""The multiple-testing correction must stay interchangeable with scanpy's.

``rank_genes_groups_ln`` is meant to be usable where ``sc.tl.rank_genes_groups``
is, and one day dispatched to from inside it. That makes the correction a
compatibility surface, not an implementation detail: a user who switches
``method=`` and gets differently-adjusted p-values has found a bug in us, not a
feature.

So this file does not check the arithmetic against a textbook. It checks it
against scanpy:

* the accepted vocabulary is scanpy's ``avail_corr``, read from scanpy at test
  time so that a change upstream shows up here rather than in a user's results;
* feeding scanpy's own p-values through our correction reproduces scanpy's own
  adjusted p-values, for both methods.

The second is the one that matters. It is not circular -- scanpy's numbers come
out of ``sc.tl.rank_genes_groups`` on real data, and ours have to land on them.
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
        """Reads scanpy's own set at runtime, so an upstream change is caught
        here rather than in a user's results.

        The set is taken from the error scanpy raises rather than from a private
        name: ``_CorrMethod`` lives under ``TYPE_CHECKING`` and is not importable,
        and the message is the part users actually see.
        """
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
        """`fdr_bh` is statsmodels' name, not scanpy's. Rejecting it keeps the
        two vocabularies from quietly diverging."""
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
        """n_genes is passed in because a truncated result must still be
        corrected over the full family, exactly as scanpy does."""
        p = np.array([0.01, 0.02])
        np.testing.assert_allclose(_adjust_pvalues(p, "bonferroni", 100),
                                   [1.0, 1.0])
        np.testing.assert_allclose(_adjust_pvalues(p, "bonferroni", 2),
                                   [0.02, 0.04])

    def test_nan_pvalues_are_treated_as_one_before_correcting(self):
        """scanpy does `pvals[np.isnan(pvals)] = 1` first; so do we."""
        p = np.array([0.001, np.nan, 0.5])
        q = _adjust_pvalues(p, "benjamini-hochberg", 3)
        assert np.all(np.isfinite(q))
        assert q[1] == pytest.approx(1.0)

    def test_input_array_is_not_modified(self):
        p = np.array([0.001, np.nan, 0.5])
        before = p.copy()
        _adjust_pvalues(p, "benjamini-hochberg", 3)
        np.testing.assert_array_equal(p, before, err_msg="corrected in place")
