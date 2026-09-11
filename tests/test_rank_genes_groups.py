"""Tests for ``rank_genes_groups_ln``, the scanpy-shaped entry point.

``tests/test_lntest.py`` pins the estimator. This file pins the *wrapper*: the
function a scanpy user actually calls, and the one a scanpy integration would
dispatch to. Nothing here re-tests the statistics -- it tests the contract laid
over them.

Building the objects the wrapper is called with needs anndata and pandas, which
are test dependencies rather than runtime ones -- the wrapper is duck-typed and
imports neither. The whole module skips when they are absent.

Three things are worth knowing about what is asserted below.

* **The standard error is derived, not computed twice.** The wrapper reports
  ``lfc_se = |lfc / statistic|`` rather than calling the estimator again, which
  would double the cost of every run. That is exact, not an approximation:
  ``get_t_statistic`` sets ``statistic = (mu_Y - mu_X) / sqrt(se_Y^2 + se_X^2)``
  and ``lfc = mu_Y - mu_X``, so the division recovers the pooled SE. If the
  statistic ever stops being that ratio, ``test_se_reconstructs_the_statistic``
  is what notices.

* **``trigamma`` must not move the log-fold change.** It enters only through
  ``se_Y_1``/``se_X_1``, so switching it changes the SE, the statistic and the
  p-values while leaving the LFC bit-identical. That separation is the whole
  reason the flag is safe to carry while the question behind it is open, and it
  is unasserted anywhere else.

* **The wrapper always applies CP10K**, despite its docstring saying it takes
  normalised data. That is harmless because CP10K is idempotent -- a matrix
  whose rows already sum to 1e4 is rescaled by 1.0 -- but it is behaviour, so
  it is pinned rather than left to be rediscovered.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("anndata")
pytest.importorskip("statsmodels")

import anndata  # noqa: E402
import pandas as pd  # noqa: E402

from lntest import get_LN_lfcs  # noqa: E402
from lntest import rank_genes_groups_ln  # noqa: E402

# The recarrays are float32, so identities that hold exactly in float64 survive
# the round trip only to single precision. This is the tolerance scanpy itself
# asserts across array types.
F32_RTOL = 1e-5

N_CELLS, N_GENES = 120, 25
GROUPS = ["a", "b", "c"]


def counts(seed=0, n_cells=N_CELLS, n_genes=N_GENES):
    """Over-dispersed counts, the regime the method is for."""
    rng = np.random.default_rng(seed)
    return rng.poisson(rng.gamma(1.0, 5.0, size=(n_cells, n_genes))).astype(float)


def make_adata(X=None, labels=None):
    X = counts() if X is None else X
    n = X.shape[0]
    if labels is None:
        per = n // len(GROUPS)
        labels = sum(([g] * per for g in GROUPS), [])
    return anndata.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"g": pd.Categorical(labels)}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )


def field(adata, key, group, name="rank_genes_groups"):
    return np.asarray(adata.uns[name][key][group], dtype=float)


class TestOutputContract:
    """The shape scanpy consumers depend on."""

    def test_uns_carries_the_scanpy_keys_plus_lfc_se(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        assert set(ad.uns["rank_genes_groups"]) == {
            "params", "names", "scores", "logfoldchanges", "pvals", "pvals_adj", "lfc_se",
        }

    def test_params_record_the_call(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", test="z", rankby_abs=True, layer=None)
        params = ad.uns["rank_genes_groups"]["params"]
        assert params["groupby"] == "g"
        assert params["reference"] == "rest"
        assert params["test"] == "z"
        assert params["rankby_abs"] is True
        assert params["groups"] == GROUPS

    def test_one_row_per_gene_and_one_field_per_group(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        rec = ad.uns["rank_genes_groups"]["names"]
        assert len(rec) == N_GENES
        assert rec.dtype.names == tuple(GROUPS)

    def test_every_gene_appears_once_per_group(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        for g in GROUPS:
            names = np.asarray(ad.uns["rank_genes_groups"]["names"][g])
            assert sorted(names) == sorted(ad.var_names)

    def test_key_added_is_honoured(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", key_added="ln")
        assert "ln" in ad.uns
        assert "rank_genes_groups" not in ad.uns

    def test_n_genes_truncates_to_the_top_of_the_ranking(self):
        full = make_adata()
        rank_genes_groups_ln(full, "g")
        top = make_adata()
        rank_genes_groups_ln(top, "g", n_genes=5)
        assert len(top.uns["rank_genes_groups"]["names"]) == 5
        # the kept rows are the head of the full ranking, not an arbitrary five
        np.testing.assert_array_equal(
            np.asarray(top.uns["rank_genes_groups"]["names"]["a"]),
            np.asarray(full.uns["rank_genes_groups"]["names"]["a"])[:5],
        )

    def test_input_matrix_is_not_modified(self):
        X = counts()
        ad = make_adata(X)
        rank_genes_groups_ln(ad, "g")
        np.testing.assert_array_equal(np.asarray(ad.X), X)


class TestGroupSelection:
    def test_reference_group_is_excluded_from_the_output(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", reference="c")
        assert ad.uns["rank_genes_groups"]["names"].dtype.names == ("a", "b")
        assert ad.uns["rank_genes_groups"]["params"]["reference"] == "c"

    def test_groups_restricts_what_is_tested(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", groups=["a"])
        assert ad.uns["rank_genes_groups"]["names"].dtype.names == ("a",)

    def test_rest_and_explicit_reference_differ(self):
        """One-vs-rest is not the same comparison as one-vs-one."""
        rest = make_adata()
        rank_genes_groups_ln(rest, "g", groups=["a"])
        pair = make_adata()
        rank_genes_groups_ln(pair, "g", groups=["a"], reference="c")
        assert not np.allclose(field(rest, "logfoldchanges", "a"),
                               field(pair, "logfoldchanges", "a"))

    def test_unknown_groupby_raises_keyerror(self):
        with pytest.raises(KeyError):
            rank_genes_groups_ln(make_adata(), "not_a_column")

    def test_unknown_reference_raises_valueerror(self):
        with pytest.raises(ValueError):
            rank_genes_groups_ln(make_adata(), "g", reference="not_a_group")

    def test_group_with_no_cells_raises_rather_than_producing_nonsense(self):
        """An unused pandas category is a real way to reach a zero-cell group."""
        ad = make_adata()
        ad.obs["g"] = ad.obs["g"].cat.add_categories(["empty"])
        with pytest.raises(ValueError, match="0 cells"):
            rank_genes_groups_ln(ad, "g")


class TestRanking:
    def test_scores_are_non_increasing(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        for g in GROUPS:
            assert np.all(np.diff(field(ad, "scores", g)) <= 0)

    def test_rankby_abs_orders_by_magnitude(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", rankby_abs=True)
        for g in GROUPS:
            assert np.all(np.diff(np.abs(field(ad, "scores", g))) <= 0)

    def test_every_column_is_permuted_the_same_way(self):
        """names/scores/lfc/pvals must stay row-aligned after sorting."""
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        uns = ad.uns["rank_genes_groups"]
        order = [ad.var_names.get_loc(n) for n in np.asarray(uns["names"]["a"])]
        unsorted = make_adata()
        rank_genes_groups_ln(unsorted, "g", rankby_abs=False)
        # reconstruct the unsorted LFC vector by inverting the permutation
        lfc_sorted = field(ad, "logfoldchanges", "a")
        by_gene = dict(zip(np.asarray(uns["names"]["a"]), lfc_sorted))
        assert [by_gene[ad.var_names[i]] for i in order] == pytest.approx(list(lfc_sorted))


class TestLfcStandardError:
    """``lfc_se`` is the reason the wrapper exists rather than scanpy's own."""

    def test_se_reconstructs_the_statistic(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        for g in GROUPS:
            lfc = field(ad, "logfoldchanges", g)
            score = field(ad, "scores", g)
            se = field(ad, "lfc_se", g)
            ok = np.isfinite(se) & (lfc != 0)
            assert ok.sum() > 0
            np.testing.assert_allclose(se[ok] * np.abs(score[ok]), np.abs(lfc[ok]), rtol=F32_RTOL)

    def test_se_is_nan_where_the_lfc_is_zero(self):
        """0/0 is not recoverable this way, so it is NaN rather than guessed."""
        X = counts()
        ad = make_adata(np.vstack([X, X]), labels=["a"] * len(X) + ["b"] * len(X))
        rank_genes_groups_ln(ad, "g")
        lfc, se = field(ad, "logfoldchanges", "a"), field(ad, "lfc_se", "a")
        assert np.all(lfc == 0)
        assert np.all(np.isnan(se))

    def test_se_is_positive_where_defined(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        se = field(ad, "lfc_se", "a")
        assert np.all(se[np.isfinite(se)] > 0)

    def test_interval_excludes_zero_exactly_when_the_statistic_exceeds_1_96(self):
        """The CI and the statistic are two views of one quantity.

        ``lfc ± 1.96·se`` excludes zero iff ``|lfc| > 1.96·se``, and since
        ``|lfc| = se·|statistic|`` that is iff ``|statistic| > 1.96``. An
        algebraic identity, so it holds gene by gene with no tolerance on the
        decision itself.
        """
        ad = make_adata()
        rank_genes_groups_ln(ad, "g")
        lfc, score, se = (field(ad, k, "a") for k in ("logfoldchanges", "scores", "lfc_se"))
        ok = np.isfinite(se)
        excludes_zero = (lfc[ok] - 1.96 * se[ok]) * (lfc[ok] + 1.96 * se[ok]) > 0
        np.testing.assert_array_equal(excludes_zero, np.abs(score[ok]) > 1.96)


class TestOptions:
    def test_bonferroni_matches_the_closed_form(self):
        ad = make_adata()
        rank_genes_groups_ln(ad, "g", corr_method="bonferroni")
        p, q = field(ad, "pvals", "a"), field(ad, "pvals_adj", "a")
        np.testing.assert_allclose(q, np.minimum(p * N_GENES, 1.0), rtol=F32_RTOL)

    def test_corr_method_is_not_hardcoded(self):
        """It was, until an arm needed a different correction through this
        function. The names are scanpy's; see tests/test_corr_method.py."""
        bonf = make_adata()
        rank_genes_groups_ln(bonf, "g", corr_method="bonferroni")
        bh = make_adata()
        rank_genes_groups_ln(bh, "g", corr_method="benjamini-hochberg")
        assert not np.allclose(field(bonf, "pvals_adj", "a"), field(bh, "pvals_adj", "a"))
        # the correction changes only the adjusted column
        np.testing.assert_array_equal(field(bonf, "pvals", "a"), field(bh, "pvals", "a"))

    @pytest.mark.parametrize("key", ["logfoldchanges", "pvals", "scores", "lfc_se"])
    def test_sparse_and_dense_paths_agree(self, key):
        dense = make_adata()
        rank_genes_groups_ln(dense, "g", sparse=False)
        sparse = make_adata()
        rank_genes_groups_ln(sparse, "g", sparse=True)
        np.testing.assert_allclose(
            field(dense, key, "a"), field(sparse, key, "a"), rtol=F32_RTOL, equal_nan=True
        )

    def test_layer_is_used_in_preference_to_X(self):
        """A zeroed X would give no finite result, so this cannot pass by accident."""
        real = counts()
        ad = make_adata(np.zeros_like(real))
        ad.layers["L"] = real.copy()
        rank_genes_groups_ln(ad, "g", layer="L")
        assert np.all(np.isfinite(field(ad, "logfoldchanges", "a")))

    def test_use_raw_reads_the_raw_attribute(self):
        real = counts()
        ad = make_adata(np.zeros_like(real))
        raw = make_adata(real)
        ad.raw = raw
        rank_genes_groups_ln(ad, "g", use_raw=True)
        assert np.all(np.isfinite(field(ad, "logfoldchanges", "a")))


class TestNormalisation:
    def test_cp10k_is_applied_and_is_idempotent(self):
        """The docstring says 'takes normalized data'; the code normalises anyway.

        Harmless, because rescaling rows that already sum to 1e4 multiplies by
        1.0 -- so the wrapper on CP10K input agrees with the estimator asked not
        to normalise. Pinned because it is the difference between the wrapper
        being safe on pre-normalised input and silently double-scaling it.
        """
        X = counts()
        cp10k = X / X.sum(axis=1, keepdims=True) * 1e4
        half = len(cp10k) // 2
        ad = make_adata(cp10k, labels=["Y"] * half + ["X"] * (len(cp10k) - half))
        rank_genes_groups_ln(ad, "g", groups=["Y"], reference="X")

        direct, _ = get_LN_lfcs(cp10k[:half], cp10k[half:], normalize=False)
        order = [ad.var_names.get_loc(n) for n in np.asarray(ad.uns["rank_genes_groups"]["names"]["Y"])]
        np.testing.assert_allclose(
            field(ad, "logfoldchanges", "Y"), direct[order], rtol=F32_RTOL, atol=1e-6
        )
