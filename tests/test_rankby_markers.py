"""Tests for ``rank_genes_groups_ln(..., rankby="markers")``, the paper's marker rule.

The rule: the genes LN's t-test calls significant and up in a group come first,
ranked by the detection term of its log-fold change over its standard error,
ties to the test statistic and then to the gene name; the other genes follow as
the default ranking orders them. Each test pins one clause, against a score
recomputed here from the counts, or a promise the default ranking keeps.

Like ``tests/test_rank_genes_groups.py``, this needs anndata and pandas, which
are test dependencies only, and skips without them.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("anndata")
pytest.importorskip("statsmodels")

import anndata  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from lntest import rank_genes_groups_ln  # noqa: E402

GROUPS = ["a", "b", "c"]
N_PER, N_GENES, N_MARKERS = 60, 30, 6
# marker_scores are stored as float32, like every other field
F32_RTOL = 1e-6


def marker_counts(seed=0):
    """Over-dispersed background, with the first genes detected mostly in group a."""
    rng = np.random.default_rng(seed)
    n = N_PER * len(GROUPS)
    X = rng.poisson(rng.gamma(1.0, 2.0, size=(n, N_GENES))).astype(float)
    for j in range(N_MARKERS):
        X[:, j] = 0
        on = rng.random(N_PER) < 0.5 + 0.08 * j
        X[:N_PER][on, j] = rng.poisson(8, on.sum()) + 1
        rest = rng.random(n - N_PER) < 0.02 * (j + 1)
        X[N_PER:][rest, j] = rng.poisson(2, rest.sum()) + 1
    labels = sum(([g] * N_PER for g in GROUPS), [])
    return X, labels


def make_adata(X, labels):
    return anndata.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"g": pd.Categorical(labels)}, index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )


def expected_scores(X, own, ref):
    """The marker score from the counts, written out without the package."""
    a_g = np.maximum((X[own] > 0).sum(0), 1).astype(float)
    a_r = np.maximum((X[ref] > 0).sum(0), 1).astype(float)
    n_g, n_r = own.sum(), ref.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((a_g / n_g) / (a_r / n_r)) / np.sqrt(1 / a_g - 1 / n_g + 1 / a_r - 1 / n_r)


def run(X, labels, rankby="markers", **kwargs):
    ad = make_adata(X, labels)
    rank_genes_groups_ln(ad, "g", corr_method="bonferroni", rankby=rankby, **kwargs)
    return ad


def field(ad, name, group):
    return np.asarray(ad.uns["rank_genes_groups"][name][group])


def is_marker(ad, group):
    """Per stored row, whether the gene is significant and up."""
    return (field(ad, "pvals_adj", group).astype(float) < 0.05) & (field(ad, "scores", group).astype(float) > 0)


class TestTheRule:
    def test_markers_come_first_and_are_the_significant_up_genes(self):
        X, labels = marker_counts()
        ad = run(X, labels)
        default = run(X, labels, rankby="score")
        for g in GROUPS:
            sig = is_marker(ad, g)
            k = int(sig.sum())
            assert sig[:k].all() and not sig[k:].any()
            assert set(field(ad, "names", g)[:k]) == set(field(default, "names", g)[is_marker(default, g)])
        assert set(field(ad, "names", "a")[: int(is_marker(ad, "a").sum())]) >= {f"g{j}" for j in range(N_MARKERS)}

    def test_markers_are_ordered_by_the_marker_score(self):
        X, labels = marker_counts()
        ad = run(X, labels)
        lab = np.asarray(labels)
        for g in GROUPS:
            k = int(is_marker(ad, g).sum())
            expected = dict(zip(ad.var_names, expected_scores(X, lab == g, lab != g)))
            ordered = np.array([expected[n] for n in field(ad, "names", g)[:k]])
            np.testing.assert_allclose(field(ad, "marker_scores", g)[:k], ordered, rtol=F32_RTOL, equal_nan=True)
            finite = ordered[np.isfinite(ordered)]
            assert np.all(np.diff(finite) <= 0)

    def test_the_other_genes_follow_by_statistic(self):
        ad = run(*marker_counts())
        for g in GROUPS:
            k = int(is_marker(ad, g).sum())
            assert np.all(np.diff(field(ad, "scores", g)[k:].astype(float)) <= 0)

    def test_a_gene_the_rest_never_detects_gets_a_finite_score(self):
        X, labels = marker_counts()
        X[N_PER:, 0] = 0
        ad = run(X, labels)
        names = list(field(ad, "names", "a"))
        assert is_marker(ad, "a")[names.index("g0")]
        assert np.isfinite(field(ad, "marker_scores", "a")[names.index("g0")])

    def test_ties_go_to_the_statistic_then_to_the_name(self):
        X, labels = marker_counts()
        # g1 and g2 copy g0's detection pattern, so all three share one marker score;
        # g1 and g2 are identical, so they also share the statistic
        X[:, 1] = (X[:, 0] > 0) * 20.0
        X[:, 2] = X[:, 1]
        ad = run(X, labels)
        lab = np.asarray(labels)
        expected = dict(zip(ad.var_names, expected_scores(X, lab == "a", lab != "a")))
        k = int(is_marker(ad, "a").sum())
        names = list(field(ad, "names", "a")[:k])
        tied = [n for n in names if expected[n] == expected["g0"]]
        assert {"g0", "g1", "g2"} <= set(tied)
        score = dict(zip(field(ad, "names", "a"), field(ad, "scores", "a").astype(float)))
        assert tied == sorted(tied, key=lambda n: (-score[n], n))
        assert tied.index("g2") == tied.index("g1") + 1

    def test_genes_every_cell_detects_come_last_among_the_markers(self):
        X, labels = marker_counts()
        X[:, 7] = 5.0
        X[:N_PER, 7] = 50.0
        ad = run(X, labels)
        k = int(is_marker(ad, "a").sum())
        names = list(field(ad, "names", "a")[:k])
        m = field(ad, "marker_scores", "a")[:k].astype(float)
        assert "g7" in names and np.isnan(m[names.index("g7")])
        first_nan = int(np.argmax(np.isnan(m)))
        assert np.all(np.isnan(m[first_nan:]))

    def test_a_reference_group_replaces_the_rest(self):
        X, labels = marker_counts()
        ad = run(X, labels, reference="c")
        lab = np.asarray(labels)
        expected = dict(zip(ad.var_names, expected_scores(X, lab == "a", lab == "c")))
        k = int(is_marker(ad, "a").sum())
        np.testing.assert_allclose(
            field(ad, "marker_scores", "a")[:k],
            [expected[n] for n in field(ad, "names", "a")[:k]],
            rtol=F32_RTOL,
            equal_nan=True,
        )

    def test_sparse_and_dense_input_agree(self):
        X, labels = marker_counts()
        dense = run(X, labels)
        sparse = run(sp.csr_matrix(X), labels)
        for g in GROUPS:
            np.testing.assert_array_equal(field(dense, "names", g), field(sparse, "names", g))


class TestContract:
    def test_the_default_ranking_is_unchanged(self):
        X, labels = marker_counts()
        implicit = make_adata(X, labels)
        rank_genes_groups_ln(implicit, "g", corr_method="bonferroni")
        explicit = run(X, labels, rankby="score")
        assert "marker_scores" not in implicit.uns["rank_genes_groups"]
        assert implicit.uns["rank_genes_groups"]["params"]["rankby"] == "score"
        for name in ("names", "scores", "logfoldchanges", "pvals", "pvals_adj", "lfc_se"):
            for g in GROUPS:
                np.testing.assert_array_equal(field(implicit, name, g), field(explicit, name, g))

    def test_markers_add_marker_scores_beside_the_scanpy_keys(self):
        ad = run(*marker_counts())
        assert set(ad.uns["rank_genes_groups"]) == {
            "params", "names", "scores", "logfoldchanges", "pvals", "pvals_adj", "lfc_se", "marker_scores",
        }
        assert ad.uns["rank_genes_groups"]["params"]["rankby"] == "markers"

    def test_every_field_is_permuted_the_same_way(self):
        X, labels = marker_counts()
        ad = run(X, labels)
        default = run(X, labels, rankby="score")
        for name in ("scores", "logfoldchanges", "pvals", "pvals_adj", "lfc_se"):
            for g in GROUPS:
                by_gene = dict(zip(field(default, "names", g), field(default, name, g)))
                np.testing.assert_array_equal(
                    field(ad, name, g), [by_gene[n] for n in field(ad, "names", g)]
                )

    def test_n_genes_keeps_the_head_of_the_marker_order(self):
        X, labels = marker_counts()
        full = run(X, labels)
        top = run(X, labels, n_genes=5)
        for g in GROUPS:
            np.testing.assert_array_equal(field(top, "names", g), field(full, "names", g)[:5])

    def test_rankby_abs_orders_the_other_genes_by_magnitude(self):
        X, labels = marker_counts()
        plain = run(X, labels)
        ad = run(X, labels, rankby_abs=True)
        for g in GROUPS:
            k = int(is_marker(ad, g).sum())
            np.testing.assert_array_equal(field(ad, "names", g)[:k], field(plain, "names", g)[:k])
            assert np.all(np.diff(np.abs(field(ad, "scores", g)[k:].astype(float))) <= 0)

    def test_an_unknown_rankby_raises(self):
        X, labels = marker_counts()
        with pytest.raises(ValueError, match="rankby"):
            run(X, labels, rankby="detection")

    def test_scanpy_reads_the_markers_first(self):
        sc = pytest.importorskip("scanpy")
        ad = run(*marker_counts())
        k = int(is_marker(ad, "a").sum())
        df = sc.get.rank_genes_groups_df(ad, group="a")
        np.testing.assert_array_equal(df["names"].to_numpy()[:k], field(ad, "names", "a")[:k])
