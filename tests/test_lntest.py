"""Tests for LN's t-test.

There is no upstream implementation to diff against -- LN's t-test is a new
estimator, not a faster implementation of an existing test -- so correctness is
pinned three ways instead:

* an exact algebraic identity the estimator must satisfy, on a hand-built
  matrix with round-number answers (``TestEstimand``);
* agreement between the dense and sparse code paths (``TestArrayTypes``);
* the calibration property the method exists for, measured against the baseline
  it is meant to replace (``TestCalibration``).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

from lntest import get_LN_lfcs, get_LN_lfcs_sparse, trigamma_diff

ALPHA = 0.05

# The dense and sparse entry points are one implementation: get_LN_lfcs
# delegates to get_LN_lfcs_sparse. Two implementations of the same algebra had
# cost ~1e-6 of disagreement, from float32 intermediates on the dense side,
# which is outside the rtol=1e-5 scanpy asserts across array types. Everything
# below is therefore held to float64 tolerance; a regression that reintroduces
# a second code path will show up here first.
FLOAT64_RTOL = 1e-12


def negative_binomial(mu, phi, size, rng):
    """NB counts with mean ``mu`` and variance ``mu + phi * mu**2``."""
    return rng.poisson(rng.gamma(1.0 / phi, mu * phi, size=size))


# --------------------------------------------------------------------------
# Hand-built fixture. Column means are round numbers and every entry is exactly
# representable in float32, so the expected answer is exact on both code paths.
# Zeros are present so the detection-rate half of the estimator is exercised.
#
#           gene0  gene1  gene2            gene0  gene1  gene2
#   mean(Y)     4      1      3    mean(X)     1      2      3
#   => LFC = log2(4/1), log2(1/2), log2(3/3) = 2, -1, 0
# --------------------------------------------------------------------------
Y_SMALL = np.array([[8.0, 4.0, 6.0], [8.0, 0.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
X_SMALL = np.array([[2.0, 4.0, 12.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
LFC_EXPECTED = np.array([2.0, -1.0, 0.0])


class TestEstimand:
    """The estimator must target log2 E[Y] - log2 E[X].

    The factorisation E[Y] = P(Y>0) * E[Y|Y>0] collapses algebraically: the
    detection rate n+/n times the positive-part mean sum+/n+ is the plain
    column mean. So the reported LFC must equal the log2 ratio of column means,
    with no dependence on how the mass is distributed across cells.
    """

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_lfc_is_the_mean_scale_log_ratio(self, sparse):
        if sparse:
            lfc, _ = get_LN_lfcs_sparse(
                sp.csr_matrix(Y_SMALL), sp.csr_matrix(X_SMALL), normalize=False
            )
        else:
            lfc, _ = get_LN_lfcs(Y_SMALL, X_SMALL, normalize=False)
        np.testing.assert_allclose(lfc, LFC_EXPECTED, atol=1e-12)

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_matches_column_means_on_random_counts(self, sparse):
        rng = np.random.default_rng(11)
        Y = rng.poisson(3.0, size=(200, 50)).astype(float)
        X = rng.poisson(5.0, size=(180, 50)).astype(float)
        expected = np.log2(Y.mean(axis=0)) - np.log2(X.mean(axis=0))
        if sparse:
            lfc, _ = get_LN_lfcs_sparse(
                sp.csr_matrix(Y), sp.csr_matrix(X), normalize=False
            )
        else:
            lfc, _ = get_LN_lfcs(Y, X, normalize=False)
        np.testing.assert_allclose(lfc, expected, rtol=FLOAT64_RTOL)

    def test_identical_groups_give_zero_lfc_and_no_significance(self):
        rng = np.random.default_rng(3)
        counts = rng.poisson(4.0, size=(400, 300)).astype(float)
        lfc, pvals = get_LN_lfcs(counts, counts.copy(), normalize=False)
        np.testing.assert_allclose(lfc, 0.0, atol=1e-12)
        assert np.all(pvals > 0.99)

    def test_uniform_scaling_gives_the_scaling_factor(self):
        rng = np.random.default_rng(5)
        base = rng.poisson(10.0, size=(300, 40)).astype(float)
        lfc, _ = get_LN_lfcs_sparse(
            sp.csr_matrix(base * 4.0), sp.csr_matrix(base), normalize=False
        )
        np.testing.assert_allclose(lfc, 2.0, atol=1e-12)


class TestArrayTypes:
    """Scanpy parameterizes its DE tests over dense, CSR and CSC, so any
    divergence between code paths surfaces there first."""

    def test_csr_and_csc_are_identical(self):
        rng = np.random.default_rng(7)
        Y = rng.poisson(0.4, size=(150, 120)).astype(float)
        X = rng.poisson(0.6, size=(140, 120)).astype(float)
        Y[0, :] = np.maximum(Y[0, :], 1.0)
        X[0, :] = np.maximum(X[0, :], 1.0)

        csr = get_LN_lfcs_sparse(sp.csr_matrix(Y), sp.csr_matrix(X))
        csc = get_LN_lfcs_sparse(sp.csc_matrix(Y), sp.csc_matrix(X))
        np.testing.assert_array_equal(csr[0], csc[0])
        np.testing.assert_array_equal(csr[1], csc[1])

    @pytest.mark.parametrize("fmt", ["csr", "csc"])
    def test_sparse_matches_dense_on_the_fixture(self, fmt):
        to_sparse = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
        dense = get_LN_lfcs(Y_SMALL, X_SMALL, normalize=False)
        sparse = get_LN_lfcs_sparse(
            to_sparse(Y_SMALL), to_sparse(X_SMALL), normalize=False
        )
        np.testing.assert_allclose(sparse[0], dense[0], atol=1e-12)
        np.testing.assert_allclose(sparse[1], dense[1], rtol=1e-8)

    @pytest.mark.parametrize("fmt", ["csr", "csc"])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_sparse_matches_dense_on_sparse_counts(self, fmt, normalize):
        rng = np.random.default_rng(7)
        to_sparse = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
        Y = rng.poisson(0.4, size=(150, 120)).astype(float)
        X = rng.poisson(0.6, size=(140, 120)).astype(float)
        Y[0, :] = np.maximum(Y[0, :], 1.0)
        X[0, :] = np.maximum(X[0, :], 1.0)

        dense = get_LN_lfcs(Y, X, normalize=normalize)
        sparse = get_LN_lfcs_sparse(to_sparse(Y), to_sparse(X), normalize=normalize)
        np.testing.assert_array_equal(sparse[0], dense[0])
        np.testing.assert_array_equal(sparse[1], dense[1])

    def test_undetected_gene_does_not_produce_nan(self):
        Y = np.array([[5.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        X = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        for lfc, pvals in (
            get_LN_lfcs(Y, X, normalize=False),
            get_LN_lfcs_sparse(sp.csr_matrix(Y), sp.csr_matrix(X), normalize=False),
        ):
            assert np.all(np.isfinite(lfc))
            assert np.all(np.isfinite(pvals))


class TestTrigamma:
    """The standard error's first term, pinned to ``1/a - 1/n``.

    Swapping in the true trigamma moves every p-value in the sparse regime
    while moving no log-fold change, so it hides. These tests catch it.
    """

    @pytest.mark.parametrize("a, n", [(1, 5), (2, 2638), (50, 500), (500, 2638)])
    def test_is_the_integral_approximation(self, a, n):
        assert trigamma_diff(a, n) == 1.0 / a - 1.0 / n

    def test_is_not_the_exact_psi_1_difference(self):
        """The gap is the whole reason this was an open question."""
        a, n = 2, 2638
        exact = sum(1.0 / (j * j) for j in range(a, n))
        assert trigamma_diff(a, n) == pytest.approx(0.499621, abs=1e-6)
        assert exact == pytest.approx(0.644555, abs=1e-6)

    def test_array_and_scalar_agree(self):
        a = np.array([1, 2, 50, 500])
        np.testing.assert_array_equal(
            trigamma_diff(a, 2638), [trigamma_diff(x, 2638) for x in a]
        )

    def test_undetected_gene_is_not_clamped(self):
        """``a_hat = 0`` gives inf rather than being rounded up to 1.

        The published behaviour: callers filter those genes out. The exact form
        this replaced did clamp, so the absence of clamping is load-bearing.
        """
        with np.errstate(divide="ignore"):
            assert np.isinf(trigamma_diff(0, 100))


class TestCalibration:
    """The property the method exists for.

    Two groups drawn from negative binomials with equal means and different
    dispersions contain no differentially expressed gene by construction. A
    calibrated test rejects at the nominal rate. This is the failure mode the
    paper proves for the log1p t-test, so the baseline is measured alongside as
    a positive control: if it ever stops failing, the fixture no longer poses
    the problem and the LN assertion has stopped meaning anything.
    """

    @staticmethod
    def _equal_mean_unequal_variance(
        rng, n=500, n_genes=1000, mu=10.0, phi_y=0.1, phi_x=1.0
    ):
        Y = negative_binomial(mu, phi_y, (n, n_genes), rng).astype(float)
        X = negative_binomial(mu, phi_x, (n, n_genes), rng).astype(float)
        return Y, X

    def test_ln_is_calibrated_where_the_log1p_t_test_is_not(self):
        rng = np.random.default_rng(0)
        Y, X = self._equal_mean_unequal_variance(rng)

        _, p_ln = get_LN_lfcs(Y, X, normalize=False)
        _, p_log1p = stats.ttest_ind(np.log1p(Y), np.log1p(X), axis=0, equal_var=False)

        fpr_ln = float(np.mean(p_ln < ALPHA))
        fpr_log1p = float(np.mean(p_log1p < ALPHA))

        assert fpr_log1p > 0.5, f"baseline unexpectedly calibrated (FPR {fpr_log1p:.3f})"
        assert fpr_ln < 0.08, f"LN's t-test inflated (FPR {fpr_ln:.3f})"

    @pytest.mark.parametrize("phi_x", [0.5, 1.0, 2.0])
    def test_calibration_holds_as_the_variance_gap_widens(self, phi_x):
        rng = np.random.default_rng(42)
        Y, X = self._equal_mean_unequal_variance(rng, phi_y=0.05, phi_x=phi_x)
        _, p_ln = get_LN_lfcs(Y, X, normalize=False)
        assert float(np.mean(p_ln < ALPHA)) < 0.08

    def test_log1p_means_diverge_in_the_direction_the_theorem_predicts(self):
        # Theorem 1: for equal-mean NBs, E[log1p(Y)] > E[log1p(X)] whenever
        # Var(Y) < Var(X). Concavity of log1p is the mechanism behind every
        # result above, so pin its direction, not only its consequence.
        rng = np.random.default_rng(1)
        Y, X = self._equal_mean_unequal_variance(rng, phi_y=0.05, phi_x=2.0)
        assert Y.mean() == pytest.approx(X.mean(), rel=0.05)
        assert np.log1p(Y).mean() > np.log1p(X).mean()


class TestInputHandling:
    """Regressions worth pinning now that dense input reaches the sparse path."""

    def test_dense_input_is_not_modified_in_the_callers_frame(self):
        # _ensure_sparse_positive zeroes non-positive entries. Before it copied,
        # a float64 array passed by a caller was edited in place -- silent, and
        # invisible unless a negative value was present.
        Y = np.array([[5.0, -1.0, 3.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
        X = np.array([[2.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
        Y_before, X_before = Y.copy(), X.copy()
        get_LN_lfcs(Y, X, normalize=False)
        get_LN_lfcs_sparse(Y, X, normalize=False)
        np.testing.assert_array_equal(Y, Y_before)
        np.testing.assert_array_equal(X, X_before)

    @pytest.mark.parametrize("normalization", ["CP10K", "median-of-ratios"])
    @pytest.mark.parametrize("test_kind", ["t", "z"])
    def test_dense_and_sparse_agree_across_options(self, normalization, test_kind):
        rng = np.random.default_rng(3)
        Y = rng.poisson(0.4, size=(120, 60)).astype(float)
        X = rng.poisson(0.6, size=(110, 60)).astype(float)
        Y[0, :] = np.maximum(Y[0, :], 1.0)
        X[0, :] = np.maximum(X[0, :], 1.0)
        kw = dict(normalize=True, normalization=normalization, test=test_kind)
        dense = get_LN_lfcs(Y, X, **kw)
        sparse = get_LN_lfcs_sparse(sp.csr_matrix(Y), sp.csr_matrix(X), **kw)
        np.testing.assert_array_equal(dense[0], sparse[0])
        np.testing.assert_array_equal(dense[1], sparse[1])
