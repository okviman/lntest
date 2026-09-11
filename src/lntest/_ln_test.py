from __future__ import annotations
import numpy as np
import scipy.stats as stats
from scipy.stats.distributions import t
# from scipy.special import polygamma
# from typing import Optional, Sequence
# import math


def trigamma_diff(a, n):
    """``psi_1(a) - psi_1(n)`` with ``psi_1(z) = 1/z``, as the paper defines it.

    Not an approximation to "fix" to the true trigamma: that breaks the
    unbiasedness identity ``theta_hat = a_hat / n`` by up to 37%. No clamping,
    so ``a_hat = 0`` gives ``inf``.
    """
    return 1.0 / np.asarray(a, dtype=float) - 1.0 / float(n)


def digamma(x):
    return np.log(x) - 1 / (2 * x)

# def trigamma(x):
#         return 1 / x  + 0.5 / (x ** 2) +  1/(6.0*x**3)

def log_beta_param_estimates(a, b):
    """Mean and variance of log Beta(a, b).

    The variance uses :func:`trigamma_diff`, giving 0.198020 at a=5, b=500 --
    the value ``fig:beta_v_ln_vis`` was drawn with.
    """
    mu = digamma(a) - digamma(a + b)
    sigma_2 = trigamma_diff(a, a + b)
    return mu, sigma_2


def intervals_ln(log_x, n, z=1.96):
    mu_bar = np.mean(log_x)
    sigma_bar = np.var(log_x)

    se = np.sqrt(sigma_bar / n + sigma_bar ** 2 / (2 * (n - 1)))
    log_intervals = mu_bar + sigma_bar / 2 + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, mu_bar, sigma_bar


def intervals_beta(a, b, z=1.96):
    mu_log_beta, var_log_beta = log_beta_param_estimates(a, b)
    se = np.sqrt(var_log_beta)
    log_intervals = mu_log_beta + var_log_beta / 2 + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, mu_log_beta, var_log_beta


def get_intervals(log_x, a, b, z=1.96, model='lognormal', eps=0.):
    if model == 'naive':
        return interval_naive(log_x, b, z)
    n = log_x.size
    if n > 1:
        _, mu_bar, sigma_bar = intervals_ln(log_x, n, z)
        squared_standard_error_ln = sigma_bar / n + (sigma_bar ** 2) / (2 * (n - 1))
    else:
        # if there are only zero or one positive values, the mean estimate will be based on the log Beta mean
        mu_bar, sigma_bar = 0, 0
        squared_standard_error_ln = 0
    _, mu_log_beta, var_log_beta = intervals_beta(a + eps ** n, b, z)

    squared_standard_error_log_beta = var_log_beta
    se = np.sqrt(squared_standard_error_ln + squared_standard_error_log_beta)

    # estimate of the log of the mean of the ZILN
    a_hat = a + eps ** n
    log_mean_estimate = mu_bar + sigma_bar / 2 + np.log(a_hat / (a + b))
    # log_mean_estimate = mu_bar + sigma_bar / 2 + mu_log_beta + var_log_beta / 2

    log_intervals = log_mean_estimate + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, log_mean_estimate, se

def interval_naive(log_x, N_0, z=1.96):
    zeros = np.zeros(N_0)
    data = np.concatenate([np.exp(log_x), zeros])
    n = data.size
    # sample mean
    mu_bar = np.mean(data)
    # use log as exp is used outside function
    log_mu_bar = np.log(mu_bar)
    sigma_bar = np.var(data)
    se = np.sqrt(sigma_bar / n)
    intervals = mu_bar + z * np.array([-se, se])
    return intervals, log_mu_bar, se


def get_intervals_synthetic_data(true_mu, true_sigma_2, true_theta, experiments=1000,
                                 n=500, z=1.96, model='lognormal', seed=0):
    # note: true_sigma_2 neq true variance, it's the var of the data-generating normal distribution
    np.random.seed(seed)
    intervals = np.zeros((2, experiments))
    estimated_means = np.zeros(experiments)
    for i in range(experiments):
        y = np.random.binomial(1, true_theta, n)
        N_plus = y.sum()
        N_0 = n - N_plus

        log_x = np.random.normal(true_mu, np.sqrt(true_sigma_2), y.sum())
        antilog_interval, log_mean_estimate, se = get_intervals(log_x, N_plus, N_0, z, model)
        intervals[:, i] = antilog_interval
        estimated_means[i] = np.exp(log_mean_estimate)
    return intervals, estimated_means


def get_LN_lfcs(Y_, X_, normalize=True, test='t', normalization='CP10K',
                return_standard_error=False, return_statistic=False,
                return_log_abs_statistic=False):
    """LN's t-test on two count matrices (cells x genes), dense or sparse.

    Delegates to :func:`get_LN_lfcs_sparse`, which accepts dense input and
    converts it, so the dense and sparse paths are one implementation and cannot
    drift apart. Keeping them separate had cost ~1e-6 of disagreement from
    float32 intermediates on the dense side -- enough to exceed the 1e-5
    tolerance scanpy asserts across array types.
    """
    return get_LN_lfcs_sparse(
        Y_,
        X_,
        normalize=normalize,
        test=test,
        normalization=normalization,
        return_standard_error=return_standard_error,
        return_statistic=return_statistic,
        return_log_abs_statistic=return_log_abs_statistic,
    )


import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def _ensure_sparse_positive(A):
    """
    Return CSR sparse matrix containing only strictly positive entries.
    Non-positive entries are dropped rather than masked to NaN; the estimator
    treats them as absent either way.
    """
    if sp is None:
        raise ImportError("scipy.sparse is required for the sparse implementation.")

    if not sp.issparse(A):
        # np.array, not np.asarray: the next line writes into A, and a float64
        # input would otherwise be modified in the caller's frame.
        A = np.array(A, dtype=np.float64)
        A[A <= 0] = 0.0
        A = sp.csr_matrix(A, dtype=np.float64)
        A.eliminate_zeros()
        return A

    A = A.tocsr().astype(np.float64, copy=True)
    if A.nnz:
        bad = A.data <= 0
        if np.any(bad):
            A.data[bad] = 0.0
            A.eliminate_zeros()
    return A


def _cp10k_sparse(A_csr):
    """Row-normalize sparse matrix to counts-per-10k. Preserves sparsity."""
    rs = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)
    scale = np.zeros_like(rs)
    nz = rs > 0
    scale[nz] = 1e4 / rs[nz]
    return sp.diags(scale).dot(A_csr)


def _geom_mean_nonzero_per_gene(A_csc):
    """Geometric mean per gene over nonzero entries only (vectorized via CSC indptr)."""
    G = A_csc.shape[1]
    indptr = A_csc.indptr
    counts = np.diff(indptr).astype(np.float64)

    denom = np.ones(G, dtype=np.float64)
    if A_csc.nnz == 0:
        return denom

    data = A_csc.data
    logdata = np.log(data)

    # sum of logs per column using reduceat
    sumlog = np.add.reduceat(logdata, indptr[:-1])
    mask = counts > 0
    denom[mask] = np.exp(sumlog[mask] / counts[mask])
    denom[~mask] = 1.0
    denom[~np.isfinite(denom)] = 1.0
    denom[denom == 0] = 1.0
    return denom


def _median_of_ratios_sparse(A_csr):
    """
    DESeq2 median-of-ratios, ignoring zeros.
    This is the only part that uses a per-row loop (still O(nnz)).
    """
    A_csc = A_csr.tocsc()
    denom = _geom_mean_nonzero_per_gene(A_csc)

    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    n = A_csr.shape[0]
    c = np.ones(n, dtype=np.float64)

    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            c[i] = 1.0
            continue
        ratios = data[s:e] / denom[indices[s:e]]
        # guard against weird denom / inf
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        c[i] = np.median(ratios) if ratios.size else 1.0

    # divide each row by its size factor
    inv = np.ones_like(c)
    ok = c > 0
    inv[ok] = 1.0 / c[ok]
    return sp.diags(inv).dot(A_csr)


def _pos_mean_var_nnz_per_gene(A):
    """
    For strictly-positive sparse A (cells x genes), compute:
    n_plus[g]  = #positive entries (nnz per column)
    mean[g]    = mean of positive entries
    var[g]     = population variance of positive entries (ddof=0) to match np.nanvar default
    """
    A_csc = A.tocsc()
    n_plus = np.diff(A_csc.indptr).astype(np.float64)  # faster than getnnz(axis=0)
    s = np.asarray(A_csc.sum(axis=0)).ravel().astype(np.float64)
    ss = np.asarray(A_csc.multiply(A_csc).sum(axis=0)).ravel().astype(np.float64)

    # Genes with no positive entries keep a mean of 1.0, so log2 of it is 0 and
    # the all-zero case contributes nothing rather than a NaN.
    mean = np.ones(A_csc.shape[1], dtype=np.float64)
    var = np.zeros(A_csc.shape[1], dtype=np.float64)

    mask = n_plus > 0
    mean[mask] = s[mask] / n_plus[mask]
    # population var (ddof=0): E[x^2] - (E[x])^2
    ex2 = np.zeros_like(var)
    ex2[mask] = ss[mask] / n_plus[mask]
    var[mask] = ex2[mask] - mean[mask] ** 2
    var[var < 0] = 0.0  # numerical guard

    all_zeros = ~mask
    return n_plus, mean, var, all_zeros


def get_LN_lfcs_sparse(
    Y_,
    X_,
    normalize=True,
    test="t",
    normalization="CP10K",
    return_standard_error=False,
    return_statistic=False,
    return_log_abs_statistic=False,
    eps=1e-12,
):
    """
    Sparse implementation of the LN log-fold-change estimator and test.

    Expects Y_, X_ to be scipy sparse (CSR/CSC) matrices (cells x genes).
    Zeros and non-positives are treated as *absent*, and moments are computed
    over strictly-positive entries only.
    """
    if sp is None:
        raise ImportError("scipy.sparse is required for sparse Y_/X_.")

    Y = _ensure_sparse_positive(Y_)
    X = _ensure_sparse_positive(X_)

    n = Y.shape[0]
    n_prime = X.shape[0]
    G = Y.shape[1]
    if X.shape[1] != G:
        raise ValueError("Y_ and X_ must have the same number of genes (columns).")

    # Normalization (keeps sparsity)
    if normalize and (normalization == "CP10K"):
        Y = _cp10k_sparse(Y)
        X = _cp10k_sparse(X)
    elif normalize and (normalization == "median-of-ratios"):
        # Works, but slower than CP10K due to per-row medians.
        Y = _median_of_ratios_sparse(Y)
        X = _median_of_ratios_sparse(X)

    # Positive-entry moments per gene
    n_plus, pos_mean_Y, var_Y, all_zeros_Y = _pos_mean_var_nnz_per_gene(Y)
    n_plus_prime, pos_mean_X, var_X, all_zeros_X = _pos_mean_var_nnz_per_gene(X)

    # \hat{a}, falling back to ones for genes with no positive entries
    a_hat_Y = np.ones(G, dtype=np.float64)
    a_hat_Y[~all_zeros_Y] = n_plus[~all_zeros_Y]
    a_hat_X = np.ones(G, dtype=np.float64)
    a_hat_X[~all_zeros_X] = n_plus_prime[~all_zeros_X]


    # log2 theta hats
    log2_theta_hat_Y = np.log2(a_hat_Y / float(n))
    log2_theta_hat_X = np.log2(a_hat_X / float(n_prime))

    # sample mean of positive counts
    # (pos_mean_* is already 1.0 for all-zero genes, so log2 contributes 0)
    log2_m_Y = np.log2(np.maximum(pos_mean_Y, eps))
    log2_m_X = np.log2(np.maximum(pos_mean_X, eps))

    se_Y_1 = trigamma_diff(a_hat_Y, int(n))
    se_X_1 = trigamma_diff(a_hat_X, int(n_prime))

    mu_Y = log2_theta_hat_Y + log2_m_Y
    mu_X = log2_theta_hat_X + log2_m_X
    lfc = mu_Y - mu_X

    se_Y_2 = np.ones(G, dtype=np.float64)
    se_X_2 = np.ones(G, dtype=np.float64)

    mY2 = np.maximum(pos_mean_Y, eps) ** 2
    mX2 = np.maximum(pos_mean_X, eps) ** 2

    maskY = ~all_zeros_Y
    maskX = ~all_zeros_X

    se_Y_2[maskY] = np.log(1.0 + var_Y[maskY] / (np.maximum(n_plus[maskY], 1.0) * mY2[maskY]))
    se_X_2[maskX] = np.log(1.0 + var_X[maskX] / (np.maximum(n_plus_prime[maskX], 1.0) * mX2[maskX]))

    se_Y = np.sqrt(np.maximum(se_Y_1 + se_Y_2, 0.0)) / np.log(2.0)
    se_X = np.sqrt(np.maximum(se_X_1 + se_X_2, 0.0)) / np.log(2.0)

    if test == "t":
        statistic, p_vals = get_t_statistic(mu_Y, mu_X, se_Y, se_X, n, n_prime)
    else:
        statistic, p_vals = compute_p_vals(mu_Y, mu_X, se_Y, se_X)

    # The three return flags select what comes back after (lfc, p_vals) and are
    # mutually exclusive. Setting more than one raises rather than silently
    # picking whichever the branch order happens to reach first: a caller that
    # asks for two quantities and is handed one has no way to notice.
    _asked = [n for n, v in (("return_standard_error", return_standard_error),
                             ("return_statistic", return_statistic),
                             ("return_log_abs_statistic", return_log_abs_statistic)) if v]
    if len(_asked) > 1:
        raise ValueError(
            "these return flags are mutually exclusive, and "
            f"{', '.join(_asked)} were all set. Call once per quantity, or use "
            "return_log_abs_statistic, which returns the statistic alongside "
            "its log-magnitude."
        )

    if return_standard_error:
        return lfc, p_vals, np.sqrt(se_X**2 + se_Y**2)

    if return_statistic:
        return lfc, p_vals, statistic

    if return_log_abs_statistic:
        # log|t| in log space: log|d| - 0.5*log(se^2). Identical to
        # log(abs(statistic)) in exact arithmetic, but finite where the
        # statistic itself overflows -- which is what GSEA ranking needs, since
        # genes tied at inf carry no ordering.
        se_combined_sq = se_Y**2 + se_X**2
        tiny = np.finfo(float).tiny
        log_abs = (np.log(np.abs(mu_Y - mu_X) + tiny)
                   - 0.5 * np.log(se_combined_sq + tiny))
        return lfc, p_vals, statistic, log_abs

    return lfc, p_vals



def transform(z):
    # log(10000 * z / z.sum(over genes for each cell) + 1)
    return np.log((z * 1e4 / z.sum(1, keepdims=True)) + 1)


def compute_p_vals(mean1, mean2, se1, se2):
    # Compute the test statistic
    z_stat = (mean1 - mean2) / ((se1 ** 2 + se2 ** 2) ** 0.5)

    # Compute the p-value for the two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value

def get_t_statistic(mean1, mean2, se1, se2, n1, n2):
    # implements two-sided t-test
    nu1, nu2 = n1 - 1, n2 - 1
    df = (se1 ** 2 + se2 ** 2) ** 2 / (se1 ** 4 / nu1 + se2 ** 4 / nu2)
    d = mean1 - mean2
    denom = ((se1 ** 2 + se2 ** 2) ** 0.5)
    t_statistic = d / denom
    t_dist = t(df)
    p_value = 2 * t_dist.sf(np.abs(t_statistic))
    return t_statistic, p_value

