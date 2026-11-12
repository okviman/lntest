from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from scipy.stats.distributions import t


def trigamma_diff(a, n):
    """``psi_1(a) - psi_1(n)`` with ``psi_1(z) = 1/z``, as the paper defines it.

    Not an approximation to the true trigamma function, which would break the
    unbiasedness of ``theta_hat = a_hat / n``. ``a = 0`` gives ``inf``.
    """
    return 1.0 / np.asarray(a, dtype=float) - 1.0 / float(n)


def get_LN_lfcs(Y_, X_, normalize=True, test='t', normalization='CP10K',
                return_standard_error=False, return_statistic=False,
                return_log_abs_statistic=False):
    """LN's t-test on two count matrices (cells x genes), dense or sparse.

    The same test as :func:`get_LN_lfcs_sparse`, which accepts dense input too.
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


def _ensure_sparse_positive(A):
    """
    Return CSR sparse matrix containing only strictly positive entries.
    Non-positive entries are dropped, since the estimator treats them as absent.
    """
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
    """Each row divided by its DESeq2 median-of-ratios size factor, over the nonzero entries."""
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
    var[g]     = sample variance of positive entries (ddof=1), 0 where there is one
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
    # sample var (ddof=1): (sum x^2 - n_+ mean^2) / (n_+ - 1)
    several = n_plus > 1
    var[several] = (ss[several] - n_plus[several] * mean[several] ** 2) / (n_plus[several] - 1)
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
    """LN's t-test of every gene, between the cells of ``Y_`` and the cells of ``X_``.

    Parameters
    ----------
    Y_, X_
        Counts of the two groups, cells x genes, dense or scipy sparse, with the
        same genes in the same order. Zero and negative values count as not
        detected.
    normalize
        Whether to normalize each cell first, by ``normalization``.
    test
        ``"t"`` for the t-test with Welch's degrees of freedom, any other value
        for a z-test.
    normalization
        ``"CP10K"`` (counts per 10,000) or ``"median-of-ratios"`` (DESeq2's size
        factors, over the nonzero values).
    return_standard_error, return_statistic, return_log_abs_statistic
        At most one can be set. ``return_standard_error`` adds the standard
        error of the LFC, ``return_statistic`` the test statistic, and
        ``return_log_abs_statistic`` the statistic and the log of its absolute
        value, which stays finite where the statistic overflows.
    eps
        Floor on the mean of the positive values before its log is taken.

    Returns
    -------
    lfc
        The log2-fold change of ``Y_`` over ``X_``.
    p_vals
        Two-sided p-values, not corrected for multiple testing.

    Followed by the quantities the return flags select. A group in which no cell
    detects a gene counts as one cell with value 1. A gene that neither group
    detects has LFC 0, statistic 0 and p-value 1.
    """
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

    # A group with no positive entries counts as one positive entry of 1.0
    # (a_hat and pos_mean above), so its level term is 0, as for any single entry.
    se_Y_2 = np.zeros(G, dtype=np.float64)
    se_X_2 = np.zeros(G, dtype=np.float64)

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

    # A gene detected in neither group has nothing to test. Left alone, the
    # fallback above gives it an LFC of log2(n' / n) when the group sizes differ.
    absent = all_zeros_Y & all_zeros_X
    lfc[absent] = 0.0
    statistic[absent] = 0.0
    p_vals[absent] = 1.0

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
        # log|t| = log|lfc| - log(se), finite where the statistic overflows, so
        # genes keep their order when ranked for GSEA.
        se_combined_sq = se_Y**2 + se_X**2
        tiny = np.finfo(float).tiny
        log_abs = (np.log(np.abs(lfc) + tiny)
                   - 0.5 * np.log(se_combined_sq + tiny))
        return lfc, p_vals, statistic, log_abs

    return lfc, p_vals


def compute_p_vals(mean1, mean2, se1, se2):
    """The z statistic of ``mean1 - mean2`` and its two-sided p-value."""
    z_stat = (mean1 - mean2) / ((se1 ** 2 + se2 ** 2) ** 0.5)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def get_t_statistic(mean1, mean2, se1, se2, n1, n2):
    """The t statistic of ``mean1 - mean2`` and its two-sided p-value, with Welch's degrees of freedom."""
    nu1, nu2 = n1 - 1, n2 - 1
    df = (se1 ** 2 + se2 ** 2) ** 2 / (se1 ** 4 / nu1 + se2 ** 4 / nu2)
    d = mean1 - mean2
    denom = ((se1 ** 2 + se2 ** 2) ** 0.5)
    t_statistic = d / denom
    t_dist = t(df)
    p_value = 2 * t_dist.sf(np.abs(t_statistic))
    return t_statistic, p_value
