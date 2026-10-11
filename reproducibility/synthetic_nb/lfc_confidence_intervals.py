"""LFC estimates and their 95% intervals against the true LFC: the log1p t-test's and LN's.

Two groups of 1,000 cells and 1,500 genes. X is NB with dispersion 2 and mean 10, a tenth of its genes shifted
up by |N(0, 5)|; Y is NB with dispersion 1 and mean 10e, so e times deeper. After normalising for depth, the
true LFC of Y against X is log2(10 / mu_X): 0 for the unshifted genes.

The log1p interval is the t-test's, around the difference of the groups' mean log1p expression; the figure's
log1p points are scanpy's reported logfoldchanges, which can fall outside that interval.

    python -m synthetic_nb.lfc_confidence_intervals      # from reproducibility/
"""

import numpy as np
from lntest import get_LN_lfcs as get_DELN_lfcs
from baselines import scanpy_sig_test
from paths import fig_name, results_dir
import plot_style
from method_colors import method_color, method_label
import matplotlib.pyplot as plt

SEED = 10


def _save(name):
    out = results_dir(__file__) / fig_name(name)
    plot_style.rasterize_dense(plt.gcf())
    plt.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close()
    print(f'wrote {out}')


"""
Scanpy function to get mean and vars used for computing the t statistic

def _get_mean_var(
    X: _SupportedArray, *, axis: Literal[0, 1] = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(X, sparse.spmatrix):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = axis_mean(X, axis=axis, dtype=np.float64)
        mean_sq = axis_mean(elem_mul(X, X), axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var
"""


def simulate(seed=SEED):
    """The true LFCs, the non-DEG mask, scanpy's reported LFCs, and each method's estimate with its standard error.

    The log1p estimate is the difference of mean log1p expression in log2 units, the quantity the t-test tests.
    """
    np.random.seed(seed)
    nx = 1000
    ny = 1000
    n_genes = 1500
    mu1 = 10 + np.abs(np.random.normal(0, 5, (1, n_genes))) * np.random.binomial(1, 0.1, (1, n_genes))
    mu2 = 10 * np.exp(1.)
    d1 = 2.
    d2 = 1.

    r1 = 1 / d1
    r2 = 1 / d2
    p1 = 1 / (1 + d1 * mu1)
    p2 = 1 / (1 + d2 * mu2)

    # Generate synthetic gene expression data
    X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
    Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

    true_lfcs = np.log2(mu2 * np.exp(-1.) / mu1)
    non_de_idx = (true_lfcs == 0.).reshape(-1)

    sc_lfcs = scanpy_sig_test(X, Y)[0].to_numpy()  # the Series index is in ranked order
    X_raw, Y_raw = X.copy(), Y.copy()
    # this is how Scanpy calculates the statistics used in the t test (based on their code)
    # the results here were reproduced in the debugging console within the scanpy code, too
    X = np.log(1e4 * X / (X.sum(1, keepdims=True)) + 1)
    Y = np.log(1e4 * Y / (Y.sum(1, keepdims=True)) + 1)

    mean_Y, mean_X = np.mean(Y, axis=0), np.mean(X, axis=0)

    mean_sq_Y, mean_sq_X = np.mean(Y * Y, axis=0), np.mean(X * X, axis=0)
    var_Y = (mean_sq_Y - mean_Y ** 2) * ny / (ny - 1)
    var_X = (mean_sq_X - mean_X ** 2) * nx / (nx - 1)

    se_Y = np.sqrt(var_Y / ny)
    se_X = np.sqrt(var_X / nx)
    se = np.sqrt(se_Y ** 2 + se_X ** 2) / np.log(2)

    deln_lfcs, _, gamma = get_DELN_lfcs(Y_raw, X_raw, return_standard_error=True)
    return {'true': true_lfcs[0], 'non_de': non_de_idx, 'scanpy': sc_lfcs,
            'log1p': ((mean_Y - mean_X) / np.log(2), se), 'LN': (deln_lfcs, gamma)}


def coverage(true, estimate, se):
    """The share of genes whose 95% interval holds the true LFC."""
    return np.mean(np.abs(estimate - true) <= 1.96 * se)


def limits(sim):
    """Both methods' plots' x and y ranges: every true LFC, point and interval, padded; x ends just past the largest true LFC."""
    ends = [sim['true'], sim['scanpy']]
    for estimate, se in (sim['LN'], sim['log1p']):
        ends += [estimate - 1.96 * se, estimate + 1.96 * se]
    lo, hi = min(e.min() for e in ends), max(e.max() for e in ends)
    pad = 0.05 * (hi - lo)
    return (lo - pad, sim['true'].max() + pad), (lo - pad, hi + pad)


def draw(ax, sim, key):
    """One method's points and intervals against the true LFC, the dashed diagonal where they would equal it.

    Both methods' plots share the same ranges.

    The log1p points are scanpy's reported logfoldchanges, drawn over the t-test's interval; LN's are its estimates.
    """
    true = sim['true']
    estimate, se = sim[key]
    method = 'LN' if key == 'LN' else 't-test'
    points = estimate if key == 'LN' else sim['scanpy']
    ci = ax.errorbar(true, estimate, yerr=1.96 * se, fmt='none', color=method_color(method), alpha=0.5, lw=0.8)
    dots, = ax.plot(true, points, 'o', ms=2, color=method_color(method))
    ax.axline((0, 0), slope=1, color='0.3', lw=0.8, ls='--')
    xlim, ylim = limits(sim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('True LFC')
    ax.set_ylabel('Estimated LFC')
    ax.set_title(method_label(method))
    ax.legend([(ci, dots)], ['Estimated LFCs and 95% CI'])


def main():
    sim = simulate()
    non_de_idx, sc_lfcs = sim['non_de'], sim['scanpy']
    log1p_lfcs, se = sim['log1p']
    deln_lfcs, gamma = sim['LN']

    # for simplicity, the confidence intervals are formed with a critical z value (not t)
    # but for these sample sizes the values are equal
    confidence_intervals = log1p_lfcs + 1.96 * np.array([-se, se])
    print(f"Estimated LFC is below CI {np.sum(sc_lfcs[~non_de_idx] < confidence_intervals[0, ~non_de_idx])} times for DEGs")
    print(f"Estimated LFC is above CI {np.sum(sc_lfcs[~non_de_idx] > confidence_intervals[1, ~non_de_idx])} times for DEGs")
    print()
    print(f"Estimated LFC is below CI {np.sum(sc_lfcs[non_de_idx] < confidence_intervals[0, non_de_idx])} times for non-DEGs")
    print(f"Estimated LFC is above CI {np.sum(sc_lfcs[non_de_idx] > confidence_intervals[1, non_de_idx])} times for non-DEGs")
    print(f"Frequency of estimated LFC that are above CI {np.sum(sc_lfcs[non_de_idx] > confidence_intervals[1, non_de_idx]) / np.sum(non_de_idx)} for  non-DEGs")
    print()
    plot_style.use()
    plt.figure(figsize=plot_style.figsize())
    draw(plt.gca(), sim, 'log1p')
    plt.tight_layout()
    _save('lfc_ci_log1p.png')

    deln_confidence_intervals = deln_lfcs + 1.96 * np.array([-gamma, gamma])
    print(f"LN's Estimated LFC is below CI {np.sum(deln_lfcs < deln_confidence_intervals[0])} times")
    print(f"LN's Estimated LFC is above CI {np.sum(deln_lfcs > deln_confidence_intervals[1])} times")

    plt.figure(figsize=plot_style.figsize())
    draw(plt.gca(), sim, 'LN')
    plt.tight_layout()
    _save('lfc_ci_ln.png')

    print()
    for name, (estimate, s) in (('log1p', sim['log1p']), ('LN', sim['LN'])):
        print(f"{name}: the 95% interval holds the true LFC for {coverage(sim['true'], estimate, s):.1%} of genes")


if __name__ == '__main__':
    main()
