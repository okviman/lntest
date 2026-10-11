import argparse

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from lntest import get_LN_lfcs
import matplotlib.pyplot as plt
from baselines import scanpy_sig_test, get_test_results
import plot_style
from method_colors import method_label, method_order
from paths import fig_name, require_input, results_dir

NX = 1000
NY = 1000
N_GENES = 1500
FANOS = [1, 2, 4, 8]  # ratio grid: Var(X) as multiples of the mean; 1 is Poisson
# Var(Y)/Var(X), a quarter-octave apart. Ratios below 1 would repeat these settings
# with the groups swapped: every test is two-sided and the groups are the same size.
RATIOS = 2.0 ** np.linspace(0, 4, 17)
# Past 16, so every Wilcoxon curve saturates too; drawn after the whole RATIOS grid, so its values are unchanged.
EXTRA = 2.0 ** np.linspace(4.25, 6, 8)
# The tests each grid runs: the ratio grid is Fig 1c's; the dispersion grid reproduces RECOMB Fig 1, which had two.
TESTS = {'ratio': ('log1p t-test', "LN's t-test", 'Wilcoxon'), 'dispersion': ('log1p t-test', "LN's t-test")}


def draw_nb(mu, d, size):
    """Counts with mean mu and variance mu + d * mu**2; d = 0 is Poisson."""
    if d == 0:
        return np.random.poisson(mu, size=size)
    return np.random.negative_binomial(n=1 / d, p=1 / (1 + d * mu), size=size)


def d1_grid(base_mu, grid):
    """Dispersions of X: the RECOMB grid's four, or Var(X) = FANOS * mean."""
    if grid == 'dispersion':
        return np.linspace(0.01, 1, 4)
    return (np.array(FANOS) - 1) / base_mu


def d2_grid(d1, base_mu, grid, ratios=RATIOS):
    """Dispersions of Y to simulate against X's dispersion d1.

    'dispersion' is the grid behind the RECOMB figure: the same Var(Y) values
    for every Var(X). 'ratio' puts Var(Y)/Var(X) on RATIOS instead.
    """
    if grid == 'dispersion':
        return np.linspace(0.0001, 2, 20)
    var_x = d1 * base_mu ** 2 + base_mu
    return (ratios * var_x - base_mu) / base_mu ** 2


def adjusted_pvalues(X, Y, test):
    """One test's Bonferroni-adjusted p-values: LN's on the counts, scanpy's t-test and tie-corrected Wilcoxon on
    CP10K log1p."""
    if test == "LN's t-test":
        return smm.multipletests(get_LN_lfcs(Y, X, test='t')[1], alpha=0.05, method='bonferroni')[1]
    return scanpy_sig_test(X, Y, method='wilcoxon' if test == 'Wilcoxon' else 't-test')[1]


def score(adj_pvals, true_lfcs, metric):
    """The metric at adjusted p < 0.05; a run that calls nothing has accuracy 1 and FPR 0, which
    get_test_results cannot build a confusion matrix for."""
    if np.sum(adj_pvals >= 0.05) == len(adj_pvals):
        return {"accuracy": 1., "fpr": 0.}[metric]
    return get_test_results(adj_pvals, true_lfcs, verbose=False)[metric]


def draws(base_mu, seed=0, grid='ratio'):
    """Every condition's X and Y in the sweep's order, one random stream: (Var(X), Var(Y), X, Y) per condition, a
    Var(X)'s conditions consecutive."""
    np.random.seed(seed)
    mu1 = base_mu + np.zeros((1, N_GENES))
    d1_list = d1_grid(base_mu, grid)
    var_1 = d1_list * base_mu ** 2 + base_mu
    for ratios in ((RATIOS, EXTRA) if grid == 'ratio' else (RATIOS,)):
        for i, d1 in enumerate(d1_list):
            d2_list = d2_grid(d1, base_mu, grid, ratios)
            var_2 = d2_list * base_mu ** 2 + base_mu
            for k, d2 in enumerate(d2_list):
                X = draw_nb(mu1, d1, (NX, N_GENES))
                Y = draw_nb(base_mu, d2, (NY, N_GENES))
                yield var_1[i], var_2[k], X, Y


def simulate(base_mu, metric, seed=0, grid='ratio'):
    """One row per (Var(X), Var(Y), test): the metric over N_GENES null genes; within a Var(X)'s run, each test's
    values over Var(Y) together, as the sweep has always written them."""
    true_lfcs = np.zeros((1, N_GENES))
    rows, run, current = [], [], None
    for var_x, var_y, X, Y in draws(base_mu, seed, grid):
        if var_x != current and run:
            rows += sorted(run, key=lambda r: TESTS[grid].index(r['method']))
            run = []
        current = var_x
        run += [{'base_mu': base_mu, 'metric': metric, 'var_x': var_x, 'var_y': var_y, 'method': test,
                 'value': score(adjusted_pvalues(X, Y, test), true_lfcs, metric)} for test in TESTS[grid]]
    rows += sorted(run, key=lambda r: TESTS[grid].index(r['method']))
    return pd.DataFrame(rows)


def plot(df, metric, out):
    """One panel per test, one line per Var(X)."""
    plot_style.use()
    methods = method_order(df.method)
    fig, axes = plt.subplots(1, len(methods), figsize=plot_style.figsize(len(methods)), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, method in zip(axes, methods):
        for vx in sorted(df.var_x.unique()):
            s = df[(df.method == method) & (df.var_x == vx)]
            ax.plot(s.var_y.astype(int), s.value, lw=1.3, label=f"Var$(X$)={int(vx)}")
        ax.set_title(method_label(method))
        ax.set_xlabel(r'Var$(Y)$')
    axes[0].set_ylabel(metric.upper())
    if metric == 'fpr':
        axes[0].set_yticks(np.arange(0, 11) * 1 / 10)
    fig.tight_layout()
    plot_style.legend_outside(fig)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run(base_mu, metric, seed=0, grid='ratio', plot_only=False):
    out = results_dir(__file__)
    stem = f"null_{metric}_by_{'ratio' if grid == 'ratio' else 'var'}_mu{int(base_mu)}"
    csv = out / f'{stem}.csv'
    if plot_only:
        df = pd.read_csv(require_input(csv, what=f"the null sweep at base_mu={base_mu:g}",
                                       source=f'python -m synthetic_nb.null --base-mu {base_mu:g}'))
    else:
        df = simulate(base_mu, metric, seed, grid)
        df.to_csv(csv, index=False)
    plot(df, metric, out / fig_name(stem))
    print(f'wrote {out / fig_name(stem)}' + ('' if plot_only else f' and {csv}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Null-scenario FPR/accuracy against Var(Y): LN's t-test, the log1p t-test and, on the ratio grid, Wilcoxon."
    )
    ap.add_argument('--base-mu', type=float, default=50,
                    help="shared NB mean (was hardcoded). RECOMB's var_v_fpr.png "
                         "came from 5; var_v_fpr_large_mu.png from this default")
    ap.add_argument('--metric', default='fpr', choices=['fpr', 'accuracy'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grid', default='ratio', choices=['ratio', 'dispersion'],
                    help="how Var(Y) is sampled; 'dispersion' reproduces the RECOMB figure "
                         "and the reference checksums")
    ap.add_argument('--plot-only', action='store_true',
                    help="redraw the figure from this sweep's CSV without re-simulating")
    args = ap.parse_args()
    run(args.base_mu, args.metric, args.seed, args.grid, args.plot_only)
