"""Two negative binomials with the same mean but different variances, before and after log1p.

The counts share their mean; their log1p means do not, which is what biases a t-test on
log1p-transformed data when the groups differ in variance.

    python -m theory.log1p_toy      # from reproducibility/
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import nbinom, poisson

import plot_style
from paths import fig_name, results_dir

COLORS = {'X': '#1f77b4', 'Y': '#ff7f0e'}  # tab10 blue and orange: distributions, not methods
TAIL = 1e-12  # expectations are summed up to this upper-tail probability
SHOWN = 0.999  # counts are drawn up to this quantile of the wider distribution


def count_distribution(mu, var):
    """Poisson when var == mu, otherwise the NB with that mean and variance."""
    if np.isclose(var, mu):
        return poisson(mu)
    if var < mu:
        raise SystemExit(f'Var = {var:g} is below the mean {mu:g}; no NB has that variance')
    r = mu ** 2 / (var - mu)
    return nbinom(r, r / (r + mu))


def log1p_mean(dist):
    k = np.arange(int(dist.ppf(1 - TAIL)) + 1)
    return float(np.sum(np.log1p(k) * dist.pmf(k)))


def plot(mu, var_x, var_y):
    dists = {'X': count_distribution(mu, var_x), 'Y': count_distribution(mu, var_y)}
    variances = {'X': var_x, 'Y': var_y}
    means = {name: log1p_mean(d) for name, d in dists.items()}
    for name in dists:
        print(f'{name}: mean {dists[name].mean():g}, Var {dists[name].var():g}, E[log1p] {means[name]:.3f}')

    k = np.arange(int(max(d.ppf(SHOWN) for d in dists.values())) + 1)
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2), sharey=True)
    for ax, x in zip(axes, (k, np.log1p(k))):
        for name, d in dists.items():
            stems = ax.stem(x, d.pmf(k), linefmt='-', markerfmt='o', basefmt=' ')
            plt.setp(stems.stemlines, color=COLORS[name], lw=0.4, rasterized=True)
            plt.setp(stems.markerline, color=COLORS[name], ms=1.5, rasterized=True)
    axes[0].axvline(mu, color='0.4', ls='--', lw=0.8)  # the shared mean
    for name in dists:
        axes[1].axvline(means[name], color=COLORS[name], ls='--', lw=0.8)

    axes[0].set_xlabel('count')
    axes[0].set_ylabel('Probability')
    axes[1].set_xlabel(r'$\log(1 + \mathrm{count})$')
    fig.tight_layout()

    def label(name):
        kind = ' (Poisson)' if np.isclose(variances[name], mu) else ''
        return f'{name}: Var = {variances[name]:g}{kind}'

    handles = [Line2D([], [], color=COLORS[n], marker='o', ms=1.5, lw=0.8) for n in dists]
    handles.append(Line2D([], [], color='0.4', ls='--', lw=0.8))
    plot_style.legend_outside(fig, handles, [label(n) for n in dists] + ['Mean'])

    out = results_dir(__file__) / fig_name('log1p_toy')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mu', type=float, default=50, help='mean of both distributions')
    ap.add_argument('--var-x', type=float, default=100, help='variance of X; equal to --mu gives a Poisson')
    ap.add_argument('--var-y', type=float, default=1000, help='variance of Y')
    args = ap.parse_args()
    plot_style.use()
    plot(args.mu, args.var_x, args.var_y)
