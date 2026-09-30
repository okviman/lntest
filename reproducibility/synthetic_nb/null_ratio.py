import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_style
from method_colors import method_color, method_draw_order, method_label, method_order
from paths import fig_name, require_input, results_dir

# Var(X) from lowest (Poisson) to highest: solid, long dashes, short dashes, dots
VAR_STYLES = ['-', (0, (6, 2)), (0, (3, 2)), (0, (1, 1.5))]
ALPHA = 0.8
STEMS = {'row': 'null_fpr', 'narrow': 'null_fpr_narrow', 'stacked': 'null_fpr_stacked'}
NARROW = 0.7  # the narrow layout's width, as a share of the row layout's


def load(base_mus):
    out = results_dir(__file__)
    frames = []
    for mu in base_mus:
        stem = f'null_fpr_by_ratio_mu{int(mu)}'
        path = require_input(out / f'{stem}.csv', what=f"ratio-grid null sweep at base_mu={mu:g}",
                             source=f'python -m synthetic_nb.null --base-mu {mu:g}')
        frames.append(pd.read_csv(path))
    df = pd.concat(frames)
    df['ratio'] = df.var_y / df.var_x
    return df


def ratio_axis(ax, sub):
    """Linear x-axis from 1, labelled at every integer up to 8 and beyond it at 1 and every multiple of 4, ending at
    the first integer by which every log1p curve has reached FPR 0.99 (or the last ratio, if one never does)."""
    hi = sub.ratio.max()
    lp = sub[(sub.method == 'log1p t-test') & (sub.ratio > 1)]
    reached = lp[lp.value >= 0.99].groupby('var_x').ratio.min()
    if len(reached) == lp.var_x.nunique():
        hi = min(math.ceil(reached.max()), hi)
    ax.set_xticks(range(1, int(hi) + 1) if hi <= 8 else [1, *range(4, int(hi) + 1, 4)])
    pad = 0.02 * (hi - 1)
    ax.set_xlim(1 - pad, hi + pad)


def var_multiples(sub, mu):
    """Var(X) as multiples of mu, so panels at different mu share one line-style key."""
    return [round(vx / mu, 6) for vx in sorted(sub.var_x.unique())]


def var_handles(multiples):
    """The Var(X) key's entries: a black line per multiple of mu, in its line style."""
    return [Line2D([], [], color='black', lw=1.3, ls=ls, label=r'$\mu$' if k == 1 else rf'{k:g}$\mu$')
            for k, ls in zip(multiples, VAR_STYLES)]


def panel(ax, sub, mu):
    # method by colour, Var(X) by line style
    for method in method_draw_order(sub.method):
        for vx, ls in zip(sorted(sub.var_x.unique()), VAR_STYLES):
            s = sub[(sub.var_x == vx) & (sub.method == method)].sort_values('ratio')
            ax.plot(s.ratio, s.value, color=method_color(method), lw=1.3, ls=ls, alpha=ALPHA)
    ax.set_title(rf'$\mu$ = {mu:g}')
    ratio_axis(ax, sub)
    ax.set_xlabel(r'Var$(Y)$ / Var$(X)$')


def narrow(fig, extra_artists, share):
    """Narrow the panels until the tight bounding box, legends included, is this share of its width.

    Text and legends keep their size, so the panels take all of the cut.
    """
    box = fig.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=extra_artists)
    pars = fig.subplotpars
    w, h = fig.get_size_inches()
    fig.set_size_inches(w - (1 - share) * box.width / (pars.right - pars.left), h)


def plot(df, base_mus, layout='row'):
    """One panel per base_mu: side by side ('row'), side by side in a narrower figure ('narrow'), or stacked."""
    multiples = {mu: var_multiples(df[df.base_mu == mu], mu) for mu in base_mus}
    if len({tuple(m) for m in multiples.values()}) > 1:
        raise SystemExit(f'Var(X) is not the same multiples of mu in every sweep: {multiples}')

    n = len(base_mus)
    stacked = layout == 'stacked'
    shape = (n, 1) if stacked else (1, n)
    fig, axes = plt.subplots(*shape, figsize=plot_style.figsize(*shape[::-1]), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, mu in zip(axes, base_mus):
        panel(ax, df[df.base_mu == mu], mu)
    for ax in axes if stacked else axes[:1]:
        ax.set_ylabel('FPR')
    axes[0].set_ylim(-0.03, 1.03)
    if stacked:
        fig.tight_layout()

    # stacked, each key sits beside a panel of its own; side by side, both beside the last
    method_ax, var_ax, var_y = (axes[0], axes[-1], 1.0) if stacked else (axes[-1], axes[-1], 0.62)
    style = {'frameon': False, 'loc': 'upper left', 'alignment': 'left'}
    methods = method_ax.legend(handles=[Line2D([], [], color=method_color(m), lw=1.4, alpha=ALPHA, label=method_label(m))
                                        for m in method_order(df.method)],
                               title='Method', bbox_to_anchor=(1.01, 1.0), **style)
    method_ax.add_artist(methods)
    var_key = var_ax.legend(handles=var_handles(multiples[base_mus[0]]),
                            title=r'Var$(X)$', bbox_to_anchor=(1.01, var_y), handlelength=3, **style)
    if layout == 'narrow':
        narrow(fig, [methods, var_key], NARROW)

    out = results_dir(__file__) / fig_name(STEMS[layout])
    fig.savefig(out, dpi=200, bbox_inches='tight', bbox_extra_artists=[methods, var_key])
    plt.close(fig)
    print(f'wrote {out}')


def run(base_mus):
    df = load(base_mus)
    n = df.groupby('base_mu').var_x.nunique().max()
    if n > len(VAR_STYLES):
        raise SystemExit(f'{n} values of Var(X), but only {len(VAR_STYLES)} line styles')

    plot_style.use()
    for layout in STEMS:
        plot(df, base_mus, layout)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="null.py's FPR sweeps against Var(Y)/Var(X), one panel per base_mu: in a row, in a narrower row and stacked."
    )
    ap.add_argument('--base-mu', type=float, nargs='+', default=[5, 50],
                    help='sweeps to plot, each from null.py --base-mu')
    run(ap.parse_args().base_mu)
