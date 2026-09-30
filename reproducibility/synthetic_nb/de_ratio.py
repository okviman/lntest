import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_style
from method_colors import method_color, method_draw_order, method_label, method_order
from paths import fig_name, require_input
from synthetic_nb.de_test import SETTINGS, out_dir

ALPHA = 0.8
METRICS = {'accuracy': 'Accuracy', 'precision': 'Precision', 'tpr': 'TPR', 'fpr': 'FPR', 'f1': 'F1'}
TITLES = {'sparse': 'Sparsely expressed genes', 'dense': 'Densely expressed genes'}
XLABEL = r'Var$(X)$ / Var$(Y)$ of the non-DEGs'


def var_ratio(setting, d1):
    """Var(X)/Var(Y) of the non-DEGs, whose mean is the same in both groups."""
    mu, d2 = SETTINGS[setting]['non_de_mu'], SETTINGS[setting]['d2']
    return (mu + d1 * mu ** 2) / (mu + d2 * mu ** 2)


def load(setting):
    """de_test's per-run metrics for one setting, with the non-DEGs' variance ratio of each run."""
    mu = SETTINGS[setting]['non_de_mu']
    df = pd.read_csv(require_input(
        out_dir(setting) / f'd1_vs_d2_01_nde_mu_{mu}.csv',
        what=f'per-run metrics of the {setting} planted-DEG simulation',
        source=f'python -m synthetic_nb.de_test --setting {setting}'))
    df['ratio'] = var_ratio(setting, df.dispersion)
    return df


def draw_metric(ax, df, col):
    """One metric's runs (dots) and their mean per ratio (line), the methods side by side at each ratio."""
    ratios = sorted(df.ratio.unique())
    dodge = 0.03 * (ratios[-1] - ratios[0])  # side by side at each ratio, so the runs don't overlap
    step = 2 if ratios[-1] > 3 else 0.5
    ticks = [1] + [t * step for t in range(1, int(ratios[-1] / step) + 1) if t * step > 1]
    methods = method_order(df.method)
    for method in method_draw_order(methods):
        k = methods.index(method)  # position by legend order, draw LN last
        s = df[df.method == method]
        c = method_color(method)
        ax.scatter(s.ratio + (k - 1) * dodge, s[col], s=4, color=c, alpha=0.25, lw=0)
        mean = s.groupby('ratio')[col].mean()
        ax.plot(mean.index + (k - 1) * dodge, mean.values, color=c, lw=1.3, marker='o', ms=3,
                alpha=ALPHA)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_ylim(-0.03, 1.03)


def method_handles(methods):
    return [Line2D([], [], color=method_color(m), lw=1.3, marker='o', ms=3, alpha=ALPHA, label=method_label(m))
            for m in method_order(methods)]


def plot(setting):
    df = load(setting)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.5, 2.1), sharey=True, layout='constrained')
    for ax, (col, name) in zip(axes, METRICS.items()):
        draw_metric(ax, df, col)
        ax.set_title(name)
    axes[0].set_ylabel('Score')
    fig.supxlabel(XLABEL)
    fig.suptitle(TITLES[setting])
    fig.legend(handles=method_handles(df.method), frameon=False, loc='outside right center', alignment='left')

    out = out_dir(setting).parent / fig_name(f'degs_metrics_{setting}')
    plot_style.rasterize_dense(fig)
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="de_test's five metrics against the non-DE variance ratio, one figure per setting."
    )
    ap.add_argument('--setting', nargs='+', default=list(SETTINGS), choices=list(SETTINGS))
    args = ap.parse_args()
    plot_style.use()
    for setting in args.setting:
        plot(setting)
