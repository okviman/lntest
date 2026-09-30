import os
import sys
import numpy as np
import pandas as pd

from paths import fig_name, require_input, results_dir
import plot_style
from method_colors import method_color, method_draw_order, method_label
import matplotlib.pyplot as plt
import argparse

plot_style.use()

SAMPLING = r'$p_\mathrm{sample}$'
SPLIT = r'$p_\mathrm{split}$'


def p_axis(p, p_symbol, odds=False):
    """x values and label for a p axis: p itself, or (1 - p) / p, which grows as the test gets harder --
    for p_split the ratio of the two groups' sampling variances, as Var(Y) / Var(X) in the synthetic NB."""
    p = np.asarray(p, dtype=float)
    if not odds:
        return p, p_symbol
    inner = p_symbol.strip('$')
    return (1 - p) / p, rf'$(1 - {inner})\,/\,{inner}$'


def plot_fpr_results(csv_file, output_file, aspect_ratio=None, p_label=SAMPLING, odds=False):
    """
    Plot FPR vs p from CSV results file: the mean over replicates, with bars of one SD.

    The axis is linear, as in the synthetic NB null figures. Bars are clipped at zero, since an FPR
    cannot be negative.

    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output plot file
    aspect_ratio : float, optional
        Aspect ratio for the plot (default: None, the panel's own)
    p_label : str
        The x-axis label: SAMPLING for UMI downsampling, SPLIT for spot splitting
    """
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's FPR-vs-subsampling results",
        source="run `python -m kidney.umi_null` first",
    ))

    df['method_display'] = df['method'].map(method_label)
    fig, ax = plt.subplots(1, 1, figsize=plot_style.figsize())
    for method in method_draw_order(df['method_display']):
        d = df[df['method_display'] == method].sort_values('p')
        mean, sd = d['fpr_mean'].to_numpy(), d['fpr_std'].to_numpy()
        ax.errorbar(p_axis(d['p'], p_label, odds)[0], mean, yerr=[np.minimum(sd, mean), sd], label=method,
                    color=method_color(method), **plot_style.ERRORBAR)

    ax.set_xlabel(p_axis([], p_label, odds)[1])
    ax.set_ylabel('FPR')
    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio, adjustable='box')

    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_fpr_log(csv_file, output_file, p_label=SAMPLING, odds=False):
    """The same mean and one-SD bars as plot_fpr_results, on a log axis with zero drawn as zero.

    Zero has no place on a log axis, so it gets its own row, labelled 0, one decade below the smallest
    positive value and set off by a break mark. A mean of zero (no false call in any replicate) sits on
    that row, and a lower bar reaching zero or below (SD at least the mean) ends there.
    """
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's FPR-vs-subsampling results",
        source="run `python -m kidney.umi_null` first",
    ))
    fig, ax = plt.subplots(1, 1, figsize=plot_style.figsize())
    draw_fpr_log(ax, df, p_label, odds)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def draw_fpr_log(ax, df, p_label=SAMPLING, odds=False):
    """plot_fpr_log's panel, drawn into ax."""
    df = df.assign(method_display=df['method'].map(method_label))
    zero = plot_style.zero_row(df['fpr_mean'], df['fpr_std'])
    for method in method_draw_order(df['method_display']):
        d = df[df['method_display'] == method].sort_values('p')
        plot_style.errorbar_log(ax, p_axis(d['p'], p_label, odds)[0], d['fpr_mean'], d['fpr_std'], zero, label=method,
                                color=method_color(method))
    plot_style.log_axis_with_zero(ax, zero, (df['fpr_mean'] + df['fpr_std']).max())
    ax.set_xlabel(p_axis([], p_label, odds)[1])
    ax.set_ylabel('FPR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot FPR results from CSV file')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Input CSV file with FPR results')
    parser.add_argument('--output', type=str, default=str(results_dir(__file__) / fig_name('fpr_plot')),
                        help='Output file for the plot (default: kidney/results/fpr_plot.pdf)')
    parser.add_argument('--aspect_ratio', type=float, default=None,
                        help="Aspect ratio for the plot (default: the panel's own)")
    parser.add_argument('--split', action='store_true',
                        help='Results are from spot_split: label the x-axis with the split probability')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    print(f"Reading results from {args.csv_file}...")
    p_label = SPLIT if args.split else SAMPLING
    plot_fpr_results(args.csv_file, args.output, args.aspect_ratio, p_label=p_label)
    plot_fpr_results(args.csv_file, f"{os.path.splitext(args.output)[0]}_odds.pdf", args.aspect_ratio,
                     p_label=p_label, odds=True)
    stem = os.path.splitext(args.output)[0]
    plot_fpr_log(args.csv_file, f"{stem}_log.pdf", p_label=p_label)
    plot_fpr_log(args.csv_file, f"{stem}_log_odds.pdf", p_label=p_label, odds=True)
    print("Done!")

