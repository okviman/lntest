"""TPR and FPR against p_sample from the results CSV of kidney.umi_de.

    python -m kidney.umi_de_plots --input kidney/results/downsample_degs_10000rep_results.csv --output kidney/results/downsample_degs_10000rep_tpr_fpr.pdf
"""

import argparse
import pandas as pd

from paths import fig_name, require_input
import plot_style
from method_colors import method_color, method_draw_order, method_label
import matplotlib.pyplot as plt

from kidney.fpr_plots import SAMPLING

plot_style.use()


def plot_de_results(csv_file, output_file):
    """
    Plot TPR and FPR from CSV results.
    
    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output plot file
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(require_input(
        csv_file,
        what="kidney.umi_de's results",
        source="run `python -m kidney.umi_de` first",
    ))
    
    # Get unique methods and p values
    methods = method_draw_order(df['method'])
    p_values = sorted(df['p'].unique())
    
    print(f"Found {len(methods)} methods: {methods}")
    print(f"Found {len(p_values)} p values")
    
    # Create figure with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2))
    
    # Plot TPR
    ax_tpr = axes[0]
    for method in methods:
        method_data = df[df['method'] == method].sort_values('p')
        tpr_means = method_data['tpr_mean'].values
        tpr_stds = method_data['tpr_std'].values
        p_vals = method_data['p'].values
        
        ax_tpr.errorbar(
            p_vals, tpr_means, yerr=tpr_stds,
            label=method_label(method),
            color=method_color(method),
            **plot_style.ERRORBAR
        )
    
    ax_tpr.set_xlabel(SAMPLING)
    ax_tpr.set_ylabel('TPR')
    ax_tpr.set_ylim([0, 1])
    
    # Plot FPR on a log axis, zero on its own row
    ax_fpr = axes[1]
    zero = plot_style.zero_row(df['fpr_mean'], df['fpr_std'])
    for method in methods:
        method_data = df[df['method'] == method].sort_values('p')
        plot_style.errorbar_log(ax_fpr, method_data['p'], method_data['fpr_mean'], method_data['fpr_std'], zero,
                     label=method_label(method), color=method_color(method))
    plot_style.log_axis_with_zero(ax_fpr, zero, (df['fpr_mean'] + df['fpr_std']).max())

    ax_fpr.set_xlabel(SAMPLING)
    ax_fpr.set_ylabel('FPR')
    
    plt.tight_layout()
    plot_style.legend_outside(fig)
    output_file = fig_name(output_file)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot TPR and FPR from DE test results CSV'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Results CSV of kidney.umi_de'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output plot file (PDF unless LNTEST_FIG_FORMAT says otherwise)'
    )
    
    args = parser.parse_args()
    
    plot_de_results(args.input, args.output)

