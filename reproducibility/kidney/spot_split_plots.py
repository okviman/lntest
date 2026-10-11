import os
import sys
import numpy as np
import pandas as pd

from paths import fig_name, require_input, results_dir
import plot_style
from method_colors import method_color, method_draw_order, method_label
import json
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D
from kidney.fpr_plots import SAMPLING, SPLIT, p_axis
plot_style.use()

GRID_COLS = 5  # the PR-curve grid's columns

def method_and_p_legends(fig, methods, selected_p, linestyle_map, p_symbol=SPLIT):
    """Two keys right of the axes: colour is the method, line style the split probability."""
    plot_style.legend_outside(fig, [Line2D([0], [0], color=method_color(m), lw=1.3) for m in methods],
                              [method_label(m) for m in methods], title='Method')
    plot_style.legend_outside(fig, [Line2D([0], [0], color='black', lw=1.3, linestyle=linestyle_map[p])
                                    for p in selected_p],
                              [f'{p:.2g}' for p in selected_p], title=p_symbol, y=0.6)


def plot_tpr_fpr_results(csv_file, output_file, p_symbol=SPLIT, odds=False):
    """Plot TPR and FPR vs p for all methods."""
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's spot-splitting results",
        source="run `python -m kidney.spot_split` first",
    ))
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2))
    draw_tpr_fpr(axes, df, p_symbol, odds)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def draw_tpr_fpr(axes, df, p_symbol=SPLIT, odds=False):
    """plot_tpr_fpr_results' two panels, drawn into two axes."""
    methods = method_draw_order(df['method'])
    method_to_color = {method: method_color(method) for method in methods}

    # Plot TPR
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        tpr_means = method_df['tpr_mean'].values
        tpr_stds = method_df['tpr_std'].values
        
        display_name = method_label(method)
        axes[0].errorbar(p_axis(p_vals, p_symbol, odds)[0], tpr_means, yerr=tpr_stds,
                        label=display_name,
                        color=method_to_color[method], **plot_style.ERRORBAR)
    
    axes[0].set_xlabel(p_axis([], p_symbol, odds)[1])
    axes[0].set_ylabel('TPR')
    axes[0].set_ylim([0, 1])
    
    # Plot FPR on a log axis, zero on its own row
    zero = plot_style.zero_row(df['fpr_mean'], df['fpr_std'])
    for method in methods:
        method_df = df[df['method'] == method].sort_values('p')
        plot_style.errorbar_log(axes[1], p_axis(method_df['p'], p_symbol, odds)[0], method_df['fpr_mean'], method_df['fpr_std'], zero,
                     label=method_label(method), color=method_to_color[method])
    plot_style.log_axis_with_zero(axes[1], zero, (df['fpr_mean'] + df['fpr_std']).max())

    axes[1].set_xlabel(p_axis([], p_symbol, odds)[1])
    axes[1].set_ylabel('FPR')


def plot_ap_pr_auc_results(csv_file, output_file, p_symbol=SPLIT, odds=False):
    """Plot AP and PR-AUC vs p for all methods."""
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's spot-splitting results",
        source="run `python -m kidney.spot_split` first",
    ))
    fig, axes = plt.subplots(1, 2, figsize=plot_style.figsize(2))
    draw_ap_pr_auc(axes, df, p_symbol, odds)
    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def draw_ap_pr_auc(axes, df, p_symbol=SPLIT, odds=False):
    """plot_ap_pr_auc_results' two panels, drawn into two axes."""
    methods = method_draw_order(df['method'])
    method_to_color = {method: method_color(method) for method in methods}

    # Plot AP
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        ap_means = method_df['ap_mean'].values
        ap_stds = method_df['ap_std'].values
        
        display_name = method_label(method)
        axes[0].errorbar(p_axis(p_vals, p_symbol, odds)[0], ap_means, yerr=ap_stds,
                         label=display_name,
                         color=method_to_color[method], **plot_style.ERRORBAR)
    axes[0].set_xlabel(p_axis([], p_symbol, odds)[1])
    axes[0].set_ylabel('Average Precision (AP)')
    axes[0].set_ylim([0, 1])
    
    # Plot PR-AUC
    for method in methods:
        method_df = df[df['method'] == method]
        method_df = method_df.sort_values('p')
        p_vals = method_df['p'].values
        pr_auc_means = method_df['pr_auc_mean'].values
        pr_auc_stds = method_df['pr_auc_std'].values
        
        display_name = method_label(method)
        axes[1].errorbar(p_axis(p_vals, p_symbol, odds)[0], pr_auc_means, yerr=pr_auc_stds,
                         label=display_name,
                         color=method_to_color[method], **plot_style.ERRORBAR)
    axes[1].set_xlabel(p_axis([], p_symbol, odds)[1])
    axes[1].set_ylabel('PR-AUC (Trapezoidal)')
    axes[1].set_ylim([0, 1])


def plot_pr_curves_all_p(json_file, output_file, q, lfc, p_symbol=SPLIT):
    """Plot Precision vs Recall curves for each p value (original style)."""
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    
    p_values = np.array(curve_data['p_values'])
    curves = curve_data['curves']
    
    # Get methods from first p value
    first_p_key = list(curves.keys())[0]
    methods = method_draw_order(curves[first_p_key])
    
    # Create subplots: one for each p value
    n_p = len(p_values)
    n_cols = min(GRID_COLS, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=plot_style.figsize(n_cols, n_rows))
    if n_p == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for p_idx, p in enumerate(p_values):
        draw_pr_curve(axes[p_idx], curves, p, methods, p_symbol)
    label_grid_outer(list(axes[:n_p]), n_cols)
    
    # Hide unused subplots
    for idx in range(n_p, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def label_grid_outer(axes, n_cols):
    """Keep axis labels and tick labels on the grid's left column and bottom row only; every panel spans [0, 1]."""
    for i, ax in enumerate(axes):
        if i % n_cols:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        if i + n_cols < len(axes):  # another panel below
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)


def draw_pr_curve(ax, curves, p, methods, p_symbol=SPLIT):
    """One panel of plot_pr_curves_all_p: every method's mean PR curve at p, one SD shaded."""
    method_to_color = {method: method_color(method) for method in methods}
    p_key = f"p_{p:.6f}"
    
    if p_key in curves:
        for method in methods:
            if method in curves[p_key]:
                method_data = curves[p_key][method]
                recall = np.array(method_data['recall'])
                precision_mean = np.array(method_data['precision_mean'])
                precision_std = method_data.get('precision_std')
                
                # Plot mean curve; one legend serves every panel
                ax.plot(recall, precision_mean,
                       label=method_label(method),
                       color=method_to_color[method], linewidth=1.3)
                
                # Plot std as shaded region (lighter shade)
                if precision_std is not None:
                    precision_std_arr = np.array(precision_std)
                    ax.fill_between(recall,
                                  precision_mean - precision_std_arr,
                                  precision_mean + precision_std_arr,
                                  alpha=0.1, color=method_to_color[method])
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{p_symbol} = {p:.2g}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def plot_pr_curves_selected_p(json_file, output_file, q, lfc, p_symbol=SPLIT):
    """Plot Precision vs Recall curves for selected p values (smallest, median, largest) in one plot."""
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    
    curves = curve_data['curves']
    methods = method_draw_order(curves[next(iter(curves))])
    fig, ax = plt.subplots(1, 1, figsize=plot_style.figsize())
    selected_p, linestyle_map = draw_pr_curves(ax, curves, methods, np.array(curve_data['p_values']))
    plt.tight_layout()
    method_and_p_legends(fig, methods, selected_p, linestyle_map, p_symbol)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def draw_pr_curves(ax, curves, methods, p_values_json):
    """Mean PR curves at the smallest, the median and the largest p: dotted, dashed and solid.

    Returns the three p and their line styles, for the key.
    """
    p_sorted = np.sort(p_values_json)
    p_min, p_max = p_sorted[0], p_sorted[-1]
    p_median = p_sorted[np.argmin(np.abs(p_sorted - np.median(p_sorted)))]
    selected_p = [p_min, p_median, p_max]
    linestyle_map = {p_min: ':', p_median: '--', p_max: '-'}
    for method in methods:
        for p in selected_p:
            p_key = f"p_{p:.6f}"
            if p_key in curves and method in curves[p_key]:
                md = curves[p_key][method]
                ax.plot(np.array(md['recall']), np.array(md['precision_mean']),
                        color=method_color(method), linewidth=1.3, linestyle=linestyle_map[p])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return selected_p, linestyle_map


def draw_auprc(ax, df, p_symbol=SPLIT, odds=False):
    """AUPRC against p: each method's mean over the replicates of its average precision (sklearn's, as the lymph node's
    AUPRC), with bars of one SD."""
    for method in method_draw_order(df['method']):
        d = df[df['method'] == method].sort_values('p')
        ax.errorbar(p_axis(d['p'], p_symbol, odds)[0], d['ap_mean'], yerr=d['ap_std'], color=method_color(method),
                    **plot_style.ERRORBAR)
    ax.set_xlabel(p_axis([], p_symbol, odds)[1])
    ax.set_ylabel('AUPRC')
    ax.set_ylim([0, 1])


def draw_pr_auc(ax, curves, methods, p_values_json, p_symbol=SPLIT, odds=False):
    """PR-AUC against p: the area under each p's mean PR curve, with bars of one SD of the replicates' areas."""
    # PR-AUC vs p (recomputed from JSON PR curves)
    # Use JSON p grid to ensure availability of curves
    p_values_for_auc = np.sort(p_values_json)
    for method in methods:
        pr_auc_list = []
        pr_auc_err = []
        p_valid = []
        for p in p_values_for_auc:
            p_key = f"p_{p:.6f}"
            if p_key in curves and method in curves[p_key]:
                md = curves[p_key][method]
                recall = np.array(md.get('recall', []))
                precision_mean = np.array(md.get('precision_mean', []))
                if recall.size > 1 and precision_mean.size == recall.size:
                    # Recompute AUC from mean curve
                    auc_val = np.trapz(precision_mean, recall)
                    pr_auc_list.append(auc_val)
                    # If std available, use it as yerr; else no error bar
                    pr_auc_std = md.get('pr_auc_curve_std', None)
                    if pr_auc_std is None:
                        pr_auc_err.append(np.nan)
                    else:
                        pr_auc_err.append(pr_auc_std)
                    p_valid.append(p)
        if len(p_valid) > 0:
            p_valid = np.array(p_valid)
            pr_auc_arr = np.array(pr_auc_list, dtype=float)
            pr_auc_err_arr = np.array(pr_auc_err, dtype=float)
            # Plot with error bars where available
            # If all yerr are NaN, omit yerr to avoid warnings
            if np.all(np.isnan(pr_auc_err_arr)):
                ax.errorbar(p_axis(p_valid, p_symbol, odds)[0], pr_auc_arr,
                                  color=method_color(method), **plot_style.ERRORBAR)
            else:
                ax.errorbar(p_axis(p_valid, p_symbol, odds)[0], pr_auc_arr, yerr=pr_auc_err_arr,
                                  color=method_color(method), **plot_style.ERRORBAR)
    ax.set_xlabel(p_axis([], p_symbol, odds)[1])
    ax.set_ylabel('PR-AUC')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major')


def plot_pr_auc(json_file, output_file, p_symbol=SPLIT, odds=False):
    """The summary figure's PR-AUC panel on its own."""
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    curves = curve_data['curves']
    methods = method_draw_order(curves[next(iter(curves))])
    fig, ax = plt.subplots(1, 1, figsize=plot_style.figsize())
    draw_pr_auc(ax, curves, methods, np.array(curve_data['p_values']), p_symbol, odds)
    plt.tight_layout()
    plot_style.legend_outside(fig, [Line2D([0], [0], color=method_color(m), marker='o', ms=plot_style.ERRORBAR['ms'], lw=plot_style.ERRORBAR['lw'])
                               for m in methods],
                              [method_label(m) for m in methods])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_summary_figure(csv_file, json_file, output_file, q, lfc, p_symbol=SPLIT, odds=False):
    """
    Create a summary figure with:
      Top row:    [TPR vs p]  [FPR vs p]
      Bottom row: [PR-AUC vs p]  [Selected PR curves (min/median/max p)]
    Keys right of the axes: colour is the method, line style the split probability of the PR curves.
    """
    # Load CSV and JSON
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's spot-splitting results",
        source="run `python -m kidney.spot_split` first",
    ))
    with open(json_file, 'r') as f:
        curve_data = json.load(f)
    p_values_json = np.array(curve_data['p_values'])
    curves = curve_data['curves']
    # Methods from CSV (ensures alignment with DE summaries)
    methods = method_draw_order(df['method'])
    method_to_color = {method: method_color(method) for method in methods}
    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=plot_style.figsize(2, 2))
    ax_tpr, ax_fpr = axes[0, 0], axes[0, 1]
    ax_prauc, ax_prcurves = axes[1, 0], axes[1, 1]
    # TPR vs p
    for method in methods:
        mdf = df[df['method'] == method].sort_values('p')
        ax_tpr.errorbar(p_axis(mdf['p'].values, p_symbol, odds)[0], mdf['tpr_mean'].values, yerr=mdf['tpr_std'].values,
                        color=method_to_color[method], **plot_style.ERRORBAR,
                        label=method_label(method))
    ax_tpr.set_xlabel(p_axis([], p_symbol, odds)[1])
    ax_tpr.set_ylabel('TPR')
    ax_tpr.set_ylim([0, 1])
    ax_tpr.tick_params(axis='both', which='major')
    # FPR vs p, on a log axis with zero on its own row
    zero = plot_style.zero_row(df['fpr_mean'], df['fpr_std'])
    for method in methods:
        mdf = df[df['method'] == method].sort_values('p')
        plot_style.errorbar_log(ax_fpr, p_axis(mdf['p'], p_symbol, odds)[0], mdf['fpr_mean'], mdf['fpr_std'], zero, color=method_to_color[method])
    plot_style.log_axis_with_zero(ax_fpr, zero, (df['fpr_mean'] + df['fpr_std']).max())
    ax_fpr.set_xlabel(p_axis([], p_symbol, odds)[1])
    ax_fpr.set_ylabel('FPR')
    ax_fpr.tick_params(axis='both', which='major')
    draw_pr_auc(ax_prauc, curves, methods, p_values_json, p_symbol, odds)
    # Selected PR curves (min, median, max p); the keys are drawn once, outside
    selected_p, linestyle_map = draw_pr_curves(ax_prcurves, curves, methods, p_values_json)
    plt.tight_layout()
    method_and_p_legends(fig, methods, selected_p, linestyle_map, p_symbol)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize results from vishd_test_shape_split_shared.py')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Input CSV file with results')
    parser.add_argument('--json_file', type=str, required=True,
                        help='Input JSON file with PR curve data')
    parser.add_argument('--output_prefix', type=str, default=str(results_dir(__file__) / 'spot_split'),
                        help='Output file prefix for plots (default: kidney/results/spot_split)')
    parser.add_argument('--downsampling', action='store_true',
                        help='Results are from umi_de: label the axis with the sampling probability')
    args = parser.parse_args()
    
    csv_file = args.csv_file
    json_file = args.json_file
    output_prefix = args.output_prefix
    p_symbol = SAMPLING if args.downsampling else SPLIT
    
    # Read q and lfc from CSV (should be same for all rows)
    df = pd.read_csv(require_input(
        csv_file,
        what="this arm's spot-splitting results",
        source="run `python -m kidney.spot_split` first",
    ))
    q = df['q'].iloc[0]
    lfc = df['lfc'].iloc[0]
    
    print(f"Visualizing results with q={q}, lfc={lfc}")
    
    for odds, suffix in ((False, ''), (True, '_odds')):
        print(f"Plotting TPR and FPR, AP and PR-AUC, PR-AUC, summary{' against (1 - p) / p' if odds else ''}...")
        plot_tpr_fpr_results(csv_file, fig_name(f"{output_prefix}_tpr_fpr{suffix}"), p_symbol, odds)
        plot_ap_pr_auc_results(csv_file, fig_name(f"{output_prefix}_ap_pr_auc{suffix}"), p_symbol, odds)
        plot_pr_auc(json_file, fig_name(f"{output_prefix}_pr_auc{suffix}"), p_symbol, odds)
        plot_summary_figure(csv_file, json_file, fig_name(f"{output_prefix}_summary{suffix}"), q, lfc, p_symbol, odds)

    print("Plotting PR curves...")
    plot_pr_curves_all_p(json_file, fig_name(f"{output_prefix}_pr_curves_grid"), q, lfc, p_symbol)
    plot_pr_curves_selected_p(json_file, fig_name(f"{output_prefix}_pr_curves"), q, lfc, p_symbol)
    
    print("Done!")

