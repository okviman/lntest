import os
import sys
import numpy as np
import statsmodels.stats.multitest as smm
import scanpy as sc
import scipy.sparse as sp
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

from paths import data_dir, fig_name, require_input, results_dir
import plot_style
from method_colors import method_color, method_draw_order, method_label, method_order

# Run from reproducibility/ as `python -m kidney.umi_de`, which puts that
# directory on sys.path -- no path manipulation needed.
from lntest import get_LN_lfcs as get_DELN_lfcs
from baselines import scanpy_sig_test
# spot_split's DE metrics, so the two kidney arms report the same quantities, PR curves included
from kidney.spot_split import (RECALL_GRID, de_test_single, interpolate_precision, save_pr_curve_data,
                               save_results_de, summarise_de)
plot_style.use()


def run_single_iteration(args):
    """Run a single iteration of the test with given parameters."""
    p, q, lfc, n_cells_remove, umis_base, library_sizes_base, seed = args
    
    # Set random seed for reproducibility within each iteration
    np.random.seed(seed)
    
    # Create a copy of the base data
    umis = umis_base.copy()
    library_sizes = library_sizes_base.copy()
    
    # Split cells by median depth (before downsampling)
    median_libsize = np.median(library_sizes)
    high_group_mask = library_sizes >= median_libsize
    low_group_mask = library_sizes < median_libsize
    
    # Randomly select genes with probability q
    n_genes = umis.shape[1]
    selected_genes = set(np.where(np.random.random(n_genes) < q)[0])
    
    # Randomly assign signs (+1 or -1) to selected genes
    true_signs = {g: np.random.choice([-1, 1]) for g in selected_genes}
    
    # Apply fold changes
    umis_modified = umis.copy()
    for gene_idx in selected_genes:
        sign = true_signs[gene_idx]
        fold_change = 2**lfc
        
        if sign == 1:
            # Multiply counts in HIGH group
            umis_modified[high_group_mask, gene_idx] = np.round(
                umis_modified[high_group_mask, gene_idx] * fold_change
            ).astype(int)
        else:  # sign == -1
            # Multiply counts in LOW group
            umis_modified[low_group_mask, gene_idx] = np.round(
                umis_modified[low_group_mask, gene_idx] * fold_change
            ).astype(int)
    
    # Downsample counts for each cell for each gene by ratio p
    # Using binomial downsampling (skip if p == 1.0)
    if p < 1.0:
        umis_modified = np.random.binomial(umis_modified.astype(int), p).astype(float)
    
    # Recalculate library sizes after downsampling
    library_sizes_modified = umis_modified.sum(1)
    
    # Split again after downsampling
    median_libsize_modified = np.median(library_sizes_modified)
    X = umis_modified[library_sizes_modified >= median_libsize_modified]
    Y = umis_modified[library_sizes_modified < median_libsize_modified]
    
    # Run the test
    results = de_test_single(X, Y, selected_genes, true_signs)
    # Keep each replicate's PR curve on the common recall grid only; full-length curves for thousands of
    # replicates would not fit in memory, and summarise_de interpolates onto this grid anyway.
    for metrics in results.values():
        metrics["precision_curve"] = interpolate_precision(metrics["precision_curve"], metrics["recall_curve"])
        metrics["recall_curve"] = RECALL_GRID
    return results


def run_tests_for_p(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs):
    """Run n_reps iterations for a given p value in parallel."""
    # Create arguments for each iteration
    seeds = np.random.randint(0, 2**31, size=n_reps)
    args_list = [(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, seed) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(run_single_iteration, args_list)
    
    return summarise_de(results_list)


def plot_results(p_values, results_by_p, output_file, q, lfc):
    """Plot TPR, FPR, FNR, and TNR vs p for all methods."""
    methods = method_draw_order(results_by_p[p_values[0]])
    colors = [method_color(m) for m in methods]
    
    fig, axes = plt.subplots(2, 2, figsize=plot_style.figsize(2, 2))
    
    # Plot TPR
    for idx, method in enumerate(methods):
        tpr_means = [results_by_p[p][method]["tpr_mean"] for p in p_values]
        tpr_stds = [results_by_p[p][method]["tpr_std"] for p in p_values]
        
        axes[0, 0].errorbar(p_values, tpr_means, yerr=tpr_stds, 
                        label=method_label(method), 
                        color=colors[idx], **plot_style.ERRORBAR)
    
    axes[0, 0].set_xlabel(r'$p_\mathrm{sample}$')
    axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_ylim([0, 1])
    
    # Plot FPR
    for idx, method in enumerate(methods):
        fpr_means = [results_by_p[p][method]["fpr_mean"] for p in p_values]
        fpr_stds = [results_by_p[p][method]["fpr_std"] for p in p_values]
        
        axes[0, 1].errorbar(p_values, fpr_means, yerr=fpr_stds, 
                        label=method_label(method), 
                        color=colors[idx], **plot_style.ERRORBAR)
    
    axes[0, 1].set_xlabel(r'$p_\mathrm{sample}$')
    axes[0, 1].set_ylabel('FPR')
    axes[0, 1].set_ylim([0, 1])
    
    # Plot FNR
    for idx, method in enumerate(methods):
        fnr_means = [results_by_p[p][method]["fnr_mean"] for p in p_values]
        fnr_stds = [results_by_p[p][method]["fnr_std"] for p in p_values]
        
        axes[1, 0].errorbar(p_values, fnr_means, yerr=fnr_stds, 
                        label=method_label(method), 
                        color=colors[idx], **plot_style.ERRORBAR)
    
    axes[1, 0].set_xlabel(r'$p_\mathrm{sample}$')
    axes[1, 0].set_ylabel('False Negative Rate (FNR)')
    axes[1, 0].set_ylim([0, 1])
    
    # Plot TNR
    for idx, method in enumerate(methods):
        tnr_means = [results_by_p[p][method]["tnr_mean"] for p in p_values]
        tnr_stds = [results_by_p[p][method]["tnr_std"] for p in p_values]
        
        axes[1, 1].errorbar(p_values, tnr_means, yerr=tnr_stds, 
                        label=method_label(method), 
                        color=colors[idx], **plot_style.ERRORBAR)
    
    axes[1, 1].set_xlabel(r'$p_\mathrm{sample}$')
    axes[1, 1].set_ylabel('True Negative Rate (TNR)')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plot_style.legend_outside(fig)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DE detection with parallel execution')
    parser.add_argument('--n_cells_remove', type=int, required=True,
                        help='Number of cells with lowest total count to remove')
    parser.add_argument('--q', type=float, required=True,
                        help='Fraction of genes to introduce DE')
    parser.add_argument('--lfc', type=float, required=True,
                        help='Log2 fold change to apply')
    parser.add_argument('--p_min', type=float, default=0.1,
                        help='Minimum downsample ratio p (default: 0.1)')
    parser.add_argument('--p_max', type=float, default=1.0,
                        help='Maximum downsample ratio p (default: 1.0)')
    parser.add_argument('--p_steps', type=int, default=10,
                        help='Number of p values to test (default: 10)')
    parser.add_argument('--n_reps', type=int, default=100,
                        help='Number of repetitions per p value (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: all available cores)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the per-repetition seeds (default: 0)')
    parser.add_argument('--output', type=str, default=str(results_dir(__file__) / fig_name('de_test_plot')),
                        help='Output file for the plot (default: kidney/results/de_test_plot.pdf)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Output file for results CSV (default: auto-generated from output name)')
    args = parser.parse_args()
    
    n_cells_remove = args.n_cells_remove
    q = args.q
    lfc = args.lfc
    p_min = args.p_min
    p_max = args.p_max
    p_steps = args.p_steps
    n_reps = args.n_reps
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    output_file = args.output
    
    # Generate results file name if not provided
    if args.results_file is None:
        base_name = os.path.splitext(output_file)[0]
        results_file = f"{base_name}_results.csv"
    else:
        results_file = args.results_file
    
    if not (0 < p_min <= p_max <= 1.0):
        raise ValueError("p_min and p_max must be between 0 and 1 (inclusive), and p_min <= p_max")
    
    if not (0 < q <= 1.0):
        raise ValueError("q must be between 0 and 1")
    
    print(f"Loading data...")
    # Load spatial ovary data (raw read counts in adata.X)
    h5ad = sc.read_h5ad(require_input(
        data_dir(__file__) / "merged_blobs_in_cluster_5.h5ad",
        what="the merged glomerular-capsule matrix arms D and E subsample",
        source="Visium HD kidney capsules, cluster-5 blobs. Not in this "
               "repository, but public and rebuildable: run kidney/preprocess.ipynb "
               "top to bottom (downloads 6.5 GB from 10x)",
    ))
    # Make gene names unique using gene_id
    h5ad.var = h5ad.var.reset_index().set_index('gene_ids')
    # Robustly convert to a dense numpy array
    X_raw = h5ad.X
    if sp.issparse(X_raw):
        umis_w_zeros = X_raw.toarray()
    else:
        umis_w_zeros = np.asarray(X_raw)

    print(f"Preprocessing data...")
    # preprocessing
    # remove genes with zero counts accross all cells
    idx = umis_w_zeros.sum(0) > 0
    umis = umis_w_zeros[:, idx]
    n_genes = umis.shape[-1]
    n_cells = umis.shape[0]

    # remove the n_cells_remove cells with lowest read counts
    library_sizes = umis.sum(1)
    order = np.argsort(library_sizes)
    if order.size > n_cells_remove:
        keep_idx = order[n_cells_remove:]
        umis_base = umis[keep_idx].copy()
        library_sizes_base = library_sizes[keep_idx].copy()
    else:
        umis_base = umis.copy()
        library_sizes_base = library_sizes.copy()
    
    print(f"Data loaded. Shape: {umis_base.shape}")
    print(f"Parameters: q={q}, lfc={lfc}")
    print(f"Running tests for {p_steps} p values, {n_reps} repetitions each, using {n_jobs} cores...")
    
    # Generate p values
    p_values = np.linspace(p_min, p_max, p_steps)
    
    np.random.seed(args.seed)
    
    # Run tests for each p value
    results_by_p = {}
    for i, p in enumerate(p_values):
        print(f"Processing p={p:.3f} ({i+1}/{p_steps})...")
        results_by_p[p] = run_tests_for_p(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs)
        # Print summary for this p
        for method, metrics in results_by_p[p].items():
            print(f"  {method}: TPR={metrics['tpr_mean']:.4f}±{metrics['tpr_std']:.4f}, "
                  f"FPR={metrics['fpr_mean']:.4f}±{metrics['fpr_std']:.4f}, "
                  f"FNR={metrics['fnr_mean']:.4f}±{metrics['fnr_std']:.4f}, "
                  f"TNR={metrics['tnr_mean']:.4f}±{metrics['tnr_std']:.4f}")
    
    print(f"\nSaving results...")
    save_results_de(p_values, results_by_p, results_file, q, lfc)
    save_pr_curve_data(p_values, results_by_p, f"{os.path.splitext(results_file)[0].removesuffix('_results')}_pr_curve_data.json", q, lfc)
    
    print(f"Generating plots...")
    # plot_results(p_values, results_by_p, output_file, q, lfc)
    print("Done!")

