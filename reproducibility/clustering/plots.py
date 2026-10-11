#!/usr/bin/env python3
"""
Script to generate clustering metrics plots using precomputed metric CSV files.

Signature-based metrics are produced for two signature sets, distinguished by a
filename tag: the top-N significant genes per cluster ("top{N}") and the full set
of significant genes per cluster ("allsig"). For each tag this script expects:
    - avg_jaccard_{tag}_by_resolution.csv
    - avg_recall_{tag}_by_resolution.csv
    - best_match_{tag}_by_resolution.csv
    - jaccard_matrix_{tag}_{method}_{resolutionkey}.csv  (for heatmaps)
    - recall_matrix_{tag}_{method}_{resolutionkey}.csv    (for heatmaps)

Signature-set-independent metrics:
    - avg_lfc_by_resolution.csv
    - n_sig_genes_by_resolution.csv

Usage:
    python plot_clustering_metrics.py --metrics-dir clustering/results/metrics [--config config.yaml]

If config.yaml is provided, cell type marker order and method name mapping can be inferred for nicer plots.
"""

import argparse
import os
import textwrap
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from paths import fig_name
import plot_style
from method_colors import MEDIANPROPS, method_color, method_label, method_order

def load_config(config_path):
    """Load configuration from YAML file, if present."""
    if config_path is None:
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# One-line y labels of the summary box plots
YLABELS = {
    "n_sig_genes": "Significant DEGs per cluster",
    "avg_abs_lfc_sig": "Mean |LFC| of significant DEGs",
    "avg_jaccard_sig": "Mean Jaccard to markers",
}


def jaccard_title(tag, top_genes):
    """The Jaccard box plots' title: which signature they compare."""
    return "All significant genes" if tag == "allsig" else f"Top {top_genes} genes"


_PRETTY_RES = {
    "verylow": "Very low", "low": "Low", "mid": "Mid", "medium": "Medium",
    "high": "High", "veryhigh": "Very high",
}


def resolution_axis(metrics_dir, config):
    """The configured resolution keys, the resolutions in plotting order, and a labeller naming each
    resolution with its number of clusters."""
    leiden_resolutions = config.get('leiden_resolutions', None)
    if leiden_resolutions:
        resolution_keys = [r[1] if isinstance(r, (list, tuple)) and len(r)==2 else str(r) for r in leiden_resolutions]
    else:
        resolution_keys = None
    top_genes = config.get('top_genes', 10)

    def _read(name):
        path = os.path.join(metrics_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    df_lfc = _read("avg_lfc_by_resolution.csv")
    df_nsig = _read("n_sig_genes_by_resolution.csv")
    ref = _read(f"avg_jaccard_top{top_genes}_by_resolution.csv")

    def _resolution_order(df):
        if resolution_keys:
            return resolution_keys
        vals = df["resolution"].astype(str).unique()
        return sorted(vals, key=lambda v: float(v) if str(v).replace('.', '', 1).isdigit() else str(v))

    resolution_order = resolution_keys
    if resolution_order is None:
        for df in (df_lfc, df_nsig, ref):
            if df is not None and "resolution" in df.columns:
                resolution_order = _resolution_order(df)
                break

    # Number of clusters per resolution (same across DE methods); derived from the
    # per-cluster n_sig CSV so we can annotate axis/subplot labels.
    n_clusters_by_res = {}
    if df_nsig is not None and {"resolution", "cluster"}.issubset(df_nsig.columns):
        n_clusters_by_res = (
            df_nsig.astype({"resolution": str})
            .groupby("resolution")["cluster"].nunique().to_dict()
        )

    def resolution_label(rk):
        """Map a resolution key (e.g. "leiden_verylow") to "Very low\n(2 clusters)"."""
        rk = str(rk)
        token = rk[len("leiden_"):] if rk.startswith("leiden_") else rk
        name = _PRETTY_RES.get(token, token.replace("_", " ").capitalize())
        n = n_clusters_by_res.get(rk)
        return f"{name}\n({n} clusters)" if n is not None else name

    def cluster_count(rk):
        """A resolution's number of clusters, or its name when the count is unknown."""
        n = n_clusters_by_res.get(str(rk))
        return str(n) if n is not None else resolution_label(rk)

    return resolution_keys, resolution_order, resolution_label, cluster_count


def draw_box(ax, df, ycol, ylabel, title, resolution_order, cluster_count, log=False):
    """Boxplot + stripplot of `ycol` grouped by resolution and method, with its method key, into ax; on a log
    y-axis if log, set first so the boxes are computed on it."""
    if log:
        ax.set_yscale("log")
    df = df.copy()
    df["resolution"] = df["resolution"].astype(str)
    df["method"] = df["method"].map(method_label)
    hue_order = method_order(df["method"])
    sns.boxplot(
        data=df, x="resolution", y=ycol, hue="method", showfliers=False,
        order=resolution_order, hue_order=hue_order,
        palette={m: method_color(m) for m in hue_order}, saturation=1,
        medianprops=MEDIANPROPS, ax=ax,
    )
    for box in ax.patches:
        box.set_edgecolor("none")
    np.random.seed(0)  # seaborn's jitter draws from the global RNG
    sns.stripplot(
        data=df, x="resolution", y=ycol, hue="method", dodge=True, alpha=0.8,
        palette={m: method_color(m) for m in hue_order}, edgecolor='0.2', linewidth=0.4,
        zorder=10, size=2, order=resolution_order, hue_order=hue_order, ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order)
    for handle in handles[:n]:
        handle.set_edgecolor("none")
    ax.legend(handles[:n], labels[:n], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, alignment="left")
    ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
    ax.set_xticklabels([cluster_count(t.get_text()) for t in ax.get_xticklabels()])
    ax.set_ylabel(textwrap.fill(ylabel, 40))
    ax.set_xlabel("Number of clusters")
    if title:
        ax.set_title(title)
    plot_style.boxplot_grid(ax)


# Known DE methods, longest first so e.g. "t_test" is matched before "t".
KNOWN_METHODS = sorted(["ln", "t_test", "wilcoxon"], key=len, reverse=True)


def _parse_matrix_name(path, file_prefix):
    """Parse {file_prefix}_{tag}_{method}_{reskey}.csv into (tag, method, reskey).

    The tag (e.g. "top20"/"allsig") has no underscore, while both method
    (e.g. "t_test") and reskey (e.g. "leiden_mid") may contain "_", so we
    split off the tag first and then match a known method prefix.
    """
    base = os.path.basename(path).replace(".csv", "")
    rest = base[len(file_prefix) + 1:]  # {tag}_{method}_{reskey}
    tag, rest = rest.split("_", 1)
    method, reskey = None, rest
    for m in KNOWN_METHODS:
        if rest.startswith(m + "_"):
            method, reskey = m, rest[len(m) + 1:]
            break
    if method is None:  # fallback: split before the resolution key
        method, _, reskey = rest.partition("_leiden")
        reskey = "leiden" + reskey
    return tag, method, reskey


def heatmap_groups(metrics_dir, file_prefix, resolution_keys):
    """(tag, method) -> (resolution keys in plotting order, {resolution key: cluster x cell-type matrix})."""
    from glob import glob
    resmaps = {}
    for path in sorted(glob(os.path.join(metrics_dir, f"{file_prefix}_*_*_*.csv"))):
        tag, method, reskey = _parse_matrix_name(path, file_prefix)
        resmaps.setdefault((tag, method), {})[reskey] = path
    groups = {}
    for key, resmap in resmaps.items():
        if resolution_keys:
            ordered = [rk for rk in resolution_keys if rk in resmap]
            ordered += [rk for rk in resmap if rk not in ordered]
        else:
            ordered = sorted(resmap)
        groups[key] = (ordered, {rk: pd.read_csv(resmap[rk], index_col=0) for rk in ordered})
    return groups


def shared_vmax(groups, tag, reskey=None):
    """The largest value in any method's matrices of this tag, so the methods share a colour scale;
    only at one resolution if reskey is given."""
    observed = max((float(m.values.max()) for (t, _), (_, mats) in groups.items() if t == tag
                    for rk, m in mats.items() if reskey in (None, rk)), default=1.0)
    return observed if observed > 0 else 1.0


def draw_heatmap(ax, mat, vmax, title, cluster_numbers=True, label_size=None):
    """One cluster x cell-type matrix, viridis from 0 to vmax; the clusters numbered unless cluster_numbers is False,
    the cell types in label_size points if given."""
    sns.heatmap(
        mat.values, ax=ax, annot=False, cmap="viridis",
        vmin=0.0, vmax=vmax, cbar=False,
        xticklabels=mat.columns, yticklabels=mat.index if cluster_numbers else False,
    )
    ax.tick_params(left=True, bottom=True, length=3.5)  # seaborn hides heatmap ticks
    ax.set_title(title)
    ax.set_xlabel("Cell type marker")
    ax.set_ylabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=90)
    if label_size:
        ax.tick_params(axis="x", labelsize=label_size)


def label_heatmap_grid(axes, row_labels, col_labels, cell_types=True):
    """Label a grid of draw_heatmap axes (a list of rows) on its outer edge only: col_labels over the top row,
    row_labels beside the first column, the cell types under the bottom row unless cell_types is False, and
    their axis label once, under the middle column."""
    last = len(axes) - 1
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            ax.set_title(col_labels[c] if r == 0 else "")
            ax.set_ylabel(row_labels[r] if c == 0 else "")
            if r < last or not cell_types:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if r < last or not cell_types or c != len(row) // 2:
                ax.set_xlabel("")


def main():
    parser = argparse.ArgumentParser(
        description='Generate clustering metrics plots from CSV summaries'
    )
    parser.add_argument(
        '--metrics-dir', type=str, required=True,
        help='Directory containing metric CSV files (e.g. clustering/results/metrics)'
    )
    parser.add_argument(
        '--config', type=str, required=False,
        help='Optional YAML config file for cell type and resolution order'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save plots (default: a sibling "figures" folder next to metrics-dir)'
    )
    parser.add_argument(
        '--skip-heatmaps', action='store_true',
        help='Skip plotting Jaccard/recall heatmaps (needs per-cluster CSVs)'
    )
    args = parser.parse_args()
    metrics_dir = args.metrics_dir
    # Default figures location: sibling "figures" folder next to the metrics dir
    default_fig_dir = os.path.join(os.path.dirname(os.path.normpath(metrics_dir)) or ".", "figures")
    output_dir = args.output_dir or default_fig_dir

    config = load_config(args.config) if args.config else {}
    cell_types = list(config.get('markers', {}).keys()) if 'markers' in config else None
    top_genes = config.get('top_genes', 10)

    os.makedirs(output_dir, exist_ok=True)
    plot_style.use()

    # Signature sets to plot: top-N significant genes and all significant genes.
    top_tag = f"top{top_genes}"
    sig_tags = [top_tag, "allsig"]
    tag_label = {top_tag: f"top {top_genes} sig genes", "allsig": "all significant genes"}

    def _read(name):
        path = os.path.join(metrics_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    # ---- Read signature-set-independent metric CSVs ----
    print("> Loading metric summary CSVs...")
    df_lfc = _read("avg_lfc_by_resolution.csv")
    df_nsig = _read("n_sig_genes_by_resolution.csv")

    resolution_keys, resolution_order, resolution_label, cluster_count = resolution_axis(metrics_dir, config)

    def boxplot(df, ycol, ylabel, title, fname, log=False):
        """Boxplot + stripplot of `ycol` grouped by resolution and method."""
        if df is None or ycol not in df.columns:
            return
        plt.figure(figsize=(5.6, 3.0))
        draw_box(plt.gca(), df, ycol, ylabel, title, resolution_order, cluster_count, log=log)
        plt.tight_layout()
        plot_style.rasterize_dense(plt.gcf())
        plt.savefig(fig_name(os.path.join(output_dir, fname)), dpi=plot_style.RASTER_DPI, bbox_inches='tight')
        plt.close()

    # ---- Signature-set-independent boxplots ----
    print("> Plotting summary boxplots...")
    boxplot(df_lfc, "avg_abs_lfc_sig", YLABELS["avg_abs_lfc_sig"], None, "avg_lfc_by_resolution.png", log=True)
    boxplot(df_nsig, "n_sig_genes", YLABELS["n_sig_genes"], None, "n_sig_genes_by_resolution.png")

    # ---- Signature-set-dependent boxplots, one set per tag ----
    for tag in sig_tags:
        lab = tag_label[tag]
        boxplot(
            _read(f"avg_jaccard_{tag}_by_resolution.csv"), "avg_jaccard_sig",
            YLABELS["avg_jaccard_sig"], jaccard_title(tag, top_genes), f"avg_jaccard_{tag}_by_resolution.png",
        )
        boxplot(
            _read(f"avg_recall_{tag}_by_resolution.csv"), "avg_recall_sig",
            "Average recall of cell-type markers by cluster signatures",
            f"Avg recall of markers ({lab})", f"avg_recall_{tag}_by_resolution.png",
        )
        df_best = _read(f"best_match_{tag}_by_resolution.csv")
        boxplot(
            df_best, "best_recall",
            "Best-matching cluster recall of cell-type markers",
            f"Best-match recall of markers per cell type ({lab})",
            f"best_recall_{tag}_by_resolution.png",
        )
        boxplot(
            df_best, "best_jaccard",
            "Best-matching cluster Jaccard to cell-type markers",
            f"Best-match Jaccard per cell type ({lab})",
            f"best_jaccard_{tag}_by_resolution.png",
        )

    # ---- Per-method grouped heatmaps (Jaccard and recall), one figure per tag/method ----
    if not args.skip_heatmaps:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        def plot_grouped_heatmaps(file_prefix, out_prefix, cbar_label, metric_name, vmax_mode):
            """Build one figure per (tag, DE method), with a subplot per resolution.

            vmax_mode: "one" fixes the color scale to [0, 1]; "observed" uses the
            maximum value across every method's matrices of that tag.
            """
            groups = heatmap_groups(metrics_dir, file_prefix, resolution_keys)
            if not groups:
                print(f"  (No {file_prefix}_*.csv files found; skipping {metric_name} heatmaps.)")
                return

            for (tag, method), (ordered, mats) in groups.items():
                mlabel = method_label(method)
                vmax = shared_vmax(groups, tag) if vmax_mode == "observed" else 1.0

                n = len(ordered)
                fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 5.2), squeeze=False)
                axes = axes[0]
                for ax, rk in zip(axes, ordered):
                    draw_heatmap(ax, mats[rk], vmax, resolution_label(rk))

                sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0.0, vmax=vmax))
                fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.02, label=cbar_label)
                taglab = tag_label.get(tag, tag)
                scale_note = f" (color scale 0–{vmax:.3g}, shared across methods)" if vmax_mode == "observed" else ""
                fig.suptitle(
                    f"{mlabel}: {metric_name} of cluster signatures to cell-type markers "
                    f"— {taglab}{scale_note}",
                )
                savepath = fig_name(os.path.join(output_dir, f"{out_prefix}_{tag}_{method}"))
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                plt.close(fig)

        print("> Plotting Jaccard heatmaps (cluster x cell type), vmax = observed max ...")
        plot_grouped_heatmaps(
            "jaccard_matrix", "jaccard_heatmaps", "Jaccard Index", "Jaccard", vmax_mode="observed",
        )
        print("> Plotting recall heatmaps (cluster x cell type) ...")
        plot_grouped_heatmaps(
            "recall_matrix", "recall_heatmaps", "Recall (fraction of markers)", "Recall", vmax_mode="one",
        )

    print("\nPlots complete! Results saved to", output_dir)

if __name__ == '__main__':
    main()
