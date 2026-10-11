#!/usr/bin/env python3
"""
Script to perform differential expression analysis between two specified cell types.

This script runs DE using both LN test and Scanpy's t-test (and wilcoxon) and MAST,
saving all results to the updated AnnData file. Uses shared de_utils for all DE steps.

Usage:
    python run_celltype_de.py --config config.yaml [--output-dir celltype/results/]
"""

import argparse
import os
import sys
import yaml
import scanpy as sc
import scipy.sparse as sp
import pandas as pd

from paths import require_input, results_dir

# Run from reproducibility/ as `python -m celltype.de`, which puts that
# directory on sys.path -- no path manipulation needed.
from de_utils import (
    prepare_adata_layers,
    run_ln_de,
    run_ttest_de,
    run_wilcoxon_de,
    run_mast_de,
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    required_fields = ["adata_path", "celltype_column", "celltype_values"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    if not isinstance(config["celltype_values"], (list, tuple)) or len(config["celltype_values"]) != 2:
        raise ValueError("celltype_values must be a list of two values: [celltype_1, celltype_2]")
    merge = config.get("celltype_merge") or {}
    if not isinstance(merge, dict) or not all(isinstance(v, list) for v in merge.values()):
        raise ValueError("celltype_merge must map a new label to a list of existing labels")
    subset = config.get("subset") or {}
    if not isinstance(subset, dict):
        raise ValueError("subset must map an obs column to the single value to keep")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Perform differential expression analysis between two specified cell types"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output-dir", type=str, default=str(results_dir(__file__)), help="Directory to save output files (default: celltype/results)")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    adata_path = config["adata_path"]
    group_col = config["celltype_column"]
    group_1, group_2 = config["celltype_values"]

    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(require_input(
        adata_path,
        what="the AnnData this arm compares two cell types in",
        source="config.yaml's adata_path, resolved from reproducibility/. "
               "For Kang: python -m fetch_data --arm celltype",
    ))

    # Restrict to one condition before anything else, so the gene filter below
    # is computed on the population actually analysed.
    for col, value in (config.get("subset") or {}).items():
        before = adata.shape[0]
        adata = adata[adata.obs[col].astype(str) == str(value)].copy()
        print(f"Subset {col} == {value!r}: {before} -> {adata.shape[0]} cells")

    # Collapse labels before selecting the two groups. celltype_merge exists
    # because the committed reference markers (results/kang_{B,T}_markers.csv)
    # were computed with CD4 and CD8 pooled into one T group; comparing against
    # them without pooling would score one population's DE against another's
    # markers.
    merge = config.get("celltype_merge") or {}
    if merge:
        mapping = {old: new for new, olds in merge.items() for old in olds}
        col = config["celltype_column"]
        adata.obs[col] = adata.obs[col].astype(str).replace(mapping)
        for new, olds in merge.items():
            print(f"Merged {olds} -> {new!r}: {int((adata.obs[col] == new).sum())} cells")

    # Basic filtering
    print("Filtering cells with fewer than 200 genes expressed...")
    sc.pp.filter_cells(adata, min_genes=200)
    print("Filtering genes detected in fewer than 3 cells...")
    sc.pp.filter_genes(adata, min_cells=3)
    if "n_genes" not in adata.obs:
        adata.obs["n_genes"] = (adata.X > 0).sum(axis=1).A1 if sp.issparse(adata.X) else (adata.X > 0).sum(axis=1)
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if sp.issparse(adata.X) else adata.X.sum(axis=1)

    # Layers for DE (de_utils)
    prepare_adata_layers(adata)

    # Only retain the two specified groups
    mask = adata.obs[group_col].isin([group_1, group_2])
    adata = adata[mask].copy()
    print(f"Number of cells retained for DE: {adata.shape[0]}")
    adata.obs[group_col] = pd.Categorical(adata.obs[group_col], categories=[group_1, group_2])

    # Run DE using de_utils (same as before: LN, t-test, wilcoxon, MAST)
    print("Running differential expression tests for the specified cell types...")
    ln_key = f"ln_{group_1}_vs_{group_2}"
    print(f"  Running LN test for {group_1} vs {group_2} ...")
    run_ln_de(
        adata,
        group_col,
        key_added=ln_key,
        layer="norm_counts",
        groups=[group_1],
        reference=group_2,
        sparse=True,
    )
    t_test_key = f"t_test_{group_1}_vs_{group_2}"
    print(f"  Running Scanpy t-test for {group_1} vs {group_2} ...")
    run_ttest_de(
        adata,
        group_col,
        key_added=t_test_key,
        layer="log1p_norm",
        use_raw=False,
        groups=[group_1],
        reference=group_2,
    )
    wilcoxon_key = f"wilcoxon_{group_1}_vs_{group_2}"
    print(f"  Running Scanpy wilcoxon test for {group_1} vs {group_2} ...")
    run_wilcoxon_de(
        adata,
        group_col,
        key_added=wilcoxon_key,
        layer="log1p_norm",
        use_raw=False,
        groups=[group_1],
        reference=group_2,
    )
    mast_key = f"mast_{group_1}_vs_{group_2}"
    print(f"  Running MAST test for {group_1} vs {group_2} ...")
    run_mast_de(
        adata,
        group_col,
        group_1=group_1,
        group_2=group_2,
        log1p_layer="log1p_norm",
        key_added=mast_key,
    )

    print("All DE tests complete.")
    base_name = os.path.splitext(os.path.basename(adata_path))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_{group_1}_vs_{group_2}_DE.h5ad")
    print(f"Saving updated AnnData with all DE results to {output_file} ...")
    adata.write(output_file)
    print("Done.")


if __name__ == "__main__":
    main()
