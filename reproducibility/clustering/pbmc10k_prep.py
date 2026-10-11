#!/usr/bin/env python3
"""PBMC10k's RNA as this arm's input: the second, deeper PBMC dataset for clustering/heldout_fp.py.

Reads the CITE-seq arm's 5' PBMC 10k matrix, keeps the cells that arm keeps (200-6,000 genes, at most 10%
mitochondrial counts, at least 500 antibody counts; citeseq/R/pbmc10k_process.R), removes the doublets scanpy's
Scrublet predicts with its defaults, and writes the remaining cells' gene-expression counts to data/pbmc10k_rna.h5ad,
which config_pbmc10k.yaml names; each QC-passing cell's doublet score and call go to results/pbmc10k/doublets.csv.
clustering/de.py then clusters and tests it as it does PBMC3k.

    python -m fetch_data --arm citeseq             # the input, if missing
    python -m clustering.pbmc10k_prep              # from reproducibility/, seconds
    python -m clustering.de --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k --skip-mast
    python -m clustering.heldout_fp --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k
"""

from pathlib import Path

import numpy as np
import scanpy as sc

from paths import require_input, results_dir

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / 'citeseq' / 'data' / 'sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5'
OUT = HERE / 'data' / 'pbmc10k_rna.h5ad'
DOUBLETS = results_dir(__file__) / 'pbmc10k' / 'doublets.csv'


def main():
    full = sc.read_10x_h5(str(require_input(RAW, what="the 5' PBMC 10k matrix",
                                            source='python -m fetch_data --arm citeseq')), gex_only=False)
    full.var_names_make_unique()
    rna = full[:, full.var['feature_types'] == 'Gene Expression'].copy()
    adt = full[:, full.var['feature_types'] == 'Antibody Capture'].X
    n_genes = np.asarray((rna.X > 0).sum(1)).ravel()
    counts = np.asarray(rna.X.sum(1)).ravel()
    mito = np.asarray(rna[:, rna.var_names.str.startswith('MT-')].X.sum(1)).ravel() / counts
    keep = (n_genes >= 200) & (n_genes <= 6000) & (mito <= 0.10) & (np.asarray(adt.sum(1)).ravel() >= 500)
    rna = rna[keep].copy()
    sc.pp.scrublet(rna, random_state=0)
    calls = rna.obs[['doublet_score', 'predicted_doublet']].assign(total_counts=np.asarray(rna.X.sum(1)).ravel())
    calls.rename_axis('barcode').to_csv(DOUBLETS)
    print(f'wrote {DOUBLETS}: {int(calls["predicted_doublet"].sum())} of {len(calls)} cells called doublets, '
          f'threshold {rna.uns["scrublet"]["threshold"]:.3f}')
    rna = rna[~rna.obs['predicted_doublet'].to_numpy()].copy()
    rna.obs = rna.obs[[]]
    rna.uns = {}
    rna.var = rna.var[['gene_ids']]
    OUT.parent.mkdir(exist_ok=True)
    rna.write(OUT)
    print(f'wrote {OUT}: {rna.n_obs} of {full.n_obs} cells, {rna.n_vars} genes')


if __name__ == '__main__':
    main()
