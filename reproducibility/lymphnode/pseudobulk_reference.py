#!/usr/bin/env python3
"""The lymph node's ground truth: a pseudo-bulk PyDESeq2 test of Cluster-3 against Cluster-1, on the cells the
subsampling runs on, at full depth.

Every 2 um spot of a cell is summed, and cells and genes pass the config's own filters (min_spots_per_cell,
min_counts_after_subsampling, min_cells_per_gene in each cluster), so every replicate's cells and genes are in the
reference. Each cluster's cells are shuffled (seed 0) and cut into pools of POOL cells, summed; the pools are
PyDESeq2's samples. Wald test of Cluster-3 against Cluster-1, BH over every gene (no independent filtering), DESeq2's
convention for what counts as a marker; the methods scored against it are called under Bonferroni. A gene is a
reference marker of Cluster-3 if its adjusted p is below ALPHA and its LFC positive, of Cluster-1 if negative. Writes
the CSV the config names as `reference`, which lymphnode/subsampling.py scores every method against.

    python -m lymphnode.pseudobulk_reference --config lymphnode/config_50rep.yaml    # from reproducibility/, a minute
"""

import argparse
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy import sparse

from lymphnode.subsampling import DEFAULT_MIN_COUNTS_AFTER_SUBSAMPLING, DEFAULT_MIN_SPOTS_PER_CELL, load_config
from paths import require_input

POOL = 200
ALPHA = 0.05
SEED = 0
CONTRAST = ('Cluster-3', 'Cluster-1')


def cell_counts(adata, cell_column, cluster_column):
    """Each cell's summed spot counts, its cluster and its number of spots."""
    codes, cells = pd.factorize(adata.obs[cell_column].astype(str).str.strip())
    to_cell = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(cells), len(codes)))
    counts = (to_cell @ sparse.csr_matrix(adata.X)).tocsr()
    clusters = pd.Series(adata.obs[cluster_column].astype(str).str.strip().values).groupby(codes).first().values
    return counts, clusters, np.bincount(codes)


def filtered(counts, clusters, n_spots, config):
    """The arm's cell and gene filters at full depth, repeated until nothing changes, as the subsampling applies them."""
    cells = (n_spots >= config.get('min_spots_per_cell', DEFAULT_MIN_SPOTS_PER_CELL))
    genes = np.ones(counts.shape[1], dtype=bool)
    min_counts = config.get('min_counts_after_subsampling', DEFAULT_MIN_COUNTS_AFTER_SUBSAMPLING)
    while True:
        X = counts[cells][:, genes]
        keep = np.ones(X.shape[1], dtype=bool)
        for cl in np.unique(clusters[cells]):
            keep &= np.asarray((X[clusters[cells] == cl] > 0).sum(0)).ravel() >= config.get('min_cells_per_gene', 0)
        new_genes = genes.copy()
        new_genes[genes] = keep
        new_cells = cells.copy()
        new_cells[cells] = np.asarray(counts[cells][:, new_genes].sum(1)).ravel() >= min_counts
        if (new_genes == genes).all() and (new_cells == cells).all():
            return cells, genes
        cells, genes = new_cells, new_genes


def pools(counts, clusters, size=POOL, seed=SEED):
    """Each cluster's cells shuffled and cut into pools of size cells, summed."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cl in sorted(np.unique(clusters)):
        idx = rng.permutation(np.where(clusters == cl)[0])
        for chunk in np.array_split(idx, int(np.ceil(len(idx) / size))):
            rows.append(np.asarray(counts[chunk].sum(0)).ravel())
            labels.append(cl)
    return np.vstack(rows).astype(int), np.array(labels)


def reference(pooled, labels, genes, n_jobs):
    """PyDESeq2's Wald test of CONTRAST, BH over every gene, and each gene's reference marker label."""
    inference = DefaultInference(n_cpus=n_jobs)
    names = [f'pool{i}' for i in range(len(labels))]
    dds = DeseqDataSet(counts=pd.DataFrame(pooled, index=names, columns=genes),
                       metadata=pd.DataFrame({'Cluster': labels}, index=names),
                       design='~Cluster', inference=inference, quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=['Cluster', *CONTRAST], alpha=ALPHA, independent_filter=False,
                       inference=inference, quiet=True)
    stats.summary()
    res = stats.results_df.rename_axis('gene').reset_index()
    sig = res['padj'] < ALPHA
    res['marker_of'] = np.where(sig & (res['log2FoldChange'] > 0), CONTRAST[0],
                                np.where(sig & (res['log2FoldChange'] < 0), CONTRAST[1], ''))
    return res


def main():
    parser = argparse.ArgumentParser(description="The lymph node's pseudo-bulk reference markers")
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-jobs', type=int, default=cpu_count())
    args = parser.parse_args()
    config = load_config(args.config)
    adata = sc.read_h5ad(require_input(config['adata_path'], what='the lymph node spot matrix',
                                       source='bash lymphnode/reproduce_vishd_cluster1_cluster3.sh'))
    counts, clusters, n_spots = cell_counts(adata, config['cell_id_column'], config['cluster_column'])
    cells, genes = filtered(counts, clusters, n_spots, config)
    pooled, labels = pools(counts[cells][:, genes], clusters[cells])
    res = reference(pooled, labels, np.asarray(adata.var_names)[genes], args.n_jobs)
    out = config['reference']
    res.to_csv(out, index=False)
    n_pools = pd.Series(labels).value_counts().sort_index().to_dict()
    print(f'{cells.sum()} of {len(cells)} cells, {genes.sum()} genes, pools {n_pools}; '
          f'markers {res["marker_of"].replace("", "none").value_counts().to_dict()}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
