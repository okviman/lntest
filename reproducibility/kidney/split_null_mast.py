"""MAST on the kidney's no-DEG spot-splitting null (Fig 1f), on the first SPLITS saved splits of each p_split.

Each split is rebuilt from its seed in split_null_per_split.csv by kidney.spot_split.create_groups_from_shape_ids, after
spot_split's capsule filter (N_REMOVE lowest-count capsules dropped, as the README's run). Genes unexpressed in either
half are dropped as spot_split.fpr_test_single drops them; MAST runs as Seurat's FindMarkers runs it (de_utils.run_mast_de
with cdr=False: the hurdle LRT on log1p CP10K, ~ group), Bonferroni; FPR = genes called over genes tested. Written to
split_null_mast_per_split.csv in spot_split's per-split columns, batch by batch; a rerun keeps the splits already there.
The figure draws Fig 1f's panel for all four tests over the same splits, the other three from spot_split's per-split CSV.

    python -m kidney.split_null_mast              # from reproducibility/; about 25 minutes; needs R_HOME; resumes
    python -m kidney.split_null_mast --plot-only  # redraw from the CSVs
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # before numpy is imported; the workers inherit it

import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from kidney.fpr_plots import SPLIT, draw_fpr_log
from paths import data_dir, fig_name, require_input, results_dir

SPLITS = 3       # the first splits of each p_split
N_REMOVE = 35    # spot_split's --n_shape_ids_remove
WORKERS = 4      # MAST fits at a time, one thread each
OUT = results_dir(__file__)
PER_SPLIT = OUT / 'split_null_per_split.csv'
CSV = OUT / 'split_null_mast_per_split.csv'
COLUMNS = ['p', 'method', 'split', 'seed', 'n_genes', 'n_degs', 'fpr']


def capsules(input_file=data_dir(__file__) / 'podocytes_2um.h5ad', n_remove=N_REMOVE):
    """The spot counts and each capsule's spot indices, after spot_split's filter, as its main builds them."""
    import scanpy as sc
    import scipy.sparse as sp
    h5ad = sc.read_h5ad(require_input(input_file, what='the per-2um-spot kidney matrix',
                                      source='kidney/preprocess.ipynb (see README)'))
    umis = sp.csr_matrix(h5ad.X)
    shape_ids = h5ad.obs['shape_id'].values
    totals = {s: umis[shape_ids == s, :].sum() for s in np.unique(shape_ids)}
    removed = {s for s, _ in sorted(totals.items(), key=lambda x: x[1])[:n_remove]}
    keep = np.array([s not in removed for s in shape_ids])
    umis, shape_ids = umis[keep, :].copy(), shape_ids[keep]
    unique = np.unique(shape_ids)
    return umis, {s: np.where(shape_ids == s)[0] for s in unique}, unique


def mast_split(task):
    """One split's MAST result, as a per-split row."""
    import anndata as ad
    from de_utils import run_mast_de
    p, split, seed, X, Y = task
    counts = np.vstack([X, Y]).astype(float)
    a = ad.AnnData(X=counts, obs=pd.DataFrame({'group': ['X'] * len(X) + ['Y'] * len(Y)},
                                              index=[f'c{i}' for i in range(len(counts))]),
                   var=pd.DataFrame(index=[f'g{j}' for j in range(counts.shape[1])]))
    a.layers['log1p_norm'] = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    padj = run_mast_de(a, 'group', 'Y', 'X', cdr=False)['Y_vs_X']['pvals_adj']
    padj = np.nan_to_num(np.asarray(padj, dtype=float), nan=1.0)
    called = int((padj < 0.05).sum())
    print(f'  p_split {p:.3f}, split {split}: {called} of {counts.shape[1]} genes', flush=True)
    return dict(zip(COLUMNS, (p, 'MAST', split, seed, counts.shape[1], called, called / counts.shape[1])))


def split_counts(caps, p, seed):
    """One saved split's two halves, the genes unexpressed in either dropped."""
    from kidney.spot_split import create_groups_from_shape_ids
    X, Y = create_groups_from_shape_ids(*caps, p, seed=seed)
    expressed = (X.sum(0) > 0) & (Y.sum(0) > 0)
    return X[:, expressed], Y[:, expressed]


def run(saved):
    """The splits built here, WORKERS at a time, so that only their counts reach the workers; each batch appended to CSV,
    and splits already in it kept."""
    done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(zip(done['p'], done['split']))
    todo = saved[(saved['method'] == 'LN_test') & (saved['split'] < SPLITS)]
    tasks = [(r.p, int(r.split), int(r.seed)) for r in todo.itertuples() if (r.p, r.split) not in have]
    caps = capsules() if tasks else None
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        for start in range(0, len(tasks), WORKERS):
            rows = pool.map(mast_split, [(p, split, seed, *split_counts(caps, p, seed))
                                         for p, split, seed in tasks[start:start + WORKERS]])
            done = pd.concat([done, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
            done.to_csv(CSV, index=False)
    keep = set(zip(todo['p'], todo['split']))
    done = done[[k in keep for k in zip(done['p'], done['split'])]]
    return done.sort_values(['p', 'split'], ignore_index=True)


def table(saved, mast):
    """Each test's FPR mean and SD per p_split over MAST's splits, as spot_split summarises them."""
    pairs = mast[['p', 'split']]
    d = pd.concat([saved[COLUMNS].merge(pairs, on=['p', 'split']), mast])
    return d.groupby(['p', 'method'], sort=False)['fpr'].agg(fpr_mean='mean', fpr_std=lambda v: np.std(v)).reset_index()


def plot(df):
    plot_style.use()
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_fpr_log(ax, df, SPLIT)
    plot_style.legend_outside(fig)
    out = OUT / fig_name('split_null_mast_fpr')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    parser = argparse.ArgumentParser(description="MAST on the kidney's no-DEG spot-splitting null")
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSVs')
    args = parser.parse_args()
    saved = pd.read_csv(require_input(PER_SPLIT, what='the no-DEG spot-splitting splits',
                                      source='python -m kidney.spot_split (see README)'))
    if args.plot_only:
        mast = pd.read_csv(require_input(CSV, what="MAST's spot-splitting FPR", source='this script without --plot-only'))
    else:
        mast = run(saved)
        mast.to_csv(CSV, index=False)
        print(f'  wrote {CSV}')
    plot(table(saved, mast))


if __name__ == '__main__':
    main()
