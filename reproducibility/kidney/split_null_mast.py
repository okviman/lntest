"""MAST on the kidney spot-splitting null without DEGs, the experiment of Fig 1f, on the first SPLITS splits of each p_split.

Each split is rebuilt from its seed in split_null_per_split.csv by kidney.spot_split.create_groups_from_shape_ids,
after the capsule filter of spot_split (the N_REMOVE capsules with the lowest counts dropped). Genes without counts in
either half are dropped, as in spot_split.fpr_test_single. MAST runs through Seurat's FindMarkers with its gene filters
off (de_utils.run_mast_two_groups: the hurdle LRT on log1p CP10K, ~ group), with Bonferroni. The FPR is the number of
genes called over the number tested.
The results go to split_null_mast_per_split.csv in the per-split columns of spot_split, batch by batch. A rerun keeps
the splits already there. The uncorrected p-values give the share of genes below each of kidney.null_calibration.ALPHAS,
written to split_null_mast_calibration.csv in the columns of null_calibration, whose replicates are the same splits.
The figures show the panel of Fig 1f and the calibration for all four tests over the same splits. The other three
tests come from the per-split CSV of spot_split and the CSV of null_calibration.

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
CALIBRATION = OUT / 'split_null_mast_calibration.csv'
COLUMNS = ['p', 'method', 'split', 'seed', 'n_genes', 'n_degs', 'fpr']
CAL_COLUMNS = ['experiment', 'p', 'replicate', 'method', 'n_genes', 'alpha', 'share']   # null_calibration's


def capsules(input_file=data_dir(__file__) / 'podocytes_2um.h5ad', n_remove=N_REMOVE):
    """The spot counts and each capsule's spot indices, after spot_split's filter, as its main builds them."""
    import scanpy as sc
    import scipy.sparse as sp
    h5ad = sc.read_h5ad(require_input(input_file, what='the per-2um-spot kidney matrix',
                                      source='python -m kidney.preprocess'))
    umis = sp.csr_matrix(h5ad.X)
    shape_ids = h5ad.obs['shape_id'].values
    totals = {s: umis[shape_ids == s, :].sum() for s in np.unique(shape_ids)}
    removed = {s for s, _ in sorted(totals.items(), key=lambda x: x[1])[:n_remove]}
    keep = np.array([s not in removed for s in shape_ids])
    umis, shape_ids = umis[keep, :].copy(), shape_ids[keep]
    unique = np.unique(shape_ids)
    return umis, {s: np.where(shape_ids == s)[0] for s in unique}, unique


def mast_split(task):
    """One split's MAST result, as a per-split row, and its uncorrected p-values."""
    from de_utils import run_mast_two_groups
    p, split, seed, X, Y = task
    res = run_mast_two_groups(X, Y)
    padj = np.nan_to_num(np.asarray(res['pvals_adj'], dtype=float), nan=1.0)
    called = int((padj < 0.05).sum())
    n_genes = X.shape[1]
    print(f'  p_split {p:.3f}, split {split}: {called} of {n_genes} genes', flush=True)
    row = dict(zip(COLUMNS, (p, 'MAST', split, seed, n_genes, called, called / n_genes)))
    return row, np.asarray(res['pvals'], dtype=float)


def calibration_rows(row, pvals):
    """One split's share of genes with uncorrected p below each alpha, as null_calibration.shares counts it (a NaN p is
    never below), in its columns."""
    from kidney.null_calibration import ALPHAS
    return [dict(zip(CAL_COLUMNS, ('spot splitting', round(float(row['p']), 4), int(row['split']), 'MAST',
                                   row['n_genes'], a, float(np.mean(pvals < a))))) for a in ALPHAS]


def split_counts(caps, p, seed):
    """One saved split's two halves, the genes unexpressed in either dropped."""
    from kidney.spot_split import create_groups_from_shape_ids
    X, Y = create_groups_from_shape_ids(*caps, p, seed=seed)
    expressed = (X.sum(0) > 0) & (Y.sum(0) > 0)
    return X[:, expressed], Y[:, expressed]


def run(saved):
    """The splits built here, WORKERS at a time, so that only their counts reach the workers; each batch written to CSV
    and CALIBRATION. A split is done once it is in both; one in CSV alone is fitted again, and its saved row kept if the
    new fit calls the same number of genes."""
    done = pd.read_csv(CSV, float_precision="round_trip") if CSV.exists() else pd.DataFrame(columns=COLUMNS)
    cal = (pd.read_csv(CALIBRATION, float_precision="round_trip") if CALIBRATION.exists()
           else pd.DataFrame(columns=CAL_COLUMNS))
    have = set(zip(done['p'].astype(float).round(4), done['split'])) & set(zip(cal['p'], cal['replicate']))
    todo = saved[(saved['method'] == 'LN_test') & (saved['split'] < SPLITS)]
    tasks = [(r.p, int(r.split), int(r.seed)) for r in todo.itertuples() if (round(r.p, 4), r.split) not in have]
    caps = capsules() if tasks else None
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        for start in range(0, len(tasks), WORKERS):
            results = pool.map(mast_split, [(p, split, seed, *split_counts(caps, p, seed))
                                            for p, split, seed in tasks[start:start + WORKERS]])
            for row, pvals in results:
                old = done[np.isclose(done['p'].astype(float), row['p']) & (done['split'] == row['split'])]
                if len(old) and int(old['n_degs'].iloc[0]) != row['n_degs']:
                    print(f"  p_split {row['p']:.3f}, split {row['split']}: {row['n_degs']} genes called, "
                          f"{int(old['n_degs'].iloc[0])} saved; the new row replaces it", flush=True)
                    done = done.drop(old.index)
                    old = old.iloc[:0]
                if not len(old):
                    done = pd.concat([done, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
                cal = pd.concat([cal, pd.DataFrame(calibration_rows(row, pvals), columns=CAL_COLUMNS)],
                                ignore_index=True)
            done.sort_values(['p', 'split'], ignore_index=True).to_csv(CSV, index=False)
            cal.sort_values(['p', 'replicate', 'alpha'], ignore_index=True).to_csv(CALIBRATION, index=False)
    keep = set(zip(todo['p'], todo['split']))
    done = done[[k in keep for k in zip(done['p'], done['split'])]]
    return done.sort_values(['p', 'split'], ignore_index=True), cal.sort_values(['p', 'replicate', 'alpha'],
                                                                                ignore_index=True)


def table(saved, mast):
    """Each test's FPR mean and SD per p_split over MAST's splits, as spot_split summarises them."""
    pairs = mast[['p', 'split']]
    d = pd.concat([saved[COLUMNS].merge(pairs, on=['p', 'split']), mast])
    return d.groupby(['p', 'method'], sort=False)['fpr'].agg(fpr_mean='mean', fpr_std=lambda v: np.std(v)).reset_index()


def calibration_table(calibration, mast_cal):
    """null_calibration's spot-splitting rows over MAST's splits, with MAST's, for its draw_calibration."""
    d = calibration[calibration['experiment'] == 'spot splitting']
    pairs = mast_cal[['p', 'replicate']].drop_duplicates()
    return pd.concat([d.merge(pairs, on=['p', 'replicate']), mast_cal], ignore_index=True)


def plot(df, cal):
    import kidney.null_calibration as nc
    plot_style.use()
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_fpr_log(ax, df, SPLIT)
    plot_style.legend_outside(fig)
    out = OUT / fig_name('split_null_mast_fpr')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')
    nc.plot(cal, 'spot splitting', OUT / fig_name('split_null_mast_calibration'))


def main():
    parser = argparse.ArgumentParser(description="MAST on the kidney's no-DEG spot-splitting null")
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSVs')
    args = parser.parse_args()
    saved = pd.read_csv(require_input(PER_SPLIT, what='the no-DEG spot-splitting splits',
                                      source='python -m kidney.spot_split (see README)'))
    if args.plot_only:
        mast = pd.read_csv(require_input(CSV, what="MAST's spot-splitting FPR", source='this script without --plot-only'))
        mast_cal = pd.read_csv(require_input(CALIBRATION, what="MAST's spot-splitting calibration",
                                             source='this script without --plot-only'))
    else:
        mast, mast_cal = run(saved)
        mast.to_csv(CSV, index=False)
        mast_cal.to_csv(CALIBRATION, index=False)
        print(f'  wrote {CSV} and {CALIBRATION}')
    import kidney.null_calibration as nc
    calibration = pd.read_csv(require_input(OUT / nc.CSV, what='the kidney null calibration',
                                            source='python -m kidney.null_calibration'))
    plot(table(saved, mast), calibration_table(calibration, mast_cal))


if __name__ == '__main__':
    main()
