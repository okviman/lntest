"""How each test's runtime scales with the number of cells, on synthetic NB counts.

Both groups drawn alike, 2,000 genes of mean 5 and variance 2μ (synthetic_nb.null.draw_nb), seed 0; total cells CELLS, half
per group. Each test timed from raw counts to Bonferroni-adjusted p-values as synthetic_nb.null runs it: LN's t-test
(lntest.get_LN_lfcs), the log1p t-test and the Wilcoxon (baselines.scanpy_sig_test); MAST (de_utils.run_mast_de, Y against
X, as Seurat's FindMarkers runs it) up to MAST_MAX cells. The median wall time of REPEATS runs, one thread for the
numerical libraries. Times depend on the machine, which is recorded with them; they are not byte-reproducible.

With --sparse, realistic sparse counts instead: gene means log-normal (SPARSE_MEANS), NB with Var = μ + SPARSE_PHI μ², stored
sparse, and each test as a user runs it on a sparse AnnData of counts: LN by lntest.rank_genes_groups_ln on the counts,
the log1p t-test and the Wilcoxon by scanpy's normalize_total, log1p and rank_genes_groups (normalisation timed with
them), Bonferroni, Y against X; MAST as above. Written to runtime_sparse.csv with the realised share of zeros.

    python -m synthetic_nb.runtime              # from reproducibility/; about 8 minutes; needs R_HOME for MAST
    python -m synthetic_nb.runtime --sparse     # the sparse version; about 5 minutes
    python -m synthetic_nb.runtime --plot-only  # redraw from the CSV (with --sparse, the sparse one's)
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # before numpy is imported

import argparse
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir
from synthetic_nb import null

GENES, MU, FANO = 2000, 5, 2
CELLS = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
MAST_MAX = 10_000
REPEATS = 3
SPARSE_MEANS = (np.log(0.15), 1.3)   # the log-normal's mean and SD of log gene means
SPARSE_PHI = 0.5
OUT = results_dir(__file__)
CSV = OUT / 'runtime.csv'
CSVS = {False: CSV, True: OUT / 'runtime_sparse.csv'}


def timed(test, X, Y):
    """One run of a test, in seconds."""
    t = time.perf_counter()
    if test == 'MAST':
        import anndata as ad
        from de_utils import run_mast_de
        counts = np.vstack([X, Y]).astype(float)
        cells = [f'c{i}' for i in range(len(counts))]
        a = ad.AnnData(X=counts, obs=pd.DataFrame({'group': ['X'] * len(X) + ['Y'] * len(Y)}, index=cells),
                       var=pd.DataFrame(index=[f'g{j}' for j in range(counts.shape[1])]))
        a.layers['log1p_norm'] = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        run_mast_de(a, 'group', 'Y', 'X', cdr=False)
    else:
        null.adjusted_pvalues(X, Y, test)
    return time.perf_counter() - t


def sparse_counts(n, means, rng):
    """n cells of NB counts with the given gene means and Var = μ + SPARSE_PHI μ², as a CSR matrix, drawn in chunks."""
    import scipy.sparse as sp
    r = 1 / SPARSE_PHI
    chunks = []
    for start in range(0, n, 5_000):
        m = min(5_000, n - start)
        chunks.append(sp.csr_matrix(rng.negative_binomial(r, r / (r + means), size=(m, len(means))).astype(np.float32)))
    return sp.vstack(chunks).tocsr()


def timed_sparse(test, counts, labels):
    """One run of a test as a user runs it on a sparse AnnData of counts, in seconds."""
    import anndata as ad
    import scanpy as sc
    from lntest import rank_genes_groups_ln
    a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({'group': pd.Categorical(labels)},
                                                     index=[f'c{i}' for i in range(counts.shape[0])]))
    a.var_names = [f'g{j}' for j in range(counts.shape[1])]
    t = time.perf_counter()
    if test == "LN's t-test":
        rank_genes_groups_ln(a, 'group', groups=['Y'], reference='X', corr_method='bonferroni')
    elif test == 'MAST':
        dense = counts.toarray()
        return timed(test, dense[labels == 'X'], dense[labels == 'Y'])
    else:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.tl.rank_genes_groups(a, 'group', groups=['Y'], reference='X', corr_method='bonferroni',
                                method='t-test' if test == 'log1p t-test' else 'wilcoxon', tie_correct=True)
    return time.perf_counter() - t


def run_sparse():
    rng = np.random.default_rng(0)
    means = np.exp(rng.normal(*SPARSE_MEANS, size=GENES))
    rows = []
    for n in CELLS:
        counts = sparse_counts(n, means, rng)
        labels = np.array(['X'] * (n // 2) + ['Y'] * (n - n // 2))
        zeros = 1 - counts.nnz / (counts.shape[0] * counts.shape[1])
        for test in (*null.TESTS['ratio'], 'MAST'):
            if test == 'MAST' and n > MAST_MAX:
                continue
            times = [timed_sparse(test, counts, labels) for _ in range(REPEATS)]
            rows.append({'cells': n, 'genes': GENES, 'zeros': zeros, 'method': test, 'seconds': float(np.median(times)),
                         'min': min(times), 'max': max(times), 'machine': f'{platform.machine()} {platform.platform()}'})
            print(f'  {n} cells ({zeros:.0%} zeros), {test}: {rows[-1]["seconds"]:.2f} s', flush=True)
    return pd.DataFrame(rows)


def run():
    np.random.seed(0)
    d = (FANO * MU - MU) / MU ** 2
    rows = []
    for n in CELLS:
        X = null.draw_nb(MU, d, (n // 2, GENES))
        Y = null.draw_nb(MU, d, (n // 2, GENES))
        for test in (*null.TESTS['ratio'], 'MAST'):
            if test == 'MAST' and n > MAST_MAX:
                continue
            times = [timed(test, X, Y) for _ in range(REPEATS)]
            rows.append({'cells': n, 'genes': GENES, 'method': test, 'seconds': float(np.median(times)),
                         'min': min(times), 'max': max(times), 'machine': f'{platform.machine()} {platform.platform()}'})
            print(f'  {n} cells, {test}: {rows[-1]["seconds"]:.2f} s', flush=True)
    return pd.DataFrame(rows)


def draw_runtime(ax, df):
    """Each test's median time against the number of cells, both axes logarithmic."""
    for m in method_draw_order(df['method']):
        e = df[df['method'] == m].sort_values('cells')
        ax.plot(e['cells'], e['seconds'], color=method_color(m), marker='o', ms=3, lw=1.3, label=method_label(m))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'Cells ({GENES:,} genes)')
    ax.set_ylabel('Time (s)')


def plot(df, sparse=False):
    plot_style.use()
    fig, ax = plt.subplots(figsize=plot_style.PANEL)
    draw_runtime(ax, df)
    plot_style.legend_outside(fig)
    out = OUT / fig_name('runtime_scaling_sparse' if sparse else 'runtime_scaling')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    parser = argparse.ArgumentParser(description="Each test's runtime against the number of cells")
    parser.add_argument('--sparse', action='store_true', help='Realistic sparse counts, as users run each test')
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSV')
    args = parser.parse_args()
    csv = CSVS[args.sparse]
    if args.plot_only:
        df = pd.read_csv(require_input(csv, what='the runtimes', source='this script without --plot-only'))
    else:
        df = run_sparse() if args.sparse else run()
        df.to_csv(csv, index=False)
        print(f'  wrote {csv}')
    plot(df, args.sparse)


if __name__ == '__main__':
    main()
