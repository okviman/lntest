"""MAST on a null in which only some genes change: each test's FPR on genes of equal mean and unequal variance.

Each condition: 1,000 + 1,000 cells, 1,500 genes of mean μ; X's genes all of variance Var(X); in Y, the first SHIFTED
genes of variance ratio × Var(X) and the rest of Var(X), so the two groups' per-cell detection rates barely differ (when
every gene changes, as in Fig 1c's null, MAST's detection-rate covariate absorbs the group effect). μ = 5 and 50,
Var(X) = μ and 4μ, Var(Y)/Var(X) = 1, 2, 4, 8, 16 and 32, seed 1. All four tests on the same draws: LN's t-test, the
log1p t-test and the Wilcoxon as synthetic_nb.null runs them, MAST as Seurat's FindMarkers runs it (de_utils.run_mast_de
with cdr=False: the hurdle LRT on log1p CP10K, ~ group); Bonferroni over every gene. The FPR over the shifted genes,
and over the others as a check.

    python -m synthetic_nb.null_mast              # from reproducibility/; about 7 minutes; needs R_HOME
    python -m synthetic_nb.null_mast --plot-only  # redraw from the CSV
"""

import argparse

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from de_utils import run_mast_de
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir
from synthetic_nb import null

MUS = (5, 50)
SHIFTED = 150                    # the genes of Y with the larger variance; the other N_GENES - SHIFTED unchanged
SEED = 1
FANOS = (1, 4)                   # Var(X) as multiples of μ
RATIOS = (1, 2, 4, 8, 16, 32)    # Var(Y) / Var(X)
OUT = results_dir(__file__)
CSV = OUT / 'null_fpr_mast.csv'
STYLES = {1: '-', 4: (0, (3, 2))}   # Var(X) = μ solid, 4μ short dashes, as Fig 1c's


def mast_pvals(X, Y):
    """MAST's Bonferroni-adjusted p-values, in gene order."""
    counts = np.vstack([X, Y]).astype(float)
    cells = [f'c{i}' for i in range(len(counts))]
    a = ad.AnnData(X=counts, obs=pd.DataFrame({'group': ['X'] * len(X) + ['Y'] * len(Y)}, index=cells),
                   var=pd.DataFrame(index=[f'g{j}' for j in range(counts.shape[1])]))
    a.layers['log1p_norm'] = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    return np.asarray(run_mast_de(a, 'group', 'Y', 'X', cdr=False)['Y_vs_X']['pvals_adj'], dtype=float)


def condition(mu, fano, ratio):
    """One condition's X and Y: every gene of mean mu; X's of variance fano * mu; Y's first SHIFTED genes of ratio times
    that, the rest of it."""
    d1 = (fano * mu - mu) / mu ** 2
    d2 = (ratio * fano * mu - mu) / mu ** 2
    X = null.draw_nb(mu, d1, (null.NX, null.N_GENES))
    Y = np.hstack([null.draw_nb(mu, d2, (null.NY, SHIFTED)), null.draw_nb(mu, d1, (null.NY, null.N_GENES - SHIFTED))])
    return X, Y


def run():
    np.random.seed(SEED)
    rows = []
    shifted = np.arange(null.N_GENES) < SHIFTED
    for mu in MUS:
        for fano in FANOS:
            for ratio in RATIOS:
                X, Y = condition(mu, fano, ratio)
                for test in (*null.TESTS['ratio'], 'MAST'):
                    p = mast_pvals(X, Y) if test == 'MAST' else np.asarray(null.adjusted_pvalues(X, Y, test), float)
                    called = np.nan_to_num(p, nan=1.0) < 0.05
                    rows.append({'base_mu': mu, 'var_x': fano * mu, 'var_y': ratio * fano * mu, 'fano': fano,
                                 'ratio': ratio, 'method': test, 'value': called[shifted].mean(),
                                 'value_unshifted': called[~shifted].mean()})
                print(f"  μ {mu}, Var(X) {fano}μ, ratio {ratio}: " + ', '.join(
                    f"{r['method']} {r['value']:.3f}" for r in rows[-4:]), flush=True)
    return pd.DataFrame(rows)


def draw_fpr(ax, table, mu):
    """Each test's FPR against Var(Y)/Var(X) at one μ, Var(X) by line style, on a log2 axis."""
    d = table[table['base_mu'] == mu]
    for m in method_draw_order(d['method']):
        for fano in FANOS:
            e = d[(d['method'] == m) & (d['fano'] == fano)].sort_values('ratio')
            ax.plot(e['ratio'], e['value'], color=method_color(m), ls=STYLES[fano], lw=1.3, marker='o', ms=2.5,
                    label=method_label(m) if fano == FANOS[0] else None)
    ax.set_xscale('log', base=2)
    ax.set_xticks(RATIOS, [str(r) for r in RATIOS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r'Var$(Y)$ / Var$(X)$')
    ax.set_ylabel('FPR')
    ax.set_title(rf'$\mu$ = {mu}')


def plot(table):
    plot_style.use()
    fig, axes = plt.subplots(1, len(MUS), figsize=plot_style.figsize(len(MUS)), sharey=True)
    for ax, mu in zip(axes, MUS):
        draw_fpr(ax, table, mu)
    fig.tight_layout()
    plot_style.legend_outside(fig)
    out = OUT / fig_name('null_fpr_mast')
    fig.savefig(out, dpi=plot_style.RASTER_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    parser = argparse.ArgumentParser(description="MAST on the synthetic null's conditions")
    parser.add_argument('--plot-only', action='store_true', help='Redraw from the CSV')
    args = parser.parse_args()
    if args.plot_only:
        mast = pd.read_csv(require_input(CSV, what="MAST's null FPR", source='this script without --plot-only'))
    else:
        mast = run()
        mast.to_csv(CSV, index=False)
        print(f'  wrote {CSV}')
    plot(mast)


if __name__ == '__main__':
    main()
