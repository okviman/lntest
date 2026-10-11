"""Extended Data Fig 2: MAST on the synthetic null, and each test's runtime, assembled from the arm's drawing code.

a, b: at μ = 5 and 50, the FPR at Bonferroni 0.05 of LN's t-test, the log1p t-test, the Wilcoxon and MAST against
Var(Y)/Var(X), Var(X) = μ (solid) and 4μ (short dashes), over the 10% of genes whose variance differs while the
rest are unchanged (synthetic_nb.null_mast); c: each test's
median wall time against the number of cells, 2,000 genes (synthetic_nb.runtime). One key. Each plot is drawn by its own
drawing code from the saved results, at the size it prints at; a change to a plot belongs in that code.

    python -m figures.ed2      # from reproducibility/, seconds; needs null_mast's and runtime's CSVs
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_style
from figures.layout import KEY_W, PLOT, assemble, axes_at, save, use
from figures.layout import ED_WIDTH as WIDTH
from method_colors import method_color, method_label, method_order
from paths import require_input
from synthetic_nb import null_mast as nm
from synthetic_nb import runtime as rt

ROW, TOP = 2.1, 0.25
SLOT = (WIDTH - KEY_W) / 3


def panel(draw, title, stem):
    """One plot in a third of the page less the key, titled."""
    fig = plt.figure(figsize=(SLOT, ROW))
    ax = axes_at(fig, PLOT['left'], PLOT['bottom'], SLOT - PLOT['left'] - PLOT['right'], ROW - PLOT['bottom'] - TOP)
    draw(ax)
    ax.set_title(title)
    if ax.get_legend():
        ax.get_legend().remove()
    return save(fig, stem)


def key(stem):
    """The method colours and Fig 1c's Var(X) line styles."""
    fig = plt.figure(figsize=(KEY_W, ROW))
    methods = method_order(["LN's t-test", 'log1p t-test', 'Wilcoxon', 'MAST'])
    handles = [Line2D([], [], color=method_color(m), lw=1.3, marker='o', ms=3) for m in methods]
    labels = [method_label(m) for m in methods]
    handles += [Line2D([], [], color='black', lw=1.3, ls=nm.STYLES[f]) for f in nm.FANOS]
    labels += [r'Var$(X)$ = $\mu$', r'Var$(X)$ = 4$\mu$']
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 1 - TOP / ROW), frameon=False,
               fontsize=plot_style.SMALL, handlelength=1.8)
    return save(fig, stem)


if __name__ == '__main__':
    use()
    mast = pd.read_csv(require_input(nm.CSV, what="MAST's null FPR", source='python -m synthetic_nb.null_mast'))
    table = mast
    times = pd.read_csv(require_input(rt.CSVS[True], what='the sparse runtimes', source='python -m synthetic_nb.runtime --sparse'))
    a, b = (panel(lambda ax, mu=mu: nm.draw_fpr(ax, table, mu), rf'Null, $\mu$ = {mu}', f'ed2_{l}_null_mast_mu{mu}')
            for l, mu in (('a', 5), ('b', 50)))
    c = panel(lambda ax: rt.draw_runtime(ax, times), 'Runtime', 'ed2_c_runtime')
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0), ('', key('ed2_key'), 0)]], 'ed2', width=WIDTH)
