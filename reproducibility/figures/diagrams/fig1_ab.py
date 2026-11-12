"""Fig 1a,b at print size: full width (7.2 in), about a third of a page.

a: UMI counts -> clusters -> a DE test of each cluster against the rest -> markers per method, each gene with the
   ranking score of the test (for LN also its detection part, boxed as in b), looping back.
b: the steps of LN's t-test and of the log1p t-test for one non-DE gene g.
The numbers come from numbers(): mu = 5, Var(X) = 10, Var(Y) = 50, n = 1000 per group, seed 0. Gene names are
placeholders. Text is 5.5-7 pt at print size.

Output: figures/diagrams/fig1a.pdf and fig1b.pdf, placed on the page by figures.fig1. PNG previews of both panels go
to figures/results/panels/. The PDFs are committed, so assembling Fig 1 needs neither LaTeX (with dvipng) nor pdftoppm.

    python -m figures.diagrams.fig1_ab      # from reproducibility/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox, TransformedBbox
from scipy import stats

from lntest._ln_test import get_LN_lfcs

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent / 'results' / 'panels'
MU, VX, VY, N, SEED = 5.0, 10.0, 50.0, 1000, 0
TEXT, SMALL, TINY = 7, 6, 5.5
C = {'X': '#1f77b4', 'Y': '#ff7f0e'}
REST = ['#ff7f0e', '#ffa75a', '#ffcb99']
LN, LOG1P, WILC = '#2ca02c', '#8c564b', '#17becf'
DET = '#98df8a'      # LN's detection part: a dashed light green box in a and in b
AXIS, ARROW = '0.55', '0.3'
plt.rcParams.update({
    'font.size': SMALL, 'axes.labelsize': TINY, 'xtick.labelsize': TINY, 'ytick.labelsize': TINY,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5,
    'axes.edgecolor': '0.3', 'xtick.color': '0.3', 'ytick.color': '0.3',
    'xtick.major.size': 1.5, 'ytick.major.size': 1.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.pad': 1, 'axes.labelcolor': AXIS, 'axes.labelpad': 1,
})
W, H = 7.2, 3.6
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'   # for the math labels in b (usetex)
B_TOP = 2.0          # panel b occupies y < B_TOP, panel a y > B_TOP


def nb(mu, var):
    r = mu ** 2 / (var - mu)
    return stats.nbinom(r, r / (r + mu))


def numbers():
    X, Y = nb(MU, VX), nb(MU, VY)
    out = dict(X=X, Y=Y)
    rng = np.random.default_rng(SEED)
    x = X.rvs(N, random_state=rng).astype(float)
    y = Y.rvs(N, random_state=rng).astype(float)
    # X vs Y throughout: every quantity is X relative to Y (X minus Y on the log scale)
    lfc, p, se = get_LN_lfcs(x[:, None], y[:, None], normalize=False, return_standard_error=True)
    out['ln'] = (lfc[0], se[0], p[0])
    ly, lx = np.log1p(y), np.log1p(x)
    t = stats.ttest_ind(lx, ly, equal_var=False)
    out['log1p'] = ((lx.mean() - ly.mean()) / np.log(2),
                    np.sqrt(ly.var(ddof=1) / N + lx.var(ddof=1) / N) / np.log(2), t.pvalue)
    # Everything right of the input is computed from this one sample, so the panels agree:
    # the gap between the log1p means is the log1p estimate, and the two LN terms sum to LN's estimate.
    out['sample'] = {'X': x, 'Y': y}
    out['elog'] = {'X': lx.mean(), 'Y': ly.mean()}  # natural log, as scanpy's log1p
    out['det'] = np.log2(np.mean(x > 0) / np.mean(y > 0))
    out['pos'] = np.log2(x[x > 0].mean() / y[y > 0].mean())
    assert np.isclose(out['det'] + out['pos'], out['ln'][0]), (out['det'] + out['pos'], out['ln'][0])
    assert np.isclose((out['elog']['X'] - out['elog']['Y']) / np.log(2), out['log1p'][0])
    return out


def main():
    v = numbers()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    def inset(x, y, w, h):
        a = fig.add_axes([x / W, y / H, w / W, h / H])
        a.set_facecolor('none')
        return a

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=5,
                                     lw=0.7, color=ARROW, shrinkA=0, shrinkB=0))

    def rbox(x, y, w, h, fc, ec='none', lw=0.5, r=0.04, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                    fc=fc, ec=ec, lw=lw, **kw))

    def colored(parts, xy, align=(0.5, 0.5), fs=SMALL):
        pk = HPacker(children=[TextArea(t, textprops=dict(fontsize=fs, **p)) for t, p in parts],
                     align='baseline', pad=0, sep=0)
        ax.add_artist(AnnotationBbox(pk, xy, box_alignment=align, frameon=False))

    def mathrow(parts, fs=TINY, base=AXIS):
        """A line of mathtext whose pieces can differ in colour; parts: (tex, colour or None)."""
        # typeset by LaTeX with amssymb, so P and E are the paper's own blackboard-bold glyphs
        return HPacker(children=[TextArea(f'${t}$', textprops=dict(fontsize=fs, color=c or base, usetex=True))
                                 for t, c in parts],
                       align='baseline', pad=0, sep=0)

    def V(g):  # a random variable, in its group's colour
        return (g, C[g])

    CLIP = []

    def layered(build, colours, xy, xycoords='data', align=(0.5, 0.5), fs=TINY, strut=(0.3, 1.0)):
        """LaTeX math with parts in different colours: one layer per colour, identical geometry, since every
        other part is typeset as a \phantom of itself. build(c) returns the math for layer c."""
        for c in colours:
            # matplotlib sizes usetex text by its ink, so invisible rules at both ends give every layer the same box
            r = r'\rule[-%gem]{0.01pt}{%gem}' % (strut[0], strut[0] + strut[1])
            ta = TextArea(rf'${r}\hspace{{1.5pt}}{build(c)}\hspace{{1.5pt}}{r}$', textprops=dict(fontsize=fs, color=c, usetex=True))
            ab = AnnotationBbox(ta, xy, xycoords=xycoords, box_alignment=align, frameon=False)
            ax.add_artist(ab)
            CLIP.append((ta, ab))   # the rules are clipped away before saving

    def join(parts, c, base):
        # a phantom relation must stay a relation, so the spacing is identical in every layer
        ph = lambda t: (r'\mathrel{\phantom{%s}}' if t in ('>', r'\mid') else r'\phantom{%s}') % t
        return ''.join(t if (col or base) == c else ph(t) for t, col in parts)

    def mathlabel(parts, xy, xycoords, align, base=AXIS):
        cols = [base] + [col for _, col in parts if col and col != base]
        layered(lambda c: join(parts, c, base), list(dict.fromkeys(cols)), xy, xycoords, align)

    # ======================= a =======================
    split = {k: len(getattr(ax, k)) for k in ('patches', 'texts', 'lines', 'artists')}
    letters = [ax.text(0.0, H - 0.01, 'a', fontsize=TEXT + 1, fontweight='bold', va='top')]
    my, mh = B_TOP + 0.4, 0.86         # matrix bottom and height
    cy = my + mh / 2

    # UMI counts
    mx, mw = 0.41, 1.0
    n_cells, n_genes, g_col = 14, 10, 1
    rng = np.random.default_rng(1)
    base = rng.gamma(1.2, 2.5, n_genes)
    cm = rng.negative_binomial(1.0, 1.0 / (1.0 + base * rng.gamma(0.8, 1.25, (n_cells, n_genes))))
    cw, ch = mw / n_genes, mh / n_cells
    vmax = np.quantile(cm, 0.95)
    for i in range(n_cells):
        for j in range(n_genes):
            ax.add_patch(Rectangle((mx + j * cw, my + mh - (i + 1) * ch), cw, ch,
                                   fc=plt.cm.viridis(min(cm[i, j] / vmax, 1.0)), ec='white', lw=0.2))
    ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle='round,pad=0,rounding_size=0.02', fc='none',
                                ec='0.5', lw=0.5))
    ax.add_patch(Rectangle((mx + g_col * cw, my), cw, mh, fc='none', ec='k', lw=0.8, zorder=5))
    ax.text(mx + (g_col + 0.5) * cw, my - 0.03, '$g$', ha='center', va='top', fontsize=TEXT)
    ax.text(mx + mw / 2 + 0.06, my - 0.05, 'Genes', ha='center', va='top', color=AXIS, fontsize=TINY)
    ax.text(mx - 0.04, cy, 'Cells', ha='right', va='center', color=AXIS, fontsize=TINY, rotation=90)
    ax.text(mx + mw / 2, my + mh + 0.13, 'UMI counts', ha='center', va='bottom', fontsize=TEXT)
    ax.text(mx + mw / 2, my + mh + 0.03, '(scRNA-seq, Visium HD, CITE-seq, ...)', ha='center', va='bottom',
            fontsize=TINY, color=AXIS, linespacing=1.05)

    # clusters
    ux, uw = 1.77, 0.76                  # square
    arrow(mx + mw + 0.05, cy, ux - 0.06, cy)
    a_u = inset(ux, my + 0.08, uw, mh - 0.1)
    rng = np.random.default_rng(3)
    for (a, b), col, n in zip([(0.0, 0.0), (1.6, 0.9), (1.5, -0.9), (3.0, 0.1)], [C['X']] + REST, (45, 40, 40, 40)):
        pts = rng.normal(0, 0.33, (n, 2)) + (a, b)
        a_u.scatter(pts[:, 0], pts[:, 1], s=0.9, color=col, lw=0)
    a_u.text(0.0, 0.6, 'A', color=C['X'], fontsize=SMALL, ha='center', va='bottom', fontweight='bold')
    a_u.text(3.0, -0.7, 'rest', color=C['Y'], fontsize=TINY, ha='center', va='top', fontweight='bold')
    a_u.set_xticks([]); a_u.set_yticks([])
    a_u.set_xlabel('UMAP', labelpad=1)
    ax.text(ux + uw / 2, my + mh + 0.13, 'Clusters', ha='center', va='bottom', fontsize=TEXT)

    # for each cluster: the test, and the markers each method returns
    fx0, fx1 = 2.88, 7.18
    fy0, fy1 = my - 0.18, H - 0.13
    rbox(fx0, fy0, fx1 - fx0, fy1 - fy0, fc='none', ec='0.4', lw=0.7, r=0.05, ls=(0, (3, 2)))
    ax.text((fx0 + fx1) / 2, fy1 + 0.02, 'For each cluster', ha='center', va='bottom', fontsize=SMALL, color=AXIS)
    arrow(ux + uw + 0.06, cy, fx0 - 0.04, cy)

    # the test, and below it as subtitles what it is used for; 'gene set enrichment analysis' wraps to fit the box
    b1x, b1w, bh, up = 2.98, 0.88, 0.69, 0.15
    rbox(b1x, cy - bh / 2, b1w, bh, fc='0.94', ec='0.6', r=0.05)
    ax.text(b1x + b1w / 2, cy - 0.035 + up, 'Differential\nexpression test', ha='center', va='bottom', fontsize=SMALL,
            linespacing=1.1)
    colored([('A', dict(color=C['X'], fontweight='bold')), (' vs ', {}),
             ('rest', dict(color=C['Y'], fontweight='bold')), ('?', {})], (b1x + b1w / 2, cy - 0.045 + up), (0.5, 1.0))
    ax.text(b1x + b1w / 2, cy - 0.065, 'marker selection\ngene set enrichment\nanalysis', ha='center', va='top',
            fontsize=TINY, color=AXIS, linespacing=1.1)

    methods = ((("LN's $t$-test"), LN, '$t$-test on the LFC', r'$\mathcal{O}(n)$ time'),
               ('log1p $t$-test', LOG1P, '$t$-test on log1p means', r'$\mathcal{O}(n)$ time'),
               ('Wilcoxon', WILC, 'rank test', r'$\mathcal{O}(n\,\log\,n)$ time'))
    # placeholders: each test's genes, ranked by its score, with its effect size (LN: the LFC; the log1p t-test: the
    # difference of mean log1p, in log2 units as b plots it; the Wilcoxon: the AUC, in the order of its z);
    # LN also shows each gene's detection part, log2 of its ratio of detection rates, highest for the true markers
    markers = {8, 3, 21, 30}   # the true markers, bold in every list
    ranked = [[6, 3, 8, 21, 12, 30], [5, 2, 3, 11, 8, 6, 17, 21], [8, 3, 2, 21, 9, 1, 12, 5]]
    scores = [['0.6', '3.1', '2.7', '2.2', '0.8', '2.5'],
              ['0.9', '0.8', '1.6', '0.7', '1.4', '0.4', '0.6', '1.2'],
              ['0.97', '0.96', '0.94', '0.93', '0.91', '0.90', '0.88', '0.87']]
    detection = ['0.2', '2.9', '2.6', '1.8', '0.5', '2.2']
    tx, col_ws, gap, row_h = 4.09, (1.17, 0.91, 0.91), 0.05, 0.079
    chip_w, num_w, sep = 0.46, 0.25, 0.05
    top = fy1 - 0.045
    arrow(b1x + b1w + 0.05, cy, tx - 0.03, cy)
    cx = tx
    for c, (name, colr, what, cost) in enumerate(methods):
        col_w = col_ws[c]
        ax.text(cx + col_w / 2, top, name, ha='center', va='top', fontsize=SMALL, color=colr)
        ax.plot([cx + 0.05, cx + col_w - 0.05], [top - 0.13] * 2, color=colr, lw=0.9)
        ax.text(cx + col_w / 2, top - 0.15, what, ha='center', va='top', fontsize=TINY, color=AXIS)
        ax.text(cx + col_w / 2, top - 0.235, cost, ha='center', va='top', fontsize=TINY, color=AXIS)
        g_top = top - 0.49
        cols = [scores[c]] + ([detection] if c == 0 else [])
        x0 = cx + (col_w - chip_w - len(cols) * (sep + num_w)) / 2   # the rows, centred in the column
        effect = ('LFC', '\u0394\u2009log1p', 'AUC')[c]   # each test's own effect size
        titles = ['DEG', effect] + (['marker\nscore'] if c == 0 else [])
        for j, t in enumerate(titles):
            xt = x0 + chip_w / 2 if j == 0 else x0 + chip_w + sep + (j - 1) * (sep + num_w) + num_w / 2
            ax.text(xt, g_top + 0.012, t, ha='center', va='bottom', fontsize=TINY, color=AXIS, linespacing=0.9)
        for r, k in enumerate(ranked[c]):
            yy = g_top - (r + 1) * row_h
            rbox(x0, yy + 0.008, chip_w, row_h - 0.016, fc=colr, r=0.025, alpha=0.18)
            ax.text(x0 + chip_w / 2, yy + row_h / 2 - 0.002, f'Gene {k}', ha='center', va='center', fontsize=TINY,
                    fontweight='bold' if k in markers else 'normal')
            for j, col in enumerate(cols):   # each number in a box like the gene's
                xn = x0 + chip_w + sep + j * (sep + num_w)
                rbox(xn, yy + 0.008, num_w, row_h - 0.016, fc=colr, r=0.025, alpha=0.18)
                bold = j == 1 and k in markers   # LN's detection part, bold for the markers
                ax.text(xn + num_w / 2, yy + row_h / 2 - 0.002, col[r], ha='center', va='center', fontsize=TINY,
                        fontweight='bold' if bold else 'normal')
        if c == 0:   # LN's detection part
            dx = x0 + chip_w + sep + (sep + num_w)
            rbox(dx - 0.02, g_top - len(ranked[0]) * row_h, num_w + 0.04, len(ranked[0]) * row_h,
                 fc='none', ec=DET, lw=0.8, r=0.03, ls=(0, (3, 2)))
        else:
            ax.text(x0 + chip_w / 2, g_top - (len(ranked[c]) + 1) * row_h + 0.045, r'$\vdots$', ha='center',
                    va='center', fontsize=TINY)
        cx += col_w + gap

    # loop: refine the clustering
    ly = my - 0.33
    lx = (fx0 + fx1) / 2
    ax.plot([lx, lx], [fy0, ly], color=ARROW, lw=0.7)
    ax.plot([lx, ux + uw / 2], [ly, ly], color=ARROW, lw=0.7)
    arrow(ux + uw / 2, ly, ux + uw / 2, my - 0.06)
    ax.text((ux + uw / 2 + lx) / 2, ly + 0.02, 'Optionally refine clusters and repeat', ha='center', va='bottom',
            fontsize=TINY, color=AXIS)
    n_axes_a = len(fig.axes)
    split_b = {k: len(getattr(ax, k)) for k in ('patches', 'texts', 'lines', 'artists')}

    # ======================= b (left) and space for c (right) =======================
    BW = 4.45                                   # panel b width; c takes the rest
    letters.append(ax.text(0.0, B_TOP - 0.02, 'b', fontsize=TEXT + 1, fontweight='bold', va='top'))
    c_placeholder = [ax.text(BW + 0.12, B_TOP - 0.02, 'c', fontsize=TEXT + 1, fontweight='bold', va='top'),
                     ax.add_patch(Rectangle((BW + 0.3, 0.1), W - BW - 0.35, B_TOP - 0.25, fc='none', ec='0.85',
                                            lw=0.5, ls=(0, (2, 2)))),
                     ax.text(BW + 0.3 + (W - BW - 0.35) / 2, B_TOP / 2, 'panel c', ha='center', va='center',
                             fontsize=SMALL, color='0.75')]

    # non-DE gene g
    ax.text(0.55, B_TOP - 0.1, 'Non-DE gene $g$', ha='center', va='top', fontsize=TEXT)
    ax.text(0.55, B_TOP - 0.26, 'Negative binomially\ndistributed counts', ha='center', va='top', fontsize=TINY,
            linespacing=1.05, color='0.2')
    a_in = inset(0.12, 0.3, 0.86, 1.0)
    k = np.arange(0, 22)
    for g in ('X', 'Y'):
        pm = v[g].pmf(k)
        # pmf: probability at each count, joined by a dotted guide line
        a_in.fill_between(k, pm, color=C[g], alpha=0.15, lw=0)
        a_in.plot(k, pm, color=C[g], lw=0.8, marker='o', ms=1.6, mew=0, clip_on=False)
    a_in.axvline(MU, color='0.3', ls='--', lw=0.5)
    a_in.set_yticks([])
    a_in.spines['left'].set_visible(False)
    a_in.set_xticks([0, 5, 10, 20])
    a_in.set_xlim(0, 21)
    a_in.set_ylim(0, None)
    a_in.set_xlabel('Counts of gene $g$')
    ymax = a_in.get_ylim()[1]
    a_in.text(MU + 0.6, ymax * 1.0, 'same mean', color='0.3', fontsize=TINY, va='bottom')
    a_in.text(6.2, ymax * 0.78, 'X, low variance', color=C['X'], fontsize=TINY, fontweight='bold')
    a_in.text(6.2, ymax * 0.64, 'Y, high variance', color=C['Y'], fontsize=TINY, fontweight='bold')

    # one box per test, method name above it
    bx, bw, bhh = 1.14, BW - 1.14, 0.8
    rows = {'ln': (B_TOP - 0.18 - bhh, LN, "LN's $t$-test", r'Test log ratio of means, $\log\,\dfrac{\mathbb{E}[X]}{\mathbb{E}[Y]}$',
                   'Asymptotically unbiased\nLFC estimate', ('no DE', '\u2714', LN)),
            'log1p': (B_TOP - 0.18 - bhh - 0.18 - bhh, LOG1P, 'log1p $t$-test', 'Test difference of log1p means',
                      'Biased LFC estimate', ('false\npositive', '\u2718', '#d62728'))}
    # parts 1-3 share x positions in both boxes: test plot, LFC estimate, conclusion
    px, pw = bx + 0.08, 1.68        # part 1
    ox, ow = bx + 1.9, 0.95         # part 2
    mkx = bx + bw - 0.22            # part 3 (tight)
    for key, (y0, col, name, sub, verdict, (text, mark, mcol)) in rows.items():
        rbox(bx, y0, bw, bhh, fc='0.94', ec='0.6', r=0.05)
        ax.text(bx + bw / 2, y0 + bhh + 0.02, name, color=col, fontsize=TEXT, ha='center', va='bottom')
        if key == 'ln':   # 'Test log ratio of means, log E[X]/E[Y]', with X and Y coloured
            ty = y0 + bhh - 0.14
            t_ = ax.text(px + pw / 2 + 0.2, ty, 'Test log ratio of means, log', fontsize=TINY, ha='right', va='center')
            num = [(r'\mathbb{E}[', None), V('X'), (']', None)]
            den = [(r'\mathbb{E}[', None), V('Y'), (']', None)]
            layered(lambda c: r'\dfrac{%s}{%s}' % (join(num, c, 'k'), join(den, c, 'k')), [C['X'], C['Y'], 'k'],
                    (px + pw / 2 + 0.21, ty), align=(0.0, 0.5), strut=(1.3, 1.6))
        else:
            ax.text(px + pw / 2, y0 + bhh - 0.04, sub, fontsize=TINY, ha='center', va='top', linespacing=1.05)
        ax.text(ox + ow / 2, y0 + bhh - 0.04, verdict, fontsize=TINY, ha='center', va='top', linespacing=1.05)
        est, se, p = v[key]
        a_o = inset(ox, y0 + 0.25, ow, 0.28)
        a_o.errorbar(est, 0, xerr=1.96 * se, color=col, marker='o', ms=2.5, lw=1.0, capsize=1.5, capthick=0.6)
        a_o.axvline(0, color='0.3', lw=0.6, ls=':')
        a_o.text(0.03, 0.55, 'true', fontsize=TINY, ha='left', va='bottom', color='0.3')
        a_o.text(0.9, 0.55, '$p \\ll 0.05$' if p < 0.05 else '$p > 0.05$', color=col, fontsize=TINY,
                 ha='right', va='bottom')
        a_o.set_ylim(-0.6, 1.3)
        a_o.set_xlim(-0.35, 0.9)
        a_o.set_xticks([0, 0.5])
        a_o.set_yticks([])
        a_o.spines['left'].set_visible(False)
        a_o.set_xlabel('LFC (95% CI)')
        ey = y0 + 0.25 + 0.28 * (0.6 / 1.9)
        ax.text(mkx, ey + 0.02, mark, color=mcol, fontsize=9, ha='center', va='bottom')
        ax.text(mkx, ey, text, fontsize=TINY, ha='center', va='top', linespacing=1.0)

    y0 = rows['log1p'][0]
    a_t = inset(px, y0 + 0.25, pw, 0.28)
    kk = np.arange(0, 36)   # log1p(35) = 3.58, inside the x-range, so no point runs past the axis
    for g in ('X', 'Y'):
        pm = v[g].pmf(kk)
        a_t.fill_between(np.log1p(kk), pm, color=C[g], alpha=0.15, lw=0)
        a_t.plot(np.log1p(kk), pm, color=C[g], lw=0.8, marker='o', ms=1.6, mew=0, clip_on=False)
        a_t.axvline(v['elog'][g], color=C[g], ls='--', lw=0.6)
    a_t.set_yticks([])
    a_t.spines['left'].set_visible(False)
    a_t.set_xlim(0, 3.6)
    a_t.set_ylim(0, None)
    a_t.set_xticks([0, 1, 2, 3])
    a_t.set_xlabel('log(1 + counts)')
    gy = a_t.get_ylim()[1] * 0.85
    a_t.annotate('', xy=(v['elog']['Y'], gy), xytext=(v['elog']['X'], gy),
                 arrowprops=dict(arrowstyle='<->', lw=0.5, color='k', shrinkA=0, shrinkB=0, mutation_scale=5))

    # LN's two terms per group, on their own scales (not log, not a shared axis): P(detected) and
    # E[count | detected], whose product is the mean; LN tests the log ratio of those means.
    # Values from the same sample the test uses.
    y0 = rows['ln'][0]
    sx, sy = v['sample']['X'], v['sample']['Y']
    # each pair: X's term above the zero line (blue), Y's below (orange); labels on the side of their bar
    def terms(g):
        return [[(r'\mathbb{P}(', None), V(g), ('>', None), ('0)', None)],
                [(r'\mathbb{E}[', None), V(g), (r'\mid', None), V(g), ('>', None), ('0]', None)],
                [(r'\mathbb{E}[', None), V(g), (']', None)]]
    vals = [((terms('X')[0], terms('Y')[0]), (np.mean(sx > 0), np.mean(sy > 0))),
            ((terms('X')[1], terms('Y')[1]), (sx[sx > 0].mean(), sy[sy > 0].mean())),
            ((terms('X')[2], terms('Y')[2]), (sx.mean(), sy.mean()))]
    assert np.isclose(np.log2(sx.mean() / sy.mean()), v['ln'][0])
    # P on its own 0-1 scale; the two count expectations share one scale, so multiplying by P visibly shrinks them
    m_counts = max(max(vals[1][1]), max(vals[2][1])) * 1.05
    cw_ = 0.25
    centers = (px + 0.28, px + 0.86, px + 1.44)
    for k, ((lx_, ly_), (vx_, vy_)) in enumerate(vals):
        a_k = inset(centers[k] - cw_ / 2, y0 + 0.17, cw_, 0.25)
        a_k.bar(0, vx_, width=0.7, color=C['X'], alpha=0.35, ec=C['X'], lw=0.6)
        a_k.bar(0, -vy_, width=0.7, color=C['Y'], alpha=0.35, ec=C['Y'], lw=0.6)
        a_k.axhline(0, color='0.3', lw=0.5)
        m_ = 1.05 if k == 0 else m_counts
        a_k.set_xlim(-0.6, 0.6)
        a_k.set_ylim(-m_, m_)
        a_k.axis('off')
        n_clip = len(CLIP)
        mathlabel(lx_, (0.5, 1.02), a_k.transAxes, (0.5, 0.0))
        mathlabel(ly_, (0.5, -0.02), a_k.transAxes, (0.5, 1.0))
        if k == 0:   # the detection term: its bars and both labels
            det_parts = [a_k] + [ta for ta, _ in CLIP[n_clip:]]
    for (c0, c1), sym in zip(zip(centers, centers[1:]), ('×', '=')):
        ax.text((c0 + c1) / 2, y0 + 0.295, sym, fontsize=SMALL, ha='center', va='center', color='0.3')

    # hide the sizing rules of the layered math: clip each layer to its box, less the rules at the ends
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for ta, ab in CLIP:
        (x0_, y0_), (x1_, y1_) = fig.transFigure.inverted().transform(ta.get_window_extent(rnd).get_points())
        dx = 0.8 / 72 / W
        ta._text.set_clip_box(TransformedBbox(Bbox([[x0_ + dx, y0_ - 0.05], [x1_ - dx, y1_ + 0.05]]), fig.transFigure))
        ta._text.set_clip_on(True)
    ext = Bbox.union([p.get_window_extent(rnd) for p in det_parts])
    (bx0, by0), (bx1, by1) = ext.get_points() / fig.dpi   # inches, as the drawing's data units
    # the labels' boxes already hold 1.5 pt beyond their ink at each end, so no extra width: the x keeps its gap
    rbox(bx0, by0 - 0.025, bx1 - bx0, by1 - by0 + 0.05, fc='none', ec=DET, lw=0.8, r=0.04, ls=(0, (3, 2)))

    # save: combined, then each half without the other
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / 'fig1_ab.png', dpi=400)
    parts_a = {k: list(getattr(ax, k)[split[k]:split_b[k]]) for k in split}
    parts_b = {k: list(getattr(ax, k)[split_b[k]:]) for k in split}
    axes_a, axes_b = fig.axes[1:n_axes_a], fig.axes[n_axes_a:]
    for part, keep, drop, axk, axd, box in (('a', parts_a, parts_b, axes_a, axes_b, Bbox([[0, B_TOP], [W, H]])),
                                            ('b', parts_b, parts_a, axes_b, axes_a, Bbox([[0, 0], [BW, B_TOP]]))):
        for grp, vis in ((keep, True), (drop, False)):
            for arts in grp.values():
                for a in arts:
                    a.set_visible(vis)
        for a in axk:
            a.set_visible(True)
        for a in axd:
            a.set_visible(False)
        for c_art in c_placeholder + letters:   # separate panels: no letters, no placeholder
            c_art.set_visible(False)
        # crop to the drawn content exactly: render, find the ink, save that box (no padding)
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=400)
        ink = np.array(Image.open(buf).convert('L')) < 255   # any ink, incl. antialiased edges
        ys, xs = np.where(ink)
        e = 0.012   # PDF text rasterises marginally wider than the 400-dpi PNG: keep glyph edges
        tight = Bbox([[xs.min() / 400 - e, H - (ys.max() + 1) / 400 - e], [(xs.max() + 1) / 400 + e, H - ys.min() / 400 + 3 * e]])   # mathtext glyphs run taller in PDF
        # then trim to what the PDF itself draws: render it, measure the blank margins, shrink the box
        import subprocess, tempfile
        pdf = HERE / f'fig1{part}.pdf'
        fig.savefig(pdf, bbox_inches=tight, pad_inches=0)
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(['pdftoppm', '-r', '1200', '-png', '-singlefile', str(pdf), f'{td}/p'],
                           check=True)
            im = np.array(Image.open(f'{td}/p.png').convert('L')) < 255
        ys, xs = np.where(im)
        d = 1200.0
        (x0, y0), (x1, y1) = tight.get_points()
        tight = Bbox([[x0 + xs.min() / d, y0 + (im.shape[0] - 1 - ys.max()) / d],
                      [x1 - (im.shape[1] - 1 - xs.max()) / d, y1 - ys.min() / d]])
        fig.savefig(pdf, dpi=400, bbox_inches=tight, pad_inches=0)
        fig.savefig(PREVIEW / f'fig1{part}.png', dpi=400, bbox_inches=tight, pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    main()
