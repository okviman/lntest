"""Supplementary Fig 8: Abdelfattah et al. 2022 as run in the original study, with the genes ranked by the Wilcoxon AUC,
assembled from the drawing code of published.abdelfattah2022.

a, the NES heatmaps of the Fig. 4d pathways of the original study under the AUC ranking (its procedure, which
reproduces 96% of the published signs) and under LN, on the colour range of the published figure.
b, the number of those pathways per cluster that keep their call (FDR < 0.05) under LN, lose it, reverse it or gain one.
c, the NES of MC3 over MC4 in the contrasted pathways under the AUC and LN, side by side.
d, the AUC and LN's score of the expressed genes of each cluster against its log2 depth ratio.
e, the Hallmark NES of each cluster against its log2 depth ratio, every set a point over its box.
The uncentred AUC is near 0.5 for nearly every gene. The genes then weigh almost equally in fgsea, which makes the NES
larger than under a score (Methods). Fig 2g-i and Extended Data Fig 5g, h are the same with the Wilcoxon z-score.

    python -m figures.suppfig8     # from reproducibility/, seconds; needs published/results/
"""

from figures import edfig5
from figures import ed_abdelfattah as ea
from figures.layout import assemble, pdf_size, use
from figures.layout import ED_WIDTH as WIDTH
from published import abdelfattah2022 as ab

COMP = 'wilcoxon'   # the authors' ranking
TAG = 'suppfig8'


if __name__ == '__main__':
    use()
    t = ea.tables()
    a, height = ea.heatmaps(t, COMP, TAG, letter='a', shared_xlabel=True, col=0.1, cbar_labels=0.42)
    b = ea.call_changes_panel(t['gsea'], height, f'{TAG}_b_call_changes', comp=COMP)
    c = ea.contrast_by_cluster(t, WIDTH - sum(pdf_size(x)[0] for x in (*a, b)) - 0.01, height, f'{TAG}_c_contrast',
                               comp=COMP)
    values = ab.values_by_cluster(t['rankings'], t['gsea'])
    d = [edfig5.genes(t, values, k, m, ylabel, ref, comp=COMP, stem=f'{TAG}_d{k + 1}_genes_{m}')
         for k, (m, ylabel, ref) in enumerate(ab.levels(COMP)['genes'])]
    shared = ab.value_range([values['pathways', m] for m in (COMP, 'ln')])
    e = [edfig5.pathways(t, values, k, m, shared, stem=f'{TAG}_e{k + 1}_pathways_{m}') for k, m in enumerate((COMP, 'ln'))]
    assemble([[('a', a[0], 0), *[('', x, 0) for x in a[1:]], ('b', b, 0), ('c', c, 0)],
              [('d', d[0], 0), ('', d[1], 0), ('e', e[0], 0), ('', e[1], 0)]], TAG, width=WIDTH)
