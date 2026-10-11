"""Extended Data Fig 10: Abdelfattah et al. 2022 as the authors ran it, their genes ranked by the Wilcoxon AUC,
assembled from published.abdelfattah2022's drawing code.

a, the NES heatmaps of their Fig. 4d pathways under the AUC ranking (their procedure, reproducing 96% of the published
signs) and under LN, on the published figure's colour range; b, per cluster, how many of those pathways keep their call
(FDR < 0.05) under LN, lose it, reverse it or gain one; c, MC3 over MC4 in the contrasted pathways, the AUC's and LN's
NES side by side; d, each cluster's expressed genes against its log2 depth ratio, the AUC and LN's score; e, each
cluster's Hallmark NES against its log2 depth ratio, every set a point over its box. The uncentred AUC lies near 0.5 for
nearly every gene, so fgsea weights the genes almost equally and its NES are larger than under a score (Methods). Fig 2g-i
and Extended Data Fig 9g-h are the same with the Wilcoxon's z-score. Each plot is drawn by its own drawing code from the
saved results, at the size it prints at; a change to a plot belongs in that code.

    python -m figures.ed10     # from reproducibility/, seconds; needs published/results/
"""

from figures import ed9
from figures import ed_abdelfattah as ea
from figures.layout import assemble, pdf_size, use
from figures.layout import ED_WIDTH as WIDTH
from published import abdelfattah2022 as ab

COMP = 'wilcoxon'   # the authors' ranking
TAG = 'ed10'


if __name__ == '__main__':
    use()
    t = ea.tables()
    a, height = ea.heatmaps(t, COMP, TAG, letter='a', shared_xlabel=True, col=0.1, cbar_labels=0.42)
    b = ea.call_changes_panel(t['gsea'], height, f'{TAG}_b_call_changes', comp=COMP)
    c = ea.contrast_by_cluster(t, WIDTH - sum(pdf_size(x)[0] for x in (*a, b)) - 0.01, height, f'{TAG}_c_contrast',
                               comp=COMP)
    values = ab.values_by_cluster(t['rankings'], t['gsea'])
    d = [ed9.genes(t, values, k, m, ylabel, ref, comp=COMP, stem=f'{TAG}_d{k + 1}_genes_{m}')
         for k, (m, ylabel, ref) in enumerate(ab.levels(COMP)['genes'])]
    shared = ab.value_range([values['pathways', m] for m in (COMP, 'ln')])
    e = [ed9.pathways(t, values, k, m, shared, stem=f'{TAG}_e{k + 1}_pathways_{m}') for k, m in enumerate((COMP, 'ln'))]
    assemble([[('a', a[0], 0), *[('', x, 0) for x in a[1:]], ('b', b, 0), ('c', c, 0)],
              [('d', d[0], 0), ('', d[1], 0), ('e', e[0], 0), ('', e[1], 0)]], TAG, width=WIDTH)
