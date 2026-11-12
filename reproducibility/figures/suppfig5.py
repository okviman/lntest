"""Supplementary Fig 5: kidney UMI downsampling, assembled from the figures of the kidney experiments in the layout of
Fig 1f-h.

a: downsample_null_fpr_log (kidney.fpr_plots).
b: the AUPRC, the average precision of each replicate, as in Fig 1g.
c: downsample_degs_100rep_pr_curves, with the method key and the p_sample key beside it, as in Fig 1.
d: the share of null genes with uncorrected p below 0.05 on the downsampling null without DEGs
(kidney.null_calibration.draw_calibration).
e: downsample_degs_100rep_tpr_fpr with its key (kidney.spot_split_plots).

    python -m figures.suppfig5      # from reproducibility/, a few seconds; needs the downsampling results
"""

import json

import pandas as pd

from figures.kidney_panels import calibration, null_auc_curves, pair
from figures.layout import assemble, use
from figures.layout import ED_WIDTH as WIDTH
from kidney import spot_split_plots as ssp
from kidney.fpr_plots import SAMPLING
from paths import require_input, results_dir

KIDNEY = results_dir(ssp.__file__)


if __name__ == '__main__':
    use()
    null = pd.read_csv(require_input(KIDNEY / 'downsample_null_results.csv', what='the no-DEG downsampling results',
                                     source='python -m kidney.umi_null (see README)'))
    df = pd.read_csv(require_input(KIDNEY / 'downsample_degs_100rep_results.csv', what='the with-DEG downsampling results',
                                   source='python -m kidney.umi_de (see README)'))
    curve_data = json.load(open(require_input(KIDNEY / 'downsample_degs_100rep_pr_curve_data.json', what='its PR curves',
                                              source='python -m kidney.umi_de (see README)')))
    a, b, c = null_auc_curves(null, df, curve_data, SAMPLING, ('suppfig5_a_downsample_null_fpr', 'suppfig5_b_downsample_auprc',
                                                           'suppfig5_c_downsample_pr_curves'), width=WIDTH)
    d = calibration('downsampling', 'UMI downsampling, no DEGs', 'suppfig5_d_downsample_calibration', WIDTH / 3)
    e = pair(ssp.draw_tpr_fpr, df, SAMPLING, 'suppfig5_e_downsample_tpr_fpr', width=WIDTH - WIDTH / 3)
    assemble([[('a', a, 0), ('b', b, 0), ('c', c, 0)],
              [('d', d, 0), ('e', e, 0)]], 'suppfig5', width=WIDTH)
