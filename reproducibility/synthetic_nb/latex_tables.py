import numpy as np
import pandas as pd

from paths import fig_name, require_input, results_dir
import matplotlib.pyplot as plt

import plot_style
from method_colors import MEDIANPROPS, method_color, method_label, method_order, method_rank

df = pd.read_csv(require_input(
    results_dir(__file__, create=False) / "nde_mu10" / "d1_vs_d2_01_nde_mu_10.csv",
    what="arm B's per-replicate metrics, sparse setting (non_de_mu=10)",
    source="run `python -m synthetic_nb.de_test` from reproducibility/ first",
))
# df2 = pd.read_csv("./simul/test/NB_test_results/nde_mu100_be/results.csv")

df.drop(columns=['recall'], inplace=True)

df['method'] = df['method'].map(method_label)
methods = method_order(df['method'])
df['dispersion'] = df['dispersion'].apply(lambda x: str(round(x, 1)))

df_avg = df.groupby(['method', 'dispersion']).mean().reset_index().drop(columns=['rep_no'])
df_avg = df_avg.sort_values(by=['dispersion', 'method'], key=lambda c: c.map(method_rank) if c.name == 'method' else c)
ln = method_label('LN')
df_avg['method'] = df_avg['method'].replace({ln: rf'\underbar{{{ln}}}'})
print(df_avg.to_latex(index=False, float_format='%.3f'))

metrics = ['accuracy', 'precision', 'tpr', 'tnr', 'fpr', 'fnr', 'f1']
mnames = ['Accuracy', 'Precision', 'TPR', 'TNR', 'FPR', 'FNR', 'F1']
plot_style.use()
rng = np.random.default_rng(0)
for dispersion in sorted(set(df.dispersion)):
    sub = df[df.dispersion == dispersion]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.6))
    for ax, metric, name in zip(axes, metrics, mnames):
        bp = ax.boxplot([sub.loc[sub.method == m, metric] for m in methods], widths=0.6,
                        patch_artist=True, medianprops=MEDIANPROPS, showfliers=False)
        for k, (box, m) in enumerate(zip(bp['boxes'], methods)):
            box.set_facecolor(method_color(m))
            box.set_edgecolor('none')
            plot_style.strip(ax, k + 1, sub.loc[sub.method == m, metric], method_color(m), 0.6, rng)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=90)
        plot_style.boxplot_grid(ax)
        ax.set_title(name)
    fig.suptitle(r'$(\phi_{Y_j}, \phi_{X_j}) =$' + f' (1.0, {dispersion})')
    fig.tight_layout()
    out = results_dir(__file__) / fig_name(f'degs_boxplots_dispersion_{dispersion}')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close('all')
    print(f'wrote {out}')
