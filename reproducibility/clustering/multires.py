#!/usr/bin/env python3
"""As Leiden resolution increases, do clusters annotated from each method's top markers stay correctly annotated, keep
their annotation across resolutions, and stay free of other types' markers?

Reads clustering/de.py's outputs (each cluster against the rest, five resolutions, both datasets) and runs no tests. Each
cluster's signature is its top TOP under the practice rule; it is annotated with the curated type (config.yaml) or tied
types of highest recall, each tied type weighted 1 over their number. Its reference type is the majority of its cells'
CellTypist labels mapped to the curated types (metrics/celltypes.csv, `coarse`). Readouts per dataset, method and
resolution: 1, annotation accuracy, the mean weight given to the reference type (clusters with a mapped majority);
2, the share of cells whose annotation is the same at consecutive resolutions from the middle one up; 3, the share of
clusters whose top TOP holds another type's marker that is not also the reference type's; 4, at the two finest
resolutions, the curated markers in one method's top TOP and not another's, with their cluster mean, detection and rank.

    python -m clustering.multires               # from reproducibility/, about a minute
    python -m clustering.multires --plot-only   # redraw from the CSVs
"""

import argparse

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

import plot_style
from clustering.celltype_labels import dataset_paths
from clustering.heldout_fp import de_output
from clustering.replication import save
from evaluation_utils import _get_rank_genes_df
from method_colors import method_color, method_draw_order, method_label
from paths import fig_name, require_input, results_dir

METHODS = ('ln', 't_test', 'wilcoxon')
TOP = 20
MIDDLE = 2          # consistency is read from the third resolution up
OUT = results_dir(__file__) / 'multires'


def ranked(adata, cluster, key):
    """The cluster's significant genes with a positive score, best first, as clustering/metrics.py takes them."""
    df = _get_rank_genes_df(adata, cluster, key)
    sig = df[(df['pvals_adj'] < 0.05) & (df['scores'] > 0)]
    return list(sig.sort_values('scores', ascending=False)['names'].astype(str))


def annotate(signature, curated):
    """Each tied best type with its weight; empty if no curated marker is in the signature."""
    recall = {t: len(signature & g) / len(g) for t, g in curated.items()}
    best = max(recall.values())
    if best == 0:
        return {}
    tied = sorted(t for t, r in recall.items() if r == best)
    return {t: 1 / len(tied) for t in tied}


def load(name):
    config_path, out = dataset_paths()[name]
    config = yaml.safe_load(open(config_path))
    keys = [k for _, k in config['leiden_resolutions']]
    with h5py.File(require_input(de_output(config['adata_path'], str(out)), what=f'the {name} DE output',
                                 source='python -m clustering.de'), 'r') as f:
        adata = ad.AnnData(obs=ad.io.read_elem(f['obs']), var=ad.io.read_elem(f['var']))
        cp = sp.csr_matrix(ad.io.read_elem(f['layers/norm_counts']))
        for m in METHODS:
            for k in keys:
                adata.uns[f'{m}_{k}'] = ad.io.read_elem(f[f'uns/{m}_{k}'])
    labels = pd.read_csv(require_input(out / 'metrics' / 'celltypes.csv', what=f'the {name} cell-type labels',
                                       source='python -m clustering.celltype_labels'), index_col=0)
    curated = {t: set(g) for t, g in config['markers'].items()}
    return adata, cp, labels['coarse'].reindex(adata.obs_names), keys, curated


def run():
    """Per-cluster annotations, the readouts, and the drivers of readout 1."""
    clusters, cells, drivers = [], [], []
    for name in dataset_paths():
        adata, cp, coarse, keys, curated = load(name)
        detected = (cp > 0).tocsr()
        all_markers = set().union(*curated.values())
        gene_index = {g: j for j, g in enumerate(adata.var_names)}
        lists = {}
        for i, key in enumerate(keys):
            lab = adata.obs[key].astype(str).to_numpy()
            ids = sorted(np.unique(lab), key=lambda v: int(v) if v.isdigit() else v)
            for c in ids:
                own = lab == c
                counts = coarse[own].fillna('(unmapped)').value_counts().rename_axis('type').reset_index()
                top = counts.sort_values(['count', 'type'], ascending=[False, True])['type'].iloc[0]
                reference = None if top == '(unmapped)' else top
                for m in METHODS:
                    full = ranked(adata, c, f'{m}_{key}')
                    lists[key, c, m] = full
                    sig = set(full[:TOP])
                    ann = annotate(sig, curated)
                    wrong = bool(reference) and any(g in curated[t] and g not in curated[reference]
                                                    for t in curated if t != reference for g in sig)
                    clusters.append({'dataset': name, 'resolution': key, 'clusters': len(ids), 'cluster': c,
                                     'cells': int(own.sum()), 'reference': reference, 'method': m,
                                     'annotation': ';'.join(ann), 'weight_reference': ann.get(reference, 0.0),
                                     'wrong_type': wrong, 'top': ';'.join(full[:TOP])})
                    if i >= MIDDLE:
                        cells.append(pd.DataFrame({'dataset': name, 'resolution': key, 'method': m,
                                                   'cell': adata.obs_names[own], 'annotation': ';'.join(ann)}))
            if i >= len(keys) - 2:
                for c in ids:
                    own = lab == c
                    mean = np.asarray(cp[own].mean(0)).ravel()
                    det = np.asarray(detected[own].mean(0)).ravel()
                    for m, other in [(a, b) for a in METHODS for b in METHODS if a != b and 'ln' in (a, b)]:
                        mine, theirs = lists[key, c, m], lists[key, c, other]
                        for g in sorted((set(mine[:TOP]) - set(theirs[:TOP])) & all_markers):
                            j = gene_index[g]
                            drivers.append({'dataset': name, 'resolution': key, 'cluster': c, 'cells': int(own.sum()),
                                            'in_top': m, 'missing_from': other, 'gene': g,
                                            'mean_cp10k': float(mean[j]), 'detection': float(det[j]),
                                            'rank_in_top': mine.index(g) + 1,
                                            'rank_in_other': theirs.index(g) + 1 if g in theirs else np.nan})
    return pd.DataFrame(clusters), pd.concat(cells, ignore_index=True), pd.DataFrame(drivers)


def readouts(clusters, cells):
    """Per dataset, method and resolution: accuracy (cluster- and cell-weighted), wrong-type share, and the
    consistency of each transition from the middle resolution up."""
    c = clusters[clusters['reference'].notna()].copy()
    c['weighted'] = c['weight_reference'] * c['cells']
    g = c.groupby(['dataset', 'resolution', 'clusters', 'method'])
    out = pd.DataFrame({'clusters_with_reference': g.size(), 'accuracy': g['weight_reference'].mean(),
                        'accuracy_cells': g['weighted'].sum() / g['cells'].sum(), 'wrong_type': g['wrong_type'].mean()})
    out = out.reset_index()
    unmapped = clusters[clusters['reference'].isna()].groupby(['dataset', 'resolution', 'method']).size()
    out['clusters_unmapped'] = [int(unmapped.get((d, r, m), 0)) for d, r, m in
                                zip(out['dataset'], out['resolution'], out['method'])]
    order = {name: [k for _, k in yaml.safe_load(open(dataset_paths()[name][0]))['leiden_resolutions']]
             for name in dataset_paths()}
    rows = []
    for (name, m), d in cells.groupby(['dataset', 'method']):
        keys = [k for k in order[name] if k in set(d['resolution'])]
        w = d.pivot(index='cell', columns='resolution', values='annotation')
        for a, b in zip(keys[:-1], keys[1:]):
            rows.append({'dataset': name, 'method': m, 'from': a, 'to': b, 'consistency': float((w[a] == w[b]).mean())})
    return out, pd.DataFrame(rows)


def plot(res, cons, clusters):
    """The three readouts against the number of clusters, and at the finest resolution each cluster's recall of each
    curated type in its top TOP, clusters grouped by reference type."""
    plot_style.use()
    names = list(dataset_paths())
    fig, axes = plt.subplots(len(names), 3, figsize=plot_style.figsize(3, len(names)))
    for row, name in zip(axes, names):
        d = res[res['dataset'] == name]
        for ax, col, label in zip(row, ('accuracy', None, 'wrong_type'),
                                  ('Clusters annotated correctly', 'Cells keeping their annotation',
                                   'Clusters with another type\'s marker')):
            for m in method_draw_order(METHODS):
                if col is None:
                    e = cons[(cons['dataset'] == name) & (cons['method'] == m)]
                    x = [d.loc[d['resolution'] == r, 'clusters'].iloc[0] for r in e['to']]
                    y = e['consistency']
                else:
                    e = d[d['method'] == m].sort_values('clusters')
                    x, y = e['clusters'], e[col]
                ax.plot(x, y, marker=plot_style.ERRORBAR['marker'], ms=plot_style.ERRORBAR['ms'],
                        lw=plot_style.ERRORBAR['lw'], color=method_color(m), label=method_label(m))
            ax.set_ylim(0, 1)
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel(label)
            ax.set_title(name)
    save(fig, OUT / fig_name('multires_annotation'))

    fig, axes = plt.subplots(len(names), len(METHODS), figsize=plot_style.figsize(len(METHODS), len(names)))
    for row, name in zip(axes, names):
        config = yaml.safe_load(open(dataset_paths()[name][0]))
        finest = config['leiden_resolutions'][-1][1]
        curated = {t: set(g) for t, g in config['markers'].items()}
        types = list(curated)
        d = clusters[(clusters['dataset'] == name) & (clusters['resolution'] == finest)]
        for ax, m in zip(row, METHODS):
            e = d[d['method'] == m].assign(reference=lambda v: v['reference'].fillna('(unmapped)'))
            e = e.assign(order=e['reference'].map({t: i for i, t in enumerate(types + ['(unmapped)'])}))
            e = e.sort_values(['order', 'cluster'])
            tops = [set(str(v).split(';')) for v in e['top'].fillna('')]
            grid = np.array([[len(top & curated[t]) / len(curated[t]) for t in types] for top in tops])
            ax.imshow(grid, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=90, fontsize=plot_style.SMALL)
            ax.set_yticks(range(len(e)))
            ax.set_yticklabels(e['reference'], fontsize=plot_style.SMALL)
            ax.set_title(f'{name}, {method_label(m)}')
    save(fig, OUT / fig_name('multires_heatmaps'), key=False)


def main():
    parser = argparse.ArgumentParser(description='Annotation from markers across Leiden resolutions')
    parser.add_argument('--plot-only', action='store_true', help='Redraw the figures from the saved CSVs')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    csv = {k: OUT / f'{k}.csv' for k in ('clusters', 'readouts', 'consistency', 'drivers')}
    if args.plot_only:
        tables = {k: pd.read_csv(require_input(p, what=f'the multires {k}', source='this script without --plot-only'))
                  for k, p in csv.items()}
    else:
        clusters, cells, drivers = run()
        res, cons = readouts(clusters, cells)
        tables = {'clusters': clusters, 'readouts': res, 'consistency': cons, 'drivers': drivers}
        for k, t in tables.items():
            t.to_csv(csv[k], index=False)
            print(f'  wrote {csv[k]}')
    plot(tables['readouts'], tables['consistency'], tables['clusters'])


if __name__ == '__main__':
    main()
