#!/usr/bin/env python3
"""Cell-type labels for both of the arm's datasets, PBMC3k and PBMC10k, from CellTypist's Immune_All_Low model with
majority voting on log1p CP10K, mapped by MAPPING to the cell types whose curated markers config.yaml lists. A type is
included if it has at least MIN_CELLS cells in each dataset. PBMC10k's labels are checked against its antibodies: per
type, the share of cells with the surface phenotype PHENOTYPES expects, each antibody gated by a Gaussian mixture on
its CLR values, as citeseq/R/pbmc10k_process.R gates CD3, CD4 and CD45RA.

Writes each dataset's metrics/celltypes.csv, the mapping (celltype_mapping.csv) and PBMC10k's
metrics/celltype_antibodies.csv.

    python -m clustering.celltype_labels      # from reproducibility/, under a minute; downloads the model once
"""

import anndata as ad
import celltypist
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml
from celltypist import models
from sklearn.mixture import GaussianMixture

from clustering.heldout_fp import de_output
from paths import require_input, results_dir

MODEL = 'Immune_All_Low.pkl'   # v2 of 2022-07-16, sha256 290874d35dac039d4c9218c343fde4aac1077709b72a331ce7266f6828c36502
MAPPING = {
    'Tcm/Naive helper T cells': 'Naive CD4 T',   # naive and central memory together, as the model has them
    'Tem/Effector helper T cells': 'Memory CD4 T',
    'Regulatory T cells': 'Memory CD4 T',
    'Tcm/Naive cytotoxic T cells': 'CD8 T',
    'Tem/Trm cytotoxic T cells': 'CD8 T',
    'Tem/Temra cytotoxic T cells': 'CD8 T',
    'CD16+ NK cells': 'NK',
    'CD16- NK cells': 'NK',
    'NK cells': 'NK',
    'B cells': 'B',
    'Naive B cells': 'B',
    'Memory B cells': 'B',
    'Classical monocytes': 'CD14 Mono',
    'Non-classical monocytes': 'FCGR3A Mono',
    'DC': 'DC',
    'DC1': 'DC',
    'DC2': 'DC',
    'Megakaryocytes/platelets': 'Platelet',
}
MIN_CELLS = {'PBMC3k': 30, 'PBMC10k': 150}
# Each type's expected surface phenotype: antibody -> positive or not. CD4 is gated in three, as the CITE-seq arm gates
# it; positive is its top component, the T helper cells' level.
PHENOTYPES = {
    'Naive CD4 T': {'CD3': True, 'CD4': True, 'CD8a': False},
    'Memory CD4 T': {'CD3': True, 'CD4': True, 'CD8a': False},
    'CD8 T': {'CD3': True, 'CD8a': True, 'CD4': False},
    'NK': {'CD3': False, 'CD56': True},
    'B': {'CD3': False, 'CD19': True},
    'CD14 Mono': {'CD3': False, 'CD14': True},
    'FCGR3A Mono': {'CD3': False, 'CD56': False, 'CD16': True},
    'DC': {'CD3': False, 'CD19': False, 'CD16': False, 'HLA-DR': True},
}
# CD197 (CCR7) is left out: in this panel it stains every type at background level.
ANTIBODIES = ('CD3', 'CD4', 'CD8a', 'CD19', 'CD14', 'CD16', 'CD56', 'HLA-DR', 'CD45RA', 'CD45RO')
COMPONENTS = {'CD4': 3}        # every other antibody in two
RAW = results_dir(__file__).parents[1] / 'citeseq' / 'data' / 'sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5'


def dataset_paths():
    """Each dataset's config and results directory."""
    here = results_dir(__file__)
    return {'PBMC3k': (here.parent / 'config.yaml', here), 'PBMC10k': (here.parent / 'config_pbmc10k.yaml', here / 'pbmc10k')}


def annotate(adata_path):
    """CellTypist's per-cell and majority-voting labels for the h5ad's cells, on its log1p CP10K layer."""
    with h5py.File(adata_path, 'r') as f:
        x = sp.csr_matrix(ad.io.read_elem(f['layers/log1p_norm']))
        obs, var = ad.io.read_elem(f['obs']), ad.io.read_elem(f['var'])
    adata = ad.AnnData(X=x, obs=pd.DataFrame(index=obs.index), var=pd.DataFrame(index=var.index))
    return celltypist.annotate(adata, model=MODEL, majority_voting=True).predicted_labels


def clr(x):
    """Seurat's CLR across cells, one antibody at a time, as NormalizeData(normalization.method = 'CLR') computes it."""
    return np.log1p(x / np.exp(np.log1p(x[x > 0]).sum() / len(x)))


def gate(values, components, seed=0):
    """Each cell's component, 0 the lowest, from a Gaussian mixture on one antibody's CLR values."""
    gm = GaussianMixture(components, random_state=seed).fit(values[:, None])
    rank = np.argsort(np.argsort(gm.means_.ravel()))
    return rank[gm.predict(values[:, None])]


def antibody_gates(barcodes):
    """Per cell, whether each of ANTIBODIES is in its top component, for PBMC10k's cells."""
    full = sc.read_10x_h5(str(require_input(RAW, what="the 5' PBMC 10k matrix", source='python -m fetch_data --arm citeseq')),
                          gex_only=False)
    full.var_names_make_unique()
    adt = full[barcodes, full.var['feature_types'] == 'Antibody Capture']
    names = [n.replace('_TotalC', '') for n in adt.var_names]
    counts = pd.DataFrame(np.asarray(adt.X.todense()), index=barcodes, columns=names)
    out = {}
    for a in ANTIBODIES:
        k = COMPONENTS.get(a, 2)
        out[a] = gate(clr(counts[a].to_numpy(dtype=float)), k) == k - 1
    return pd.DataFrame(out, index=barcodes)


def check(labels, gates):
    """Per included type, its cells, the share with the phenotype PHENOTYPES expects, each antibody's positive share,
    and the share naive (CD45RA+ CD45RO-)."""
    rows = []
    for t, expected in PHENOTYPES.items():
        cells = labels.index[labels['coarse'] == t]
        if not len(cells):
            continue
        g = gates.loc[cells]
        match = np.logical_and.reduce([g[a] == pos for a, pos in expected.items()])
        rows.append({'type': t, 'cells': len(cells), 'expected': ', '.join(f'{a}{"+" if p else "-"}' for a, p in expected.items()),
                     'agreement': float(match.mean()), 'naive': float((g['CD45RA'] & ~g['CD45RO']).mean()),
                     **{f'{a}+': float(g[a].mean()) for a in ANTIBODIES}})
    return pd.DataFrame(rows)


def main():
    labels = {}
    for name, (config_path, out) in dataset_paths().items():
        config = yaml.safe_load(open(config_path))
        adata_path = require_input(de_output(config['adata_path'], str(out)), what=f'the {name} DE-annotated AnnData',
                                   source='python -m clustering.de (see README)')
        lab = annotate(adata_path)[['predicted_labels', 'over_clustering', 'majority_voting']].astype(str)
        lab['coarse'] = lab['majority_voting'].map(MAPPING).fillna('')
        labels[name] = lab
    counts = pd.DataFrame({name: lab['coarse'].value_counts() for name, lab in labels.items()}).fillna(0).astype(int)
    included = sorted(t for t in counts.index if t and all(counts.at[t, n] >= MIN_CELLS[n] for n in labels))
    paths = dataset_paths()
    for name, lab in labels.items():
        lab['included'] = lab['coarse'].isin(included)
        out = paths[name][1] / 'metrics' / 'celltypes.csv'
        lab.rename_axis('barcode').to_csv(out)
        print(f'wrote {out}: {name}, {int(lab["included"].sum())} of {len(lab)} cells in {len(included)} included types')
    fine = pd.DataFrame({name: lab['majority_voting'].value_counts() for name, lab in labels.items()}).fillna(0).astype(int)
    mapping = fine.rename_axis('celltypist').reset_index().rename(columns={n: f'cells_{n}' for n in labels})
    mapping.insert(1, 'coarse', mapping['celltypist'].map(MAPPING).fillna(''))
    mapping['included'] = mapping['coarse'].isin(included)
    out = results_dir(__file__) / 'celltype_mapping.csv'
    mapping.sort_values(['included', 'coarse', 'celltypist'], ascending=[False, True, True]).to_csv(out, index=False)
    print(f'wrote {out}')
    gates = antibody_gates(labels['PBMC10k'].index)
    result = check(labels['PBMC10k'], gates)
    out = paths['PBMC10k'][1] / 'metrics' / 'celltype_antibodies.csv'
    result.to_csv(out, index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
