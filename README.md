# LN's $t$-test

[![CI](https://github.com/okviman/lntest/actions/workflows/ci.yml/badge.svg)](https://github.com/okviman/lntest/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ln-ttest.svg)](https://pypi.org/project/ln-ttest/)

Differential expression testing with LN's $t$-test, on an asymptotically unbiased estimator of the log-fold change.

## Installation

```bash
pip install ln-ttest
```

## Usage

```python
from lntest import rank_genes_groups_ln

rank_genes_groups_ln(adata, groupby="leiden", layer="norm_counts")  # counts or normalized values, not log-transformed
adata.uns["rank_genes_groups"]["logfoldchanges"]
adata.uns["rank_genes_groups"]["lfc_se"]
```

`rank_genes_groups_ln` takes the arguments of `scanpy.tl.rank_genes_groups` and stores its results in the same place.
`lfc_se` holds the standard error of each log-fold change.

To select the markers of each cluster, we recommend taking the genes that LN's $t$-test calls significant and up in the
cluster, ranked by the log ratio of their detection rates over its standard error:

```python
rank_genes_groups_ln(adata, groupby="leiden", layer="norm_counts", corr_method="bonferroni", rankby="markers")
adata.uns["rank_genes_groups"]["names"]["3"][:20]  # the top 20 markers of cluster 3
```

For two groups of cells, `get_LN_lfcs(Y, X)` returns the log-fold change of `Y` over `X` and the p-value of each gene.

## Reproducing the paper

[`reproducibility/`](reproducibility/README.md) holds the code behind the analyses and figures of the paper, with a
table from each figure to the script that builds it.

## Example

The notebook [PBMC3k](notebooks/pbmc3k.ipynb) compares the genes that LN's $t$-test and scanpy's $t$-test call
significant on the 3k PBMC data set from 10x Genomics.
