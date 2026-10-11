# LN's $t$-test

[![CI](https://github.com/okviman/lntest/actions/workflows/ci.yml/badge.svg)](https://github.com/okviman/lntest/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ln-ttest.svg)](https://pypi.org/project/ln-ttest/)

Differential expression testing for single-cell transcriptomics with reduced false discoveries.

## Installation

```bash
pip install ln-ttest
```

## Usage

```python
from lntest import rank_genes_groups_ln

rank_genes_groups_ln(adata, groupby="leiden", layer="norm_counts") # normalized, not log-transformed
adata.uns["rank_genes_groups"]["logfoldchanges"]
adata.uns["rank_genes_groups"]["lfc_se"]
```

## Example

The notebook [PBMC3k](notebooks/pbmc3k.ipynb) showcases LN's $t$-test finding far fewer DEGs than scanpy's $t$-test on the 3k PBMCS data set from 10x Genomics.
