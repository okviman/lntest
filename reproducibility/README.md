# Reproducing the paper

This directory holds the code behind the analyses and figures of the paper, one directory per experiment.
The table below gives the script that assembles each figure. The sections after it give the commands that produce
the results each figure is drawn from.
The `lntest` package itself is described in the [top-level README](../README.md).

## Figures

| In the paper | Built by | Shows |
|---|---|---|
| Fig 1 | `figures.fig1` | The test, the synthetic null and kidney spot splitting |
| Fig 2 | `figures.fig2` | Lymph node, PBMC10k and the glioma myeloid cell atlas |
| Extended Data Fig 1 | `figures.edfig1` | Synthetic negative binomial data |
| Extended Data Fig 2 | `figures.edfig2` | Comparison with MAST and runtime scaling |
| Extended Data Fig 3 | `figures.edfig3` | Spot splitting experiment on kidney Visium HD data |
| Extended Data Fig 4 | `figures.edfig4` | Differentially expressed genes and reported effect sizes of the PBMC clusters |
| Extended Data Fig 5 | `figures.edfig5` | Cluster depth, differentially expressed genes and enrichment in the glioma myeloid cell atlas |
| Supplementary Fig 1 | no script | Beta distribution and its log-normal approximation |
| Supplementary Fig 2 | no script | Standard errors of the $P(Y>0)$ estimator |
| Supplementary Fig 3 | no script | KL maps between the Beta and the log-normal |
| Supplementary Fig 4 | `figures.suppfig4` | Kidney capsules split by sequencing depth |
| Supplementary Fig 5 | `figures.suppfig5` | Robustness to UMI downsampling in the kidney data set |
| Supplementary Fig 6 | `figures.suppfig6` | Robustness to spots and UMI downsampling on lymph node Visium HD data |
| Supplementary Fig 7 | `figures.suppfig7` | PBMC3k and PBMC10k multi-resolution clustering |
| Supplementary Fig 8 | `figures.suppfig8` | The glioma myeloid cell atlas under the Wilcoxon AUC |

Supplementary Figs 1-3 illustrate the log-normal approximation of the Supplementary Notes. Only their image files
exist, in the manuscript sources.

Each figure script places panels that the experiment's own code draws from its saved results, at the size they print.
Run the experiments first, then:

```bash
python -m figures.fig1        # -> figures/results/paper/fig1.pdf
python -m figures.fig2
python -m figures.edfig1      # likewise edfig2 to edfig5
python -m figures.suppfig4    # likewise suppfig5 to suppfig8
```

Each takes seconds and writes the figure to `figures/results/paper/` and its panels to `figures/results/panels/`.
Fig 1a,b and Fig 2a are diagrams, committed as PDFs in `figures/diagrams/`, so assembling Fig 1 and Fig 2 needs
neither LaTeX nor the full-resolution H&E. `python -m figures.diagrams.fig1_ab` redraws Fig 1a,b with LaTeX, and
`python -m figures.diagrams.fig2_lymphnode` redraws Fig 2a from the lymph node H&E.

Figures outside the paper are written to `figures/results/other/`: `figures.ed_abdelfattah` (the glioma rerun with
every panel), `figures.citeseq`, `figures.pbmc_lfc`, `figures.pbmc_story` and `figures.lymphnode_story`.

## Setup

```bash
conda env create -f env.yaml
conda activate de-ziln-reproducibility
pip install -e ..                     # the lntest package, from ../src/
export R_HOME="$CONDA_PREFIX/lib/R"
```

`env.lock.yaml` records every version of the environment the figures were made in, on arm64 macOS, where
`conda env create -f env.lock.yaml` builds it.
`R_HOME` is required wherever rpy2 is imported (the clustering, lymph node and MAST scripts). Without it, the
interpreter aborts at import with `Error in substring(x, m + 1L) : invalid substring arguments`.

Run every script from this directory, as a module:

```bash
python -m <experiment>.<script>       # for example python -m synthetic_nb.small_test
```

The `-m` form puts this directory on `sys.path`, which is how the scripts import the shared modules beside the
experiment directories (`paths`, `baselines`, `de_utils`, `evaluation_utils`, `method_colors`, `plot_style`).
Most analysis scripts take `--plot-only` to redraw their figures from the saved results.

### Data

| Experiment | Input | How to get it |
|---|---|---|
| `synthetic_nb`, `theory` | none | |
| `kidney` | Visium HD Human Kidney FFPE (10x Genomics) | `python -m kidney.preprocess`, 11 GB with the full-resolution H&E |
| `lymphnode` | Visium HD Human Lymph Node FFPE (10x Genomics) | `bash lymphnode/reproduce_vishd_cluster1_cluster3.sh`, 4.4 GB and several hours |
| `clustering` | PBMC3k and 5' PBMC 10k (10x Genomics), Monaco et al. 2019, Human Protein Atlas 25.1 | `python -m fetch_data --arm clustering` and `--arm citeseq` |
| `published` | GEO GSE182109 and the authors' cell metadata | `python -m fetch_data --arm published`, 2.5 GB |
| `citeseq` | memory CD4 T cells of the 5' PBMC 10k | committed as `citeseq/data/memory_CD4.h5ad` |

```bash
python -m fetch_data                  # every input that is missing
python -m fetch_data --arm clustering # the inputs of one experiment
python -m fetch_data --list           # what is present
```

Every download listed in `data_sources.yaml` is verified against its sha256, except the three multi-GB spatial
archives, which are checked by size.
Every script checks its inputs before it starts and names any missing file and where it comes from.

## Experiments

### `synthetic_nb`: negative binomial simulations (Fig 1c, Extended Data Figs 1 and 2)

```bash
python -m synthetic_nb.null --base-mu 5            # about 40 s, the null sweep at mean 5
python -m synthetic_nb.null --base-mu 50           # and at mean 50
python -m synthetic_nb.null_ratio                  # Fig 1c, from both sweeps
python -m synthetic_nb.lfc_confidence_intervals    # Extended Data Fig 1a, b
python -m synthetic_nb.de_test                     # about 3 minutes, simulations with DEGs, sparse regime
python -m synthetic_nb.de_test --setting dense     # dense regime
python -m synthetic_nb.de_ratio                    # Extended Data Fig 1c, d, from both regimes
python -m synthetic_nb.null_mast                   # about 7 minutes, Extended Data Fig 2a, b
python -m synthetic_nb.runtime --sparse            # about 45 minutes, Extended Data Fig 2c
```

`synthetic_nb.small_test` is one run of the simulations with DEGs, for a quick check of the setup. Its numbers are not
those of a figure, which summarise 20 runs.
Runtimes depend on the machine, which `synthetic_nb.runtime` records with the timings.

### `kidney`: Visium HD kidney capsules (Fig 1, Extended Data Fig 3, Supplementary Figs 4 and 5)

```bash
python -m kidney.preprocess          # the downloads, then about 9 minutes: the capsule and 2 um spot matrices
python -m kidney.full_depth          # seconds, Supplementary Results 1
python -m kidney.capsule_qc          # about 10 s, Supplementary Fig 4
python -m kidney.capsule_figure      # the capsule overview and split illustration of Fig 1
```

The full-depth control splits all 161 capsules at the median depth. Under Bonferroni, LN's $t$-test calls 8 DEGs,
the log1p $t$-test 1,099 and the Wilcoxon 3,732. It takes seconds and checks the setup before the longer runs.

Spot splitting, 100 replicates for each of 10 values of $p_\mathrm{split}$ from 0.1 to 0.5, about 4 minutes per run on 8
cores:

```bash
python -m kidney.spot_split --n_shape_ids_remove 35 --n_reps 100 --output kidney/results/split_null.pdf
python -m kidney.spot_split --n_shape_ids_remove 35 --q 0.005 --lfc 3 --n_reps 100 \
    --output kidney/results/split_degs.pdf
python -m kidney.fpr_plots --csv_file kidney/results/split_null_results.csv \
    --output kidney/results/split_null_fpr.pdf --split
python -m kidney.spot_split_plots --csv_file kidney/results/split_degs_results.csv \
    --json_file kidney/results/split_degs_pr_curve_data.json --output_prefix kidney/results/split_degs
python -m kidney.split_null_mast     # about 25 minutes, MAST on the first 3 replicates (Extended Data Fig 3a, b)
python -m kidney.split_degs_mast     # about 25 minutes, the same with DEGs (Extended Data Fig 3c, d)
```

UMI downsampling, without DEGs at 10 values of $p_\mathrm{sample}$ from 0.1 to 1 with 10,000 replicates each, with
DEGs at 10 values from 0.125 to 1 with 100 replicates each:

```bash
python -m kidney.umi_null --n_cells_remove 35 --n_reps 10000 --output kidney/results/downsample_null.pdf
python -m kidney.fpr_plots --csv_file kidney/results/downsample_null_results.csv \
    --output kidney/results/downsample_null_fpr.pdf
python -m kidney.umi_de --n_cells_remove 35 --q 0.005 --lfc 3 --p_min 0.125 --n_reps 100 \
    --output kidney/results/downsample_degs_100rep.pdf
python -m kidney.spot_split_plots --downsampling \
    --csv_file kidney/results/downsample_degs_100rep_results.csv \
    --json_file kidney/results/downsample_degs_100rep_pr_curve_data.json \
    --output_prefix kidney/results/downsample_degs_100rep
python -m kidney.null_calibration    # tens of minutes, the calibration of both nulls (Extended Data Fig 3, Supplementary Fig 5)
```

`umi_null` with 10,000 replicates takes several hours on 8 cores.
The parameters are those of the Methods. The 35 capsules with the fewest UMIs are removed. Each gene is a DEG with
probability 0.005, its counts multiplied by $2^3$ in one group. `umi_de` splits the capsules at the median depth before
planting the DEGs and again after downsampling. A capsule near the median can therefore change group between the two
splits, together with its planted fold changes (up to 6 of 126 capsules).

### `lymphnode`: Visium HD lymph node (Fig 2a-c, Supplementary Fig 6)

```bash
bash lymphnode/reproduce_vishd_cluster1_cluster3.sh     # downloads 4.4 GB, several hours
C=lymphnode/config_50rep.yaml
python -m lymphnode.pseudobulk_reference --config $C          # the PyDESeq2 reference, under a minute
python -m lymphnode.depth_imbalance --config $C --thin bins   # about 4 minutes, T-cell spots kept with p_sub: Fig 2b, Supplementary Fig 6a-c
python -m lymphnode.depth_gsea --config $C --thin bins        # about 7 minutes, Fig 2c
python -m lymphnode.depth_imbalance --config $C               # about 5 minutes, T-cell UMIs kept with p_sample: Supplementary Fig 6d, e
python -m lymphnode.depth_gsea --config $C                    # about 10 minutes, Supplementary Fig 6f
```

`lymphnode.subsampling` thins both compartments alike and is outside the paper's figures. The other scripts reuse its
cell and gene filters.

### `clustering`: PBMC3k and PBMC10k (Fig 2d-f, Extended Data Fig 4, Supplementary Fig 7)

```bash
python -m fetch_data --arm clustering
python -m fetch_data --arm citeseq

# PBMC3k
python -m clustering.de --config clustering/config.yaml --output-dir clustering/results --skip-mast
python -m clustering.metrics --config clustering/config.yaml --output-dir clustering/results
python -m clustering.plots --metrics-dir clustering/results/metrics --config clustering/config.yaml

# PBMC10k: the RNA of the 5' PBMC 10k after QC and doublet removal
python -m clustering.pbmc10k_prep
python -m clustering.de --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k --skip-mast
python -m clustering.metrics --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k
python -m clustering.plots --metrics-dir clustering/results/pbmc10k/metrics --config clustering/config_pbmc10k.yaml

# both datasets
python -m clustering.celltype_labels     # CellTypist labels, downloads its model once
python -m clustering.replication
python -m clustering.multires_gaps       # about 15 minutes, Fig 2e, Supplementary Fig 7
python -m clustering.multires_depth      # Supplementary Fig 7
python -m clustering.marker_rule         # Fig 2f, Supplementary Fig 7
python -m clustering.effect_sizes        # Extended Data Fig 4
```

`--skip-mast` leaves out MAST, which takes about 2 hours on PBMC3k and whose results no later script reads.
The other scripts in `clustering/` (`heldout_fp`, `exact_null`, `subcluster_de`, `subcluster_recursive`,
`subcluster_gsea`, `lower_bound`, `concordance`, `multires` and `lfc_se`) are analyses outside the paper's figures.
Each states what it reads and writes in its docstring.

### `published`: the glioma myeloid cell atlas of Abdelfattah et al. 2022 (Fig 2g-i, Extended Data Fig 5, Supplementary Fig 8)

```bash
python -m fetch_data --arm published
python -m published.abdelfattah2022 --rankings   # 1, the gene rankings, a few minutes
Rscript published/R/abdelfattah2022_fgsea.R      # 2, fgsea on every ranking as the original study ran it
Rscript published/R/abdelfattah2022_umap.R       # 3, the myeloid UMAP, several minutes
python -m published.abdelfattah2022              # 4, the readouts and plots
python -m published.abdelfattah2022 --variance   # each cluster's variance ratio at equal mean
python -m published.abdelfattah2022 --claims     # the pathway claims of the original study under each ranking
python -m published.abdelfattah2022 --calls      # each test's Bonferroni calls per cluster
```

### `citeseq`: planted effects in PBMC10k (outside the paper's figures)

```bash
python -m fetch_data --arm citeseq
(cd citeseq/R && Rscript pbmc10k_process.R)      # the protein gating, about 20 s
python -m citeseq.exp                            # 100 replicates
(cd citeseq/R && Rscript pbmc10k_metrics.R)      # the metrics table
python -m citeseq.plots
```

The memory CD4 T cells, `citeseq/data/memory_CD4.h5ad`, are committed.
`(cd citeseq/R && Rscript pbmc10k_to_h5ad.R)` rebuilds them after `pbmc10k_process.R`, with the same cells, genes and
counts but not the same bytes.

### `theory`: illustrations without data

```bash
python -m theory.mean_ci_coverage
python -m theory.lfc_ci_coverage
python -m theory.concave_ordering
python -m theory.log1p_toy
```

These scripts are self-contained and do not use `lntest`, so that the algebra can be read in one file.
`mean_ci_coverage` and `lfc_ci_coverage` import `utils_frozen.py`, a copy of an earlier version of the estimator,
which divides the variance of the positive values by $n_+$ where `lntest` divides by $n_+ - 1$.

## Outputs

Every experiment writes to its own `results/` directory, which git ignores.
DEGs are called at Bonferroni-adjusted $p < 0.05$, with the tie-corrected Wilcoxon, for every method in every experiment.
The exceptions are the lymph node's pseudo-bulk reference, which uses DESeq2's Benjamini-Hochberg convention, and
`clustering.heldout_fp`, which also counts Benjamini-Hochberg and uncorrected calls. `lntest` itself defaults to
Benjamini-Hochberg, as scanpy does.

Every output is byte-reproducible except wall-clock timings. Figures carry no timestamp (`paths.py` sets
`SOURCE_DATE_EPOCH`). The jitter of strip plots is seeded. Results computed in parallel are sorted before they are
written.

## Known limitations

- The tests in `../tests/` cover the `lntest` package. No test checks that an experiment reproduces the numbers the
  paper prints.
- Supplementary Figs 1-3 have no generating script.
