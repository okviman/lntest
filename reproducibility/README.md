# Reproducing the results

Everything in the Nature submission that depends on code lives here, one directory per experiment.
This file is the reproduction instruction; the top-level `README.md` covers the `lntest` package itself.

Read [Known gaps](#known-gaps) before trusting any number you produce.

## Run everything from this directory

```bash
cd reproducibility
python -m <arm>.<script>          # e.g. python -m synthetic_nb.small_test
```

Not `python synthetic_nb/small_test.py`.
The `-m` form puts the **working directory** on `sys.path`, so a script inside an arm can import the shared modules that sit beside the arms — `paths`, `method_colors`, `plot_style`, `baselines`, `de_utils`, `evaluation_utils`, `utils_frozen` — with no `__init__.py`, no packaging file for `reproducibility/`, and no `sys.path` manipulation.
Running a script by path puts *its own directory* on `sys.path` instead, and the shared imports fail.

**Notebooks are the exception.** Run a notebook from the arm directory that contains it; each does `sys.path.append('../')` in its first cell to reach the same shared modules.

## The environment

```bash
conda env create -f env.yaml
conda activate de-ziln-reproducibility
pip install -e ..                 # installs the lntest package from ../src/
export R_HOME="$CONDA_PREFIX/lib/R"
```

**`R_HOME` is not optional.** Without it, `import rpy2.robjects` aborts the interpreter with

```
Error in substring(x, m + 1L) : invalid substring arguments
```

and exit code 139, *before any of your code runs*, because `clustering/de.py`, `de_utils.py` and `lymphnode/subsampling.py` all import rpy2 at module scope.
`--skip-mast` does not help: the import happens either way.
This affects the clustering, celltype and lymph node arms. R itself is fine — only rpy2's initialisation of it needs the variable.

**Figures are written as PDF.** Every plotting entry point in this tree routes its filename
through `paths.fig_name`, which forces the extension. Set `LNTEST_FIG_FORMAT` to override for a
run (`LNTEST_FIG_FORMAT=png python -m theory.concave_ordering`).

**Every figure names, orders and sizes things the same way.**
`method_colors.py` holds each method's colour, its display name and its position:
LN's $t$-test, log1p $t$-test, Wilcoxon, MAST, listed in legends in that order whatever label an
arm's results use ("DELN", "LN test", "Scanpy t-test (log1p)", ...). Where methods overlap,
LN's $t$-test is drawn last, so its data sits on top.
`plot_style.py` holds the text sizes: 8 pt labels, ticks and titles, 6 pt legends, at a panel of
about 2.6 x 2.2 in, so each figure is drawn at roughly its printed size and its text matches the
others when included unscaled. One font (matplotlib's default sans) and no bold titles; means
with error bars share one marker and bar weight (`plot_style.ERRORBAR`).
Axes have tick marks, light grid lines (horizontal only on box plots, whose x-axis is categorical)
and no top or right spine; an axes with more than 5,000 dots has them rasterized
(`plot_style.rasterize_dense`), everything else stays vector. Legends have no frame, sit
outside the axes (`plot_style.legend_outside`), appear once per figure however many panels it
has, and carry a title only when a second key needs telling apart.
Box plots use the method colours with plain black medians, show every observation as a point
(`plot_style.strip`) and draw no outline around the boxes; axes say FPR, TPR and DEGs.
**No figure is drawn in R**: the R scripts write what they would have plotted to CSV, and
`citeseq/plots.py` draws it.

The tree used to emit a mix of `.eps` and `.png`, and four scripts hardcoded `format='eps'` in the
`savefig` call, so `--output x.png` produced PostScript named `.png`. Those are gone; the filename
now decides. PDF is also markedly smaller for these plots — `null_fpr_by_var_mu5` (then `variance_vs_fpr_mu5`) went from 460 KB
raster to 17 KB vector.

**Two consequences worth knowing before regenerating anything.** The 22 figure entries in
`reference/clustering/checksums.sha256` name `.png` files and cannot match PDF output; they were
already due for regeneration after the single-trigamma change, and 10 of them were never
verifiable anyway. And any `\includegraphics` in the manuscripts that spells out `.png` or `.eps`
needs updating, or the extension dropped so LaTeX picks what exists.

**`pandas<3` is a hard pin, not caution.** pandas 3.0 makes the arrow-backed `str` dtype the
default for string columns and indices, and anndata 0.12.6 has no writer registered for it, so
`adata.write()` fails on *any* AnnData:

```python
import numpy as np, anndata as ad
ad.AnnData(np.ones((3, 2))).write("probe.h5ad")   # IORegistryError
```

Reading is unaffected, so an arm loads its input, computes everything, and dies on the last line.
`celltype/de.py` does exactly that: it finishes LN, t-test, Wilcoxon and MAST, then fails on its save.
`clustering/de.py` is the one arm that survives pandas 3, because it carries its own
`coerce_arrow_strings()` helper (added in `c7c5d94`, the commit behind the reference checksums) and
calls it immediately before writing. Do not read that as the problem being solved — it is solved in
one file. If you built this environment before the pin was added,
`conda install -n de-ziln-reproducibility 'pandas<3'`.

**If your environment predates 2026-09-10**, update it: `geopandas`, `shapely` and `pyarrow` were missing, so neither spatial arm's preprocessing could run.

```bash
conda env update -f env.yaml --prune
```

## What you can run right now

Three data files are committed for the live arms — `citeseq/data/memory_CD4.h5ad` and the two
`nullsplit/data/split_*.csv` — plus five under `synthetic_nb/legacy/data/`. Everything else is
fetched, downloaded or generated.

| Arm | Input | In a fresh clone? |
|---|---|---|
| `synthetic_nb` | none — generates its own | **yes** |
| `theory` | none | **yes** |
| `nullsplit` | `nullsplit/data/split_{a,b}.csv` (committed) | **yes** |
| `citeseq` | `citeseq/data/memory_CD4.h5ad` (committed, 16 MB) | **yes** — also rebuildable from a download, see below |
| `clustering` | `data/pbmc3k_filtered_gene_bc_matrices.tar.gz`; for `multires_gaps`, `data/GSE107011_Processed_data_TPM.txt.gz` (Monaco et al. 2019 sorted-cell bulk, 15 MB); for `marker_rule`, `data/monaco2019_supplementary.zip` (their supplementary tables, 55 MB) and `data/hpa_25.1_proteinatlas.tsv.zip` (Human Protein Atlas 25.1, 7 MB) | **yes** — `python -m fetch_data --arm clustering`, all pinned by sha256 |
| `celltype` | `data/pbmcs3k_pre.h5ad`, `data/kang_2018.h5ad` | **yes** — `fetch_data --arm celltype`, then `python -m celltype.preprocess` |
| `kidney` | `kidney/data/{merged_blobs_in_cluster_5,podocytes_2um}.h5ad` | **download yes** — `fetch_data --arm kidney` (6.5 GB), then `preprocess.ipynb` |
| `lymphnode` | `lymphnode/data/vishd-cluster1-cluster3-2um-with-clusters-subsampled.h5ad` | **download yes** — `fetch_data --arm lymphnode` (4.7 GB), then the reproduce script (hours) |
| `published` | `data/GSE182109_RAW.tar`, `data/Meta_Data_GBMatlas.txt` | **download yes** — `fetch_data --arm published` (2.5 GB), both pinned by sha256 |

`data/` and `*/data/` are gitignored, so a file being present in one working copy says nothing about a fresh clone.
Every entry point checks its input first and exits naming the missing path and where it comes from, rather than failing part-way through.

Fetching them:

```bash
python -m fetch_data                  # everything missing, verified against sha256
python -m fetch_data --arm clustering # just what one arm needs
python -m fetch_data --list           # what is present, what is not
```

`data_sources.yaml` carries a URL for every fetched input — all seven, spatial included — and
`fetch_data.py` downloads what is missing, verifies it, and unpacks the tarballs.

Verification is two-tier. Four entries are pinned by **sha256**. The three multi-GB spatial archives
carry a **byte count only**, because computing a digest for them means downloading 11 GB first; those
report as `SIZE-OK` rather than `OK`, which catches truncation and nothing else. Pin them properly by
hashing after a download and filling in the `sha256:` field. It is idempotent, and it
refuses to overwrite a file whose digest does not match rather than silently replacing it.

**CITE-seq rebuilds from its download too**, as of 2026-09-18:

```bash
python -m fetch_data --arm citeseq          # 30 MB raw 5' PBMC 10k matrix
cd citeseq/R && Rscript pbmc10k_process.R   # ~17 s -> results/pbmc10k_cd4_memory.rds, adt_gating.csv
Rscript pbmc10k_to_h5ad.R                   # ~2 min -> data/memory_CD4.h5ad
```

Verified: the rebuilt `memory_CD4.h5ad` matches the committed one exactly — 1318 x 2035, same cells,
same genes, `max|dX| = 0`. That also settled which raw file is the source: 10x serves two under the
same name, and only `cell-vdj/5.0.0` reproduces it, so the entry is now sha256-pinned.

`memory_CD4.h5ad` stays committed for convenience — the rebuild takes ~2 minutes and needs the R
stack — but it is no longer the only way to get it. Note the rebuilt file is *content*-identical and
not *byte*-identical: HDF5 containers differ in compression and metadata, so `git status` will show
it as modified after a rebuild even when nothing in the data changed.

One thing this does **not** do: **no Zenodo DOI exists**, so the non-derivable intermediates named in
the vault's `decision-data-out-of-head` have nowhere to be fetched from.

Verified 2026-09-18: deleting `data/pbmc3k*` and re-running `python -m fetch_data --arm clustering`
downloads, verifies and unpacks it, after which `clustering.de` and `celltype.preprocess` both run
off the fetched matrix.

`data/pbmcs3k_pre.h5ad` is the pbmc3k matrix after standard QC; `clustering/legacy/resolution_sweep.ipynb` writes it, and `clustering/de.py` applies the same QC inline when handed the raw `.tar.gz`.

## The arms

### `synthetic_nb` — negative-binomial simulations

```bash
python -m synthetic_nb.small_test      # ~10 s. The reviewer-facing single run
python -m synthetic_nb.de_test         # ~3 min; sparse genes -> synthetic_nb/results/nde_mu10/
python -m synthetic_nb.de_test --setting dense       # -> synthetic_nb/results/nde_mu100/
python -m synthetic_nb.de_ratio        # both settings: the five metrics against Var(X)/Var(Y)
python -m synthetic_nb.latex_tables    # formats the sparse CSV as a LaTeX table
python -m synthetic_nb.null            # ~40 s; base_mu 50 by default
python -m synthetic_nb.null --base-mu 5
python -m synthetic_nb.null --base-mu 5 --plot-only   # redraw from the sweep's CSV
python -m synthetic_nb.null_ratio      # both sweeps against Var(Y)/Var(X), side by side, narrower and stacked
python -m synthetic_nb.lfc_confidence_intervals
```

`null.py` samples a grid of variance ratios by default: Var(X) is 1, 2, 4 and 8 times the mean (1 is Poisson), and Var(Y)/Var(X) runs from 1 to 64 in quarter-octave steps; the ratios past 16 are drawn after the rest, so adding them left every earlier value as it was.
`null_ratio`'s axis ends where every log1p and every Wilcoxon curve has reached FPR 0.99.
On it the null runs LN's t-test, the log1p t-test and Wilcoxon, each Bonferroni-corrected and the Wilcoxon tie-corrected, as everywhere; the dispersion grid below keeps RECOMB's two tests.
Ratios below 1 would repeat the same settings with the groups swapped: every test is two-sided and the groups are the same size.
`--grid dispersion` is the grid behind RECOMB Fig 1 — the same 20 values of Var(Y) for every Var(X) — and writes `null_fpr_by_var_mu{5,50}.*`; the default ratio grid
writes `null_fpr_by_ratio_mu{5,50}.*`, which `null_ratio` combines into `null_fpr.pdf` (panels side by side),
`null_fpr_narrow.pdf` (side by side, 30% narrower, the same text sizes) and `null_fpr_stacked.pdf` (one above the other).
`synthetic_nb/results/checksums.sha256` checksums both grids' outputs.

`small_test.py` reproduces **one** run of a table averaged over 20; its numbers are not expected to match the paper exactly.
`de_test.py`'s `sparse` setting has RECOMB Table 2's design and its `dense` setting Table 1's, without the batch effect both tables were run with and the paper does not describe: the first half of each group's cells had every mean multiplied by e.
So this script does not reproduce the printed numbers; the git history does, with Bonferroni: Table 2 from `large_scale_NB_DE_test.py` at commit `2ed26e3`, and Table 1 from the same script at `1bbcb0a` with 10,000 cells per group.
The NB parameters behind the paper's table are not captured in any config: the submission said they "need to be adjusted according to the text", and nothing here records what they were.

### `citeseq` — CITE-seq, surface protein as ground truth

```bash
python -m citeseq.exp                  # writes to citeseq/results/
cd citeseq/R && Rscript pbmc10k_metrics.R && cd ../..   # the metrics table, CITE_seq_metrics.tex
python -m citeseq.plots                # every CITE-seq figure, from the CSVs above and the R chain's
```

`degs_metrics.pdf` shows the metrics of the table (RECOMB Table 3: accuracy, precision, TPR, TNR, F1),
and FPR, as box plots, one point per replicate. `degs_summary.pdf` has the kidney with-DEG summary's panels for
this experiment's one balanced condition: TPR and FPR at Bonferroni 0.05, PR-AUC from the ranking by adjusted
p-value, and the mean precision-recall curves, computed as `kidney/spot_split.py` computes them.

The R stages that build `memory_CD4.h5ad` from the raw 10x download are in `citeseq/R/`, run with `Rscript`.
The committed `memory_CD4.h5ad` means you do not need them unless you are rebuilding from raw.
`citeseq/R/pbmc10k_seurat.R` generates Seurat LFC estimates that **were not used in the paper**, because they did not affect the conclusion; it is kept for provenance.
It writes `seurat_lfc_results.csv`, and `citeseq.plots` draws the Seurat figures only when that file exists.
Until 2026-09-26 `exp.py` wrote the replicates for it truncated to integers, so Seurat saw different counts from the other methods; on the same counts its `avg_log2FC`, the log of the mean normalized expression, tracks the true LFC as closely as LN's.

### `clustering` — PBMC3k, clustering resolution sweep

```bash
python -m clustering.de      --config clustering/config.yaml --output-dir clustering/results

# metrics and plots do NOT read --output-dir to find their input. Point them at
# what the previous stage wrote, or they fail / silently do nothing. See below.
sed 's|^adata_path:.*|adata_path: "clustering/results/pbmc3k_filtered_gene_bc_matrices_DE.h5ad"|' \
    clustering/config.yaml > .metrics_config.yaml          # gitignored, see .gitignore
python -m clustering.metrics --config .metrics_config.yaml --output-dir clustering/results
python -m clustering.plots   --metrics-dir clustering/results/metrics --config clustering/config.yaml
python -m clustering.heldout_fp --config clustering/config.yaml   # about a minute; false positives on held-out negatives
# the same on a second, deeper dataset: PBMC10k's RNA from the CITE-seq arm's input (python -m fetch_data --arm citeseq),
# with the doublets scanpy's Scrublet predicts removed; under a minute
python -m clustering.pbmc10k_prep
python -m clustering.de --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k --skip-mast   # minutes
sed 's|^adata_path:.*|adata_path: "clustering/results/pbmc10k/pbmc10k_rna_DE.h5ad"|' \
    clustering/config_pbmc10k.yaml > .metrics_config.yaml
python -m clustering.metrics --config .metrics_config.yaml --output-dir clustering/results/pbmc10k
python -m clustering.plots   --metrics-dir clustering/results/pbmc10k/metrics --config clustering/config_pbmc10k.yaml
python -m clustering.heldout_fp --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k    # tens of minutes
# exact null: random halves of each leiden_mid cluster, one half thinned; seconds per dataset
python -m clustering.exact_null --config clustering/config.yaml --n-workers 4
python -m clustering.exact_null --config clustering/config_pbmc10k.yaml --output-dir clustering/results/pbmc10k --n-workers 4
# cell-type labels for both datasets (CellTypist, downloads its model once), PBMC10k's checked against its antibodies
python -m clustering.celltype_labels
# each method's markers of those types in both datasets, their concordance, and the effect transfer; seconds
python -m clustering.replication
# each type of >= 200 cells subclustered, each subcluster tested against the rest of its type; under a minute
python -m clustering.subcluster_de
# each type of >= 400 cells split recursively to five levels, each child tested against its siblings; under a minute
python -m clustering.subcluster_recursive
# LN's markers ranked by the lower bound of the fold change, against the score; under a minute
python -m clustering.lower_bound
# each method's concordance between the datasets: markers, score rankings, scores and GSEA enrichments; minutes
python -m clustering.concordance
# each Leiden cluster annotated from each method's top-20 markers, across the five resolutions; seconds
python -m clustering.multires
# each subcluster's Hallmark GSEA against its siblings, as observed and with the two sides' depth matched; minutes
python -m clustering.subcluster_gsea
# across the five resolutions: each comparison's variance and depth gaps, each method's calls, top-20 markers and GSEA,
# judged against sorted-cell bulk and the curated markers, and their stability between resolutions; about 15 minutes
python -m clustering.multires_gaps
# LN's LFC and SE: split-half calibration, carry-over between the datasets by SE, the detection-level decomposition,
# and the concordance of LN's markers (its calls ranked by the detection score); a few minutes
python -m clustering.lfc_se
# every cluster at the five resolutions: each method's median score against the cluster's depth ratio, with its
# least-squares line and Pearson r; about half a minute
python -m clustering.multires_depth
# LN's markers against its score and the baselines: housekeeping genes, HPA 25.1, sorted bulk and PBMC10k's surface
# protein, for the 6 types and the five resolutions; about a minute
python -m clustering.marker_rule
# PBMC10k and PBMC3k: each cluster's mean |LFC| over its significant genes as each package reports it (LN's estimate, Scanpy's
# for the log1p t-test, Seurat's avg_log2FC for the Wilcoxon) and the log1p t-test's difference of log1p means; seconds
python -m clustering.effect_sizes
```

Run them in that order, and **pass `--skip-mast`** (it still needs `R_HOME` — the flag does not avoid
the rpy2 import). Measured 2026-09-10:

| | DE stage | Reference metric CSVs | Reference figures |
|---|---|---|---|
| with MAST | **2 h** (63 fits, one per cluster; the 30-cluster resolution is most of it) | 68/68 byte-identical | 12/22 |
| `--skip-mast` | **15 s** | 68/68 byte-identical | 12/22 |

The two are **equivalent for every verifiable output**, because nothing downstream reads the MAST
results. `metrics.py` hardcodes `de_methods = ["ln", "t_test", "wilcoxon"]` and neither it nor
`plots.py` mentions MAST anywhere, so the 5 `mast_*` keys the 2-hour stage writes into the `_DE.h5ad`
are stored and never consumed. Run with MAST only if you need those keys for something outside this
tree — they are not deleted here, because whether the manuscript leans on them is an open question.

This is the arm with reference checksums: see [Outputs](#outputs-and-reference-values).
Verified end to end on 2026-09-10 — **all 68 metric CSVs byte-identical to `reference/`**.

Two wiring defects to know about, both found by running it:

- **`metrics` re-reads `config['adata_path']` and opens it with `read_h5ad`.** It does not read the
  `_DE.h5ad` that `de` just wrote into `--output-dir`. With the committed config, which names the
  `.tar.gz`, it dies with `OSError: file signature not found`. Hence the `sed` above.
- **`clustering/plots.py` silently drops four figures without `--config`.** Line 75 reads
  `top_genes = config.get('top_genes', 10)` while `clustering/config.yaml` sets `top_genes: 20`,
  so with no config the script looks for `avg_jaccard_top10_by_resolution.csv`, finds nothing, and
  its `boxplot()` helper returns early. You get 18 figures instead of 22, with no message. The
  heatmaps are unaffected because they `glob` rather than build the name, which is why the `top20`
  heatmaps appear while the `top20` boxplots do not. One of the four,
  `avg_jaccard_top20_by_resolution`, is a panel in the Nature submission. Always pass
  `--config clustering/config.yaml`.
- **`plots --metrics-dir` wants the `metrics` directory itself**, not the output root — its `--help`
  says `e.g. clustering/results/metrics`. Given the output root it finds no CSVs, prints
  `(No recall_matrix_*.csv files found; skipping ...)`, reports `Plots complete!`, **exits 0 and
  writes nothing**. Figures default to a `figures/` sibling of `--metrics-dir`, which is where the
  reference manifest expects them.

**Ten of the 22 figures cannot be byte-reproduced by anyone**, including whoever made the reference.
The `*_by_resolution.png` boxplots overlay `sns.stripplot`, whose jitter is drawn from the unseeded
global RNG, so two consecutive runs of the same script on identical CSVs differ. Confirmed by running
it twice: the 12 heatmaps are byte-identical both to each other and to `reference/`; the same 10
boxplots differ every time. Their reference hashes are therefore not verifiable — judge those panels
by the metric CSVs underneath them, which do match exactly.

### `celltype` — Kang 2018 cell types

```bash
python -m celltype.de --config celltype/config.yaml --output-dir celltype/results
```

`celltype/kang_markers.ipynb` produces the committed marker CSVs in `celltype/results/`.

### `kidney` — Visium HD, glomerular capsules

Preprocessing first, either form -- they do the same work:

```bash
python -m kidney.preprocess      # from reproducibility/
```

or open `kidney/preprocess.ipynb` **from `reproducibility/kidney/`** and run it top to bottom.
Either downloads 6.5 GB from 10x and writes both matrices into `kidney/data/`.
Took ~9 min on an M-series laptop; produces 161 capsules over 18,085 genes, which is what
`main_recomb25.tex` reports for this dataset.

```bash
python -m kidney.umi_null --n_cells_remove 35 --n_reps 10000 --output kidney/results/downsample_null.pdf
python -m kidney.umi_de   --n_cells_remove 35 --q 0.005 --lfc 3 --p_min 0.125 --n_reps 10000 \
    --output kidney/results/downsample_degs_10000rep.pdf
python -m kidney.fpr_plots --csv_file kidney/results/downsample_null_results.csv \
    --output kidney/results/downsample_null_fpr.pdf
python -m kidney.umi_de_plots --input kidney/results/downsample_degs_10000rep_results.csv \
    --output kidney/results/downsample_degs_10000rep_tpr_fpr.pdf
```

`umi_de` computes the same DE metrics as `spot_split` (it calls its `de_test_single`), PR-AUC and precision-recall
curves included, so `spot_split_plots --downsampling` draws the same with-DEG figures for it. A 100-replicate run,
beside the published 10,000-replicate one:

```bash
python -m kidney.umi_de --n_cells_remove 35 --q 0.005 --lfc 3 --p_min 0.125 --n_reps 100 \
    --output kidney/results/downsample_degs_100rep.pdf
python -m kidney.spot_split_plots --downsampling \
    --csv_file kidney/results/downsample_degs_100rep_results.csv \
    --json_file kidney/results/downsample_degs_100rep_pr_curve_data.json \
    --output_prefix kidney/results/downsample_degs_100rep
```

`umi_de` splits the capsules at the median depth twice, before planting the DEGs and again after downsampling,
so a capsule near the median can change group between the two (up to 6 of 126, in 38-70% of replicates) and
take its planted fold changes with it. Known, not fixed.

Files in `kidney/results/` are named `<experiment>_<condition>_<content>`: `split` (spot splitting) or `downsample`
(UMI downsampling), `null` (no DEGs) or `degs` (planted DEGs), with the replicate count where two runs of the same
experiment sit side by side. They were renamed on 2026-09-27; the vault's figure-style decision maps the old names.

Spot splitting, behind the two kidney panels both manuscripts show (`split_null_fpr`, `split_degs_summary`;
the manuscripts still include them under their old names, `2um_nodeg_fpr.eps` and
`withdeg_2um_100rep_with_auc_lfc3_pretty_plot_summary.eps`). About 4 minutes per run on a 16 GB laptop, 8 workers:

```bash
python -m kidney.spot_split --n_shape_ids_remove 35 --n_reps 100 --output kidney/results/split_null.pdf
python -m kidney.spot_split --n_shape_ids_remove 35 --q 0.005 --lfc 3 --n_reps 100 \
    --output kidney/results/split_degs.pdf
python -m kidney.fpr_plots --csv_file kidney/results/split_null_results.csv \
    --output kidney/results/split_null_fpr.pdf --split
python -m kidney.spot_split_plots \
    --csv_file kidney/results/split_degs_results.csv \
    --json_file kidney/results/split_degs_pr_curve_data.json \
    --output_prefix kidney/results/split_degs
```

`spot_split` draws no figure; its `--output` only names the results files. The no-DEG run also writes
`split_null_per_split.csv`, one row per split and method: the seed, the genes tested and the DEGs called. The FPR panels plot the mean over replicates on a linear axis, with bars of one SD; `fpr_plots` also
writes a `_log.pdf` of each on a log axis, where zero gets its own row below a break instead of a floor.
The FPR panels of `downsample_degs_*_tpr_fpr.pdf`, `split_degs_tpr_fpr.pdf` and the `*_summary.pdf` figures use that axis too.
Every figure with p on its x-axis also comes as `*_odds.pdf`, plotted against $(1-p)/p$, which grows as the test gets
harder: for $p_\mathrm{split}$ it is the ratio of the two groups' sampling variances, as Var(Y)/Var(X) in the synthetic NB
figures. `spot_split_plots` also writes `*_pr_auc.pdf`, the summary's PR-AUC panel alone; `*_pr_curves.pdf`
is its PR-curve panel alone.

The capsule figures — every capsule outlined on the H&E (`capsule_overview.pdf`), and one capsule's
2 µm spots split at $p_\mathrm{split}$ = 0.5, 0.3 and 0.1 on the full-resolution image, group A in blue and B in
orange (`capsule_split_illustration.pdf`) — replace RECOMB's hand-composited `podocytes.png`. They need the 4.3 GB full-resolution H&E (`kidney-he-fullres` in `data_sources.yaml`):

```bash
python -m fetch_data --arm kidney          # includes the full-resolution H&E
python -m kidney.capsule_figure            # ~3 s -> kidney/results/capsule_overview.pdf, capsule_split_illustration.pdf
```

Calibration from uncorrected p-values on both null experiments, 100 replicates per $p$, for the calibration Extended
Data figure:

```bash
python -m kidney.null_calibration              # tens of minutes on all cores -> kidney/results/null_calibration.csv
python -m kidney.null_calibration --plot-only  # split_null_calibration.pdf, downsample_null_calibration.pdf
```

MAST on the no-DEG spot-splitting null, for Extended Data Fig 4a: Fig 1f's first 3 splits of each $p_\mathrm{split}$,
rebuilt from the seeds in `split_null_per_split.csv`, the other three tests' FPR taken from that CSV over the same splits.
About 3 minutes per MAST fit on one thread, four at a time (each takes about 1.5 GB); needs `R_HOME`; a rerun keeps the
splits already in the CSV:

```bash
python -m kidney.split_null_mast              # about 25 minutes -> kidney/results/split_null_mast_per_split.csv
python -m kidney.split_null_mast --plot-only  # split_null_mast_fpr.pdf
```

**These parameters come from the manuscript, not from the code.** `--n_cells_remove`, `--q` and
`--lfc` are `required=True` with no defaults, so nothing in this repository records what was run;
until 2026-09-24 the block above showed `50 / 0.1 / 1.0`, which were placeholders and matched no
published figure. The values now shown are what `main_recomb25.tex` states:

| Flag | Value | Source |
|---|---|---|
| `--n_cells_remove` | 35 | "we first filtered out the 35 capsules with the lowest total UMI" (§ Visium HD); the published panels were named `fdr_results_35.eps` and `DE_plot_lfc3.eps` |
| `--q` | 0.005 | "With probability 0.005, we sample DEGs ... around 90 DEGs" |
| `--lfc` | 3 | "amplified to ensure a three-fold $\log_2$ change, i.e. LFC=3" |
| `--n_reps` | 10000 | "For each downsampling ratio $p$, 10,000 random tests were ran" |
| `--p_min` | 0.125 | "$p\ge 2^{-3}$" -- a **derived** bound, see below |

`p_min` is not free: the manuscript ties it to `lfc` as $p \ge 2^{-\mathrm{LFC}}$, because a gene
amplified by LFC log2-units cannot survive downsampling below that ratio. **No code enforces this.**
Change `--lfc` and you must change `--p_min` with it; the script defaults (`p_min 0.1`) satisfy the
constraint for LFC=3 only by being close to 0.125, and violate it for any LFC below 3.

`spot_split`'s capsule filter has no published value: the spot-subsampling section states no
filtering. The commands above drop the same 35 smallest capsules as the UMI arms (`--n_shape_ids_remove 35`,
since 2026-09-27). With all 161, the smallest capsules (186 to 450 UMIs) give semi-capsules in which one
count is worth about 100 times a typical CP10k value, and LN's $t$-test then calls a false DEG in 34-100%
of no-DEG splits; without them, in 2-9%. The all-161 outputs are in `kidney/results/archive/all_capsules_2026-09-27/`.
Its other defaults, `p_min 0.1 / p_max 0.5 / p_steps 10`, match the published panels, whose
PR curves are drawn at p = 0.100, 0.322 and 0.500.

`spot_split` keeps the 2 µm matrix sparse. It used to densify it — 546K spots x 18K genes, ~40 GB as
float32 — and copy it into shared memory, which only a server could hold; the sums per semi-capsule
are the same numbers from the same random draws. Its per-replicate seeds now come from `--seed`
(default 0) instead of an unseeded RNG, and a re-run is byte-identical.

**Full-depth negative control.** `python -m kidney.full_depth` (seconds) splits all 161 capsules at the
median library size, with no downsampling and no capsule removed, and writes the DEGs each method calls
under BH and under Bonferroni to `full_depth_null.csv`, plotting the Bonferroni counts in `full_depth_null.pdf`.
It repeats the split after removing the 20 and the 35 lowest-UMI capsules, and `full_depth_null_by_removal.pdf`
puts the three capsule sets side by side (Bonferroni, LN / log1p / Wilcoxon: 10 / 1,099 / 3,732, then 1 / 26 / 107, then 0 / 0 / 2).
`python -m kidney.capsule_qc` (about 10 s) shows why the other kidney runs drop the 35 capsules with the lowest
total UMI, the reason `main_recomb25.tex:751` gives: `capsule_qc.pdf` has the capsule-size distribution, split at
the median into the shallower and the deeper half, and the DEGs each method calls on this split as the smallest capsules are removed one by one
(`capsule_removal_scan.csv`). Under Bonferroni LN calls none from 22 removed and the log1p t-test none from 33, while the
tie-corrected Wilcoxon still calls 2 at 35, the end of the scan. The 35 was chosen when the scan was BH with the untied
Wilcoxon, under which every method called none from 33.
The manuscript's numbers are the Bonferroni ones, which the current defaults reproduce for LN and the log1p t-test;
the manuscript's Wilcoxon had no tie correction (under BH: LN 627, log1p 7,130, tie-corrected Wilcoxon 12,171, of 15,216 genes):

| Method | Here | `main_recomb25.tex` |
|---|---|---|
| LN's $t$-test | 10 | 10 |
| log1p $t$-test | 1099 | 1099 |
| Wilcoxon | 3732 (2878 without tie correction) | 2876 |

Two exact. Without tie correction Wilcoxon gives 2,878, +2 of 2,876 (0.07%), most likely rank-tie handling across scipy
versions; the tie-corrected test, the default since 2026-09-30, calls 3,732.
Takes seconds, and is worth running before committing to a 10,000-replicate sweep.

The two UMI scripts read `merged_blobs_in_cluster_5.h5ad` (one row per capsule); `spot_split` reads `podocytes_2um.h5ad` (one row per 2 µm spot, with a `shape_id` column). They are not interchangeable — see [Known gaps](#known-gaps).

### `lymphnode` — Visium HD, Cluster-1 vs Cluster-3

Preprocessing is a driver script; read its header first, it downloads 4.4 GB and the middle step is hours of single-threaded point-in-polygon:

```bash
bash lymphnode/reproduce_vishd_cluster1_cluster3.sh
```

Then:

```bash
python -m lymphnode.pseudobulk_reference --config lymphnode/config_50rep.yaml   # ~10 s: the reference every method is scored against
python -m lymphnode.subsampling --config lymphnode/config_50rep.yaml --output-dir lymphnode/results --skip-mast
python -m lymphnode.subsampling --config lymphnode/config_50rep.yaml --output-dir lymphnode/results --plot-only  # redraw figures from the CSV
python -m lymphnode.ln_de_vs_rest   --input <h5ad> --output de.csv
python -m lymphnode.cluster_de_gsea --input <h5ad> --output_de de.csv --output_gsea gsea.csv
# the T-cell zone's reads thinned (p 0.2-0.9, every spot kept), the follicles whole: recovery and false calls; ~5 minutes
python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml
# the same conditions (10 replicates): each test's Hallmark GSEA of B over T against the pseudo-bulk reference's; ~10 minutes
python -m lymphnode.depth_gsea --config lymphnode/config_50rep.yaml
# the spot version: the T-cell zone's 2 um spots subsampled (p 0.4-0.9) above a filter floor; ~4 and ~7 minutes
python -m lymphnode.depth_imbalance --config lymphnode/config_50rep.yaml --thin bins
python -m lymphnode.depth_gsea --config lymphnode/config_50rep.yaml --thin bins
```

### `published` — a published study's pathway figure, rerun

Abdelfattah et al. 2022 (Nat Commun 13:767): their glioma myeloid clusters, their Fig. 4d Hallmark enrichments under the
authors' procedure (genes ranked by the Wilcoxon AUC, fgsea on msigdbr's Hallmark sets, as their code ran it) and under
LN, against each cluster's depth relative to the other
myeloid cells, at the level of the gene ranking and of the enrichments; and a UMAP of the myeloid cells as the paper
describes its embedding (Seurat and harmony in R, `published/R/abdelfattah2022_umap.R`), by cluster and by depth.

```bash
python -m fetch_data --arm published           # 2.5 GB: GEO GSE182109 and the authors' per-cell metadata
python -m published.abdelfattah2022 --rankings # 1, the gene rankings and per-cluster tables; a few minutes
Rscript published/R/abdelfattah2022_fgsea.R    # 2, the authors' fgsea on every ranking (msigdbr 7.5.1); a minute
Rscript published/R/abdelfattah2022_umap.R     # 3, the myeloid UMAP (Seurat, harmony); several minutes
python -m published.abdelfattah2022            # 4, the readouts and plots; seconds
python -m published.abdelfattah2022 --claims   # the paper's Fig. 4d claims under the AUC, the Wilcoxon z and LN; seconds
python -m published.abdelfattah2022 --calls    # each test's Bonferroni calls per cluster and the share up; a few minutes
```

Each plot is its own file in `published/results/` (`abdelfattah_*.pdf`), drawn by the module's `draw_*` functions; `figures.ed_abdelfattah` places them as an Extended Data figure.
The CSVs and plots are listed in `published/results/checksums.sha256`, byte-identical across two runs.

### `theory` — illustrations, no data

```bash
python -m theory.mean_ci_coverage    # ~1 s
python -m theory.lfc_ci_coverage
python -m theory.concave_ordering    # ~1 s
python -m theory.log1p_toy           # ~1 s; equal means, unequal log1p means (was notebooks/toy.ipynb)
```

These are exempt from depending on `lntest` — being readable in one file, with the algebra inline, matters more here.
The exemption is on the *code*, not the maths: where the manuscript prints a formula, that formula is the specification.

### `figures` — Fig 1, Fig 2 and the Extended Data figures, assembled

`figures/results/paper/` holds the twelve figures the paper prints (Fig 1, Fig 2, Extended Data Figs 1-10),
`figures/results/other/` the assembled figures it does not, and `figures/results/panels/` the parts every one of them
is assembled from.

```bash
python -m figures.fig1     # a few seconds -> figures/results/paper/fig1.pdf and its panels fig1_{c..h}_*.pdf
python -m figures.fig2     # a few seconds -> figures/results/paper/fig2.pdf and its panels fig2_{b,c,de,f..i}_*.pdf (a: the lymph node images)
python -m figures.ed1      # a few seconds -> figures/results/paper/ed1.pdf and its panels ed1_{a..d}_*.pdf
python -m figures.ed2      # seconds -> figures/results/paper/ed2.pdf and its panels ed2_{a..c}_*.pdf
python -m figures.ed3      # a few seconds -> figures/results/paper/ed3.pdf and its panels ed3_{a,b}_*.pdf
python -m figures.ed4      # a few seconds -> figures/results/paper/ed4.pdf and its panels ed4_{a..d}_*.pdf
python -m figures.ed5      # a few seconds -> figures/results/paper/ed5.pdf and its panels ed5_{a..e}_*.pdf
python -m figures.ed6      # a few seconds -> figures/results/paper/ed6.pdf and its panels ed6_{a..f}_*.pdf
python -m figures.ed7      # seconds -> figures/results/paper/ed7.pdf and its panels ed7_{a..g}_*.pdf
python -m figures.ed8      # seconds -> figures/results/paper/ed8.pdf and its panels ed8_{a..d}_*.pdf
python -m figures.ed9      # seconds -> figures/results/paper/ed9.pdf and its panels ed9_{a..h}_*.pdf
python -m figures.ed10     # seconds -> figures/results/paper/ed10.pdf and its panels ed10_{a..e}*_*.pdf
python -m figures.ed7 --pbmc10k-only   # seconds -> figures/results/other/ed7_pbmc10k.pdf, Extended Data Fig 7 without PBMC3k
python -m figures.ed_abdelfattah   # seconds -> figures/results/other/ed_abdelfattah.pdf (the authors' Wilcoxon AUC against LN, every panel), ed_abdelfattah_wilcoxon_z.pdf and ed_abdelfattah_log1p.pdf, with their panels; unnumbered
python -m figures.citeseq          # seconds -> figures/results/other/citeseq.pdf, the CITE-seq planted effects (Extended Data Fig 8 until 2026-10-08); unnumbered
python -m figures.pbmc_lfc         # seconds -> figures/results/other/pbmc_lfc.pdf, the standalone PBMC figure (clustering.lfc_se, Fig 2's subcluster panels), with its panels
python -m figures.pbmc_story       # seconds -> figures/results/other/pbmc_story.pdf, the PBMC block after the marker decision (clustering.multires_depth, clustering.lfc_se, clustering.marker_rule), with its panels
python -m figures.lymphnode_story  # seconds -> figures/results/other/lymphnode_story.pdf, the lymph node block: AUPRC and GSEA against the fraction of T-cell zone reads kept (lymphnode.depth_imbalance, lymphnode.depth_gsea), with its panels; --thin bins for the spot version, lymphnode_story_bins.pdf
```

Draws panels c-h from the arms' results — `synthetic_nb`'s `null_ratio` sweeps and `kidney`'s capsule figures and
spot-splitting runs, which must exist first — and assembles the whole figure with panels a and b, which are
drawn elsewhere and read as `fig1a.pdf` and `fig1b.pdf` from `Claude outputs/` beside the repositories.
Nature's figure guide sets the page: main figures 183 mm wide, Extended Data at most 180 mm, both at most 170 mm tall,
text 5-7 pt (`figures/layout.py`: `WIDTH`, `ED_WIDTH`, `MAX_H`; `assemble` warns past 170 mm).
The figure is 183 mm (7.2 in) wide. Every panel is placed at 100%, never scaled, so its text prints at the size it was
drawn: 7 pt for axis labels, ticks and titles, as in a and b, and 6 pt for legends.
The layout is four rows (a / b c / d e / f g h), each panel top-aligned in its row; c takes the width b leaves, and a row
wider than the figure stops the script. The axes are placed in inches, so f, g and h have identical axes, and the
method colours are keyed once, beside h. The letters are drawn over the panels, 8 pt bold.
The same panels at their own sizes, for the supplement or slides, are the arms' figures.
Every figure shares `figures/layout.py`: the page width, the text sizes, the margins and the assembly.

Fig 2 is three blocks, a row each. The lymph node, its spot version (`--thin bins`): a the section and a window on the
full-resolution H&E with the T-cell zone's spots kept or dropped, keyed as Fig 1e (T cells, GC B cells); b each test's
AUPRC of the T-cell zone's pseudo-bulk markers ("DGE AUPRC") and c its Hallmark NES correlation with the pseudo-bulk
reference ("GSEA NES correlation"), against p_sub, the fraction of the T-cell zone's 2 µm spots kept, each the mean
over the replicates with bars of one SD as Fig 1's error bars, on 0-1 (`lymphnode.depth_imbalance.draw_errorbar`,
`lymphnode.depth_gsea.draw_agreement`); the method key right of c. PBMC10k (CITE-seq): d the UMAP coloured by the Leiden
clusters at each of the five resolutions under an arrow of clustering resolution
(`clustering.multires_gaps.draw_clusters`), and e under each the variance ratio of every cluster to the rest at equal
mean (2 to the median |log2 variance ratio|, from 1 as Fig 1's Var(Y) / Var(X)), a box with every cluster a point
(`clustering.multires_gaps.draw_gaps`); f the share of each rule's top 20 that are housekeeping genes and the share of
the cluster's protein markers in it, the mean over each resolution's clusters with bars of one standard error, side by
side (`clustering.marker_rule.draw_by_resolution`), the key of the four marker rules right of it; e and f space the
resolutions evenly, labelled by their number of clusters. Abdelfattah et al. 2022, the Wilcoxon ranking genes by its
tie-corrected z-score (Scanpy's `scores`, the counterpart of LN's LFC/SE): g the NES heatmaps of their Fig. 4d pathways
under each test, on ±3 (`figures.ed_abdelfattah.heatmaps`); h per cluster, how many of those pathways keep their GSEA
call under LN, lose it, reverse it or gain one (`published.abdelfattah2022.draw_call_changes`); i MC3 and MC4's NES in
the contrasted pathways under each (`figures.ed_abdelfattah.contrast_by_cluster`). The authors' own ranking, the AUC, is
Extended Data Fig 10 (Pedro, 2026-10-09). Panel a's images come from `Claude outputs/fig2-drafts/fig2_diagrams.py`, run in
this environment, on the full-resolution H&E (`python -m fetch_data --arm lymphnode`, `lymphnode-he-fullres`, 2.2 GB).

Extended Data Fig 1 assembles `synthetic_nb`'s `lfc_ci_ln`, `lfc_ci_log1p`, `degs_metrics_sparse` and `degs_metrics_dense`,
each redrawn at its print size by its own drawing code (`lfc_confidence_intervals.draw`, `de_ratio.draw_metric`);
Extended Data Fig 2 is MAST on a synthetic null and each test's runtime. a, b: the FPR of all four tests on genes of
equal mean and unequal variance, in a null where only 150 of 1,500 genes change, so that MAST's detection-rate covariate
does not absorb the group effect as it does when every gene changes (`synthetic_nb.null_mast`, `python -m
synthetic_nb.null_mast`, about 7 minutes). c: each test's median wall time against the number of cells on sparse
counts, each run as a user runs it (`python -m synthetic_nb.runtime --sparse`, about 5 minutes; without `--sparse`,
dense mean-5 counts, `runtime.csv`); times depend on the machine.
Extended Data Fig 3 is `kidney`'s `capsule_qc`, its two panels as a and b, drawn from the saved `capsule_removal_scan.csv`
by `capsule_qc.draw_sizes` and `draw_scan`.
The Extended Data are grouped by dataset (Pedro, 2026-10-07): synthetic (1, 2), kidney (3-5), lymph node (6), PBMC (7, 8),
Abdelfattah et al. 2022 (9, 10). The CITE-seq gating panels left on 2026-10-08 and its two figures became one (8).
Extended Data Fig 4 is kidney spot splitting in full: Fig 1f with MAST, over the first 3 splits of each $p_\mathrm{split}$
(`kidney.split_null_mast`, via `figures.kidney_panels.null_fpr`); the no-DEG null's calibration (the share of null genes with
uncorrected p below 0.05, `kidney.null_calibration.draw_calibration`, via `figures.kidney_panels.calibration`), and
`split_degs_tpr_fpr` and `split_degs_pr_curves_grid`, drawn by `spot_split_plots.draw_tpr_fpr` and `draw_pr_curve`; the
AUPRC is Fig 1g.
Extended Data Fig 5 is the UMI-downsampling counterpart in Fig 1 f-h's layout: `downsample_null_fpr_log`, the AUPRC
and `_pr_curves` sharing one method key; then the downsampling null's calibration and `downsample_degs_100rep_tpr_fpr`.
Fig 1g and Extended Data Fig 5b are the AUPRC as Fig 2b's: each replicate's average precision (sklearn's), mean with
bars of one SD (`spot_split_plots.draw_auprc`, from the results CSV's `ap_mean` and `ap_std`); the area under the mean PR
curve (`draw_pr_auc`) stays in the arm's own figures only. Every AUPRC in the paper's figures is average precision:
Fig 1g, Fig 2b, Extended Data Figs 5, 6 and 9.
The rows the kidney figures share are in `figures/kidney_panels.py`.
Extended Data Fig 6 is the lymph node in full: Fig 2's spot version's AUPRC for T cells and B cells and its calls
outside the reference markers (`per_task_false_calls`); the read version's AUPRC for both compartments and its GSEA agreement
(`--thin reads`), in Fig 1's error bars. Equal thinning (`lymphnode.subsampling`, both compartments thinned alike) left
the Extended Data on 2026-10-08; its arm figures stay.
PBMC10k is one dataset, 10x Genomics' 5' PBMC 10k, a CITE-seq run (RNA and surface protein in the same cells): the
`clustering` arm clusters and tests its RNA and takes its protein as a truth (Fig 2d-f); the `citeseq` arm selects its
memory CD4 T cells by protein and plants effects in their RNA (`figures.citeseq`, unnumbered). The figures call it PBMC10k
(CITE-seq).
Extended Data Fig 7 is the PBMC clustering beyond Fig 2: PBMC3k's variance ratio by resolution (Fig 2e for PBMC3k),
each cluster's variance ratio against its |log2 depth ratio| in both datasets (`clustering.multires_gaps.draw_gap_vs_depth`),
each test's median score against the cluster's depth ratio in both datasets (`clustering.multires_depth.draw_depth`;
PBMC10k's was Fig 2e until 2026-10-07), PBMC3k's housekeeping fraction (Fig 2f for PBMC3k) and the HPA fraction in
both datasets, and two keys. The thinned random halves' calibration
(`clustering.exact_null`) left on 2026-10-08; its arm figures stay.
LN's top 10 lists, the PBMC3k-to-PBMC10k marker replication and the recovery per protein left the Extended Data on
2026-10-07; they stay in `figures.pbmc_story` and the arms' own figures.
Extended Data Fig 8 is the PBMC clusters' DEGs and their effect sizes, both datasets (`clustering.effect_sizes`): each
test's significant genes per cluster over the mean |LFC| of those genes as each package reports it, each box under
its counterpart. The CITE-seq planted effects (`citeseq`'s LFC against the truth and detection summary, Extended Data
Fig 8 until 2026-10-08) are `figures.citeseq`'s unnumbered figure; the ADT gating is `citeseq/results/adt_gating.csv`.
Extended Data Fig 9 is the Abdelfattah et al. 2022 rerun beyond Fig 2: the UMAP by cluster and by depth, each
cluster's depth ratio and its variance ratio at equal mean against |log2 depth ratio| with its least-squares line
(`python -m published.abdelfattah2022 --variance`, as Fig 2e's); each cluster's DEGs under the Wilcoxon and LN, as
the depth-ratio bars, each test's DEGs against the cluster's cells, and the Wilcoxon's over LN's against |log2 depth
ratio| (`published.abdelfattah2022 --calls`);
then the gene-level Wilcoxon z and LN score against depth, and each cluster's Hallmark NES against depth
(`published.abdelfattah2022`), the z as in Fig 2g-i.
Extended Data Fig 10 is the authors' own procedure, the Wilcoxon AUC ranking the genes for fgsea, against LN: the
heatmaps (on ±5, the published figure's range), the call changes and MC3 and MC4's contrast, as Fig 2g-i; then the
gene-level AUC and LN score, and each cluster's Hallmark NES, against depth, as Extended Data Fig 9g, h.
`figures.ed_abdelfattah` builds the same rerun with every panel, and with the
Wilcoxon's z or the log1p t-test in place of the AUC; those three are unnumbered.
The held-out false positives and the Jaccard figures are in no figure; their arm
outputs stay.
A change to a panel is made in the arm's drawing code, so the arm's own figure changes with it; the `figures/`
scripts only place panels and set what is figure-wide (page, text size, a key shared by several panels).

### `nullsplit` — null split false-positive rate

`nullsplit/ziln_null_fpr.ipynb`, run from `reproducibility/nullsplit/`. Its two CSVs are committed.

## Outputs and reference values

Every arm writes to its own `results/`, beside its scripts: `kidney/results/`, `lymphnode/results/`,
`clustering/results/`, `citeseq/results/`, `celltype/results/`, `synthetic_nb/results/`, `theory/results/`.
Each also holds `logs/` from its last run and `archive/<run>/` for earlier runs kept for comparison.
**Nothing new under `*/results/` is committed**: `.gitignore` ignores those directories, and only the
nine files tracked before that rule stay tracked (below).

**Every gene-wise test is corrected with Bonferroni, and the Wilcoxon is tie-corrected**, Seurat's defaults
(`FindMarkers`: `p.adjust(method = "bonferroni", n = nrow(object))`; its Wilcoxon adjusts for ties), for every
method in every experiment, since 2026-09-30. The shared helpers (`baselines.scanpy_sig_test`, `de_utils`) default
to both, and the other call sites pass them explicitly. Until 2026-09-24 LN was corrected with Bonferroni and its
baselines with BH; on 2026-09-24 every method moved to Bonferroni, on 2026-09-25 to BH (scanpy's default), and on
2026-09-30 back to Bonferroni with the tie-corrected Wilcoxon. `lntest`'s own default stays BH, scanpy's.
Three BH remain on purpose: the lymph node's pseudo-bulk reference (DESeq2's convention; it defines the truth, it is
not a method under comparison), GSEA across gene sets in `lymphnode/gsea_utils.py`, and `clustering.heldout_fp`'s
CSV, which also counts BH and uncorrected calls, and scanpy's Wilcoxon without tie correction, for Extended Data.

**Every output is byte-reproducible except wall-clock timings.** Regenerated twice from scratch on
2026-09-24 and checked file by file, with LN's trigamma as psi_1(a) = 1/a. Three things make that
hold, and each is easy to undo by accident:

- PDFs carry a timestamp unless pinned. `paths.py` sets `SOURCE_DATE_EPOCH` for matplotlib, which
  draws every figure.
- `clustering/plots.py` seeds the global RNG before each stripplot, which is where seaborn draws its jitter.
- `lymphnode/subsampling.py` sorts its results before writing, because workers finish in any order.

The exceptions: `lymphnode/results/runtime.pdf`, the `de_time_s` columns of the lymph node CSVs,
a timing field in its metadata JSON, and `synthetic_nb`'s `runtime*.csv` and `runtime_scaling*.pdf`.

The checksums of the current outputs are in each arm's `results/checksums.sha256`, outside git; verify one
with `shasum -a 256 -c <arm>/results/checksums.sha256`. The committed `reference/` holds the
June 2026 manifests, which pin the run behind the published PBMC3k and CITE-seq figures.

| Manifest | Files |
|---|---|
| `clustering/results/checksums.sha256` | 68 metric CSVs, 22 figures |
| `citeseq/results/checksums.sha256` | metrics, per-gene results, LaTeX table, 102 figures |
| `kidney/results/checksums.sha256` | all 13 outputs, spot splitting and UMI downsampling |
| `lymphnode/results/checksums.sha256` | the pseudo-bulk reference and 25 figures (the results CSVs carry wall-clock columns, so compare them by value) |
| `synthetic_nb/results/checksums.sha256` | the two variance sweeps on each grid (`null`, `null --grid dispersion`), Fig 1c's three `null_ratio` figures, three dispersion boxplots |
| `theory/results/checksums.sha256` | three figures, two CSVs |

`reference/unattributed/` holds two `.npy` files no script reads. Read `PROVENANCE.md` before assuming anything about them.

Getting a figure into the paper is a manual copy from the arm's `results/`, under the file name its
`\includegraphics` expects. Earlier runs are in `<arm>/results/archive/`: `june_bh` and
`stale_2026-09-11` (clustering), `bh_2026-09-24` (Bonferroni LN against BH baselines) and
`bonferroni_2026-09-25` (Bonferroni for every method). `output/` now holds only scratch from earlier
verification sessions.

Figure files were renamed on 2026-09-27 (kidney, synthetic NB, CITE-seq, lymph node); the vault's figure-style
decision maps old names to new, and the manuscripts still include the old ones until the figures are copied again.

**Nine tracked files live under `*/results/`**, committed before results were ignored:

```
celltype/results/kang_{B,T}_markers.csv
citeseq/results/CITE_seq_{results,metrics}.csv
citeseq/results/CITE_seq_lfc_error_plot.png         # moved to archive/stale_png_2026-09-27/ on 2026-09-27
citeseq/results/CD{3,4,45RA}_density_plot.pdf       # renamed gating_{cd3,cd4,cd45ra}.pdf on 2026-09-27
synthetic_nb/results/de_metrics_mu10_nobatch.csv
```

`citeseq/exp.py` writes the two CSVs, so a run replaces the June versions in the working tree and
`git status` shows them; `git checkout` restores them. Under BH with the untied Wilcoxon, `CITE_seq_metrics.csv` came
back byte-identical to the June file, which was BH for all three methods; under the current defaults it differs.

## The equivalence harness

`check_utils_vs_lntest.py` compares the frozen RECOMB-era estimator in `utils_frozen.py` against the published `lntest` package, arm by arm.

```bash
python -m check_utils_vs_lntest --arm all
```

It exists because the arms were migrated onto `lntest` one at a time and something had to prove no headline number moved.
`utils_frozen.py` is a frozen copy, kept for that comparison and for the two `theory/` scripts — do not build on it.

## Known gaps

- **Nothing here has a test suite.** `small_test.py` is a reviewer-facing single run, explicitly not expected to match the submission.
- **The kidney arm's subsampling parameters were never recorded in code.** `--n_cells_remove`, `--q` and `--lfc` are required arguments with no defaults and no config, so the only record of what produced the published panels is the manuscript prose. Recovered 2026-09-24 and written into the command block above; `spot_split`'s capsule filter has no stated value anywhere.
- **The kidney arm's published spot-split panel has unclear provenance.** `spot_split.py` defaulted to the per-capsule aggregate, whose `shape_id` is an index rather than a column, so its own guard raised `ValueError` before doing any work. The default now points at `podocytes_2um.h5ad`, per this repository's own top-level README, but which file produced the published figure is an open question.
- **`celltype/config.yaml` names an input that cannot satisfy it.** It asks for
  `celltype_column: cell_type` with the PBMC3k labels `Naive CD4 T` and `CD14 Mono`, in
  `data/pbmcs3k_pre.h5ad` — whose `obs` is only `n_genes, percent_mito, n_counts`. Nothing in this
  tree writes a `cell_type` column onto that file, and the config has not changed since it was first
  committed, so the arm has never run as configured. `de.py` itself is fine: given `data/kang_2018.h5ad`
  and two of its real labels it runs LN, t-test, Wilcoxon and MAST and saves. Which input was meant is
  an open question.
- **The null split notebook does not reproduce its committed outputs.** `ziln_null_fpr.ipynb` shows 10
  rejections / FPR 0.00304; running it gives 21 / 0.00638. The notebook was last executed 2024-11-06 and
  the estimator changed nineteen times after that, so its outputs are stale rather than wrong — 21 is what
  the frozen RECOMB-era estimator gives. See question 9 of the vault's math-questions handoff.
- **Both spatial arms have now run end to end here** (lymph node 2026-09-24; kidney 2026-09-25, where the two UMI sweeps take about 1.3 hours each on 8 cores of a laptop). Two results were ever tied to printed numbers: the kidney full-depth counts, and the lymph node Jaccard values, which matched under the old mixed correction (Bonferroni LN, BH baselines) and do not under BH for every method.
- **The trigamma question is settled.** `lntest` has one trigamma difference, `trigamma_diff(a, n) = 1/a - 1/n`, and no flag to select another. That is the form the paper defines — `psi_1(z) = 1/z`, at `nature_submission/sections/methods.tex:79-83` — and the form every published result came from. Nothing in this tree asks for a trigamma any more, because there is nothing to ask for.
