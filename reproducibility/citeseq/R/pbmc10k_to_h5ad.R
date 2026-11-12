rm(list=ls())
# Seurat for LayerData and as.SingleCellExperiment, zellkonverter for writeH5AD.
library(Seurat)
library(zellkonverter)

# Run from citeseq/R/ (cd citeseq/R && Rscript <script>), as the paths are relative to it.
# The committed input is in citeseq/data/, and every output goes to citeseq/results/.
data_dir <- "../data/"
results_dir <- "../results/"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# zellkonverter manages its own Python through basilisk.

MIN_READS <- 3
MIN_CELLS <- 5

dat <- readRDS(paste0(results_dir, "pbmc10k_cd4_memory.rds"))

cts <- LayerData(dat, assay = "RNA", layer="counts")
dim(cts)
row_idxs <- which(rowSums(cts > MIN_READS) >= MIN_CELLS)
length(row_idxs)

sub_dat <- dat[row_idxs,]
out_path <- paste0(data_dir, "memory_CD4.h5ad")
sce <- as.SingleCellExperiment(sub_dat)
writeH5AD(sce, file = out_path)

