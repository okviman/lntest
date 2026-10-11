rm(list=ls())
# Only these two are used: Seurat for LayerData/as.SingleCellExperiment, and
# zellkonverter for writeH5AD. This script also loaded anndata, basilisk,
# data.table, dplyr, ggplot2, reticulate and splatter and touched none of them,
# which made the arm's R dependencies look far heavier than they are.
library(Seurat)
library(zellkonverter)

# Paths are relative to this script's directory (citeseq/R/), which is what
# ZILN.Rproj sets as the R working directory. The arm's committed input lives in
# citeseq/data/; everything this chain produces goes to citeseq/results/.
data_dir <- "../data/"
results_dir <- "../results/"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)


# Removed: use_python("/Users/sjun6/opt/anaconda3/envs/nbsr/bin/python"), an
# absolute path into a co-author's machine. zellkonverter manages its own Python
# through basilisk, so nothing here needs a interpreter named by hand.
#
# Also removed an unused `methods` vector naming five Seurat DE tests; this
# script converts an object and runs no test.

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

