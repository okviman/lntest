rm(list=ls())
library(data.table)
library(glue)
library(Seurat)
library(tidyverse)
library(xtable)

# Run from citeseq/R/ (cd citeseq/R && Rscript <script>), as the paths are relative to it.
# The committed input is in citeseq/data/, and every output goes to citeseq/results/.
data_dir <- "../data/"
results_dir <- "../results/"
# No figures are drawn in R: citeseq/plots.py draws them from the CSVs this chain writes.

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)


# Seurat's avg_log2FC on each replicate citeseq/exp.py writes. Not used in the paper.
replicates <- 0:99
lfc_results <- list()
for (rep_no in replicates)
{
  data_path <- glue("{results_dir}replicates/rep{rep_no}/")
  X <- read.csv(glue("{data_path}/X.csv"), header=F)
  Y <- read.csv(glue("{data_path}/Y.csv"), header=F)
  true_lfcs <- read.csv(glue("{data_path}/true_lfcs.csv"), header=F)
  gene_names_df <- read.csv(glue("{data_path}/gene_names.csv"), header = FALSE)
  cts <- cbind(t(X), t(Y))
  rownames(cts) <- gene_names_df$V1
  seurat_obj <- CreateSeuratObject(counts = cts)
  seurat_obj$group <- c(rep("A", nrow(X)), rep("B", nrow(Y)))
  seurat_obj <- NormalizeData(seurat_obj)
  Idents(seurat_obj) <- seurat_obj$group
  results <- FindMarkers(seurat_obj, 
                         ident.1 = "A", 
                         ident.2 = "B", 
                         logfc.threshold = 0, 
                         test.use = "wilcox")
  true_lfcs_dt <- data.table(gene_names = gene_names_df$V1, true_lfcs=true_lfcs$V1)
  results_dt <- data.table(gene_names = rownames(results), 
                           seurat_lfc = results$avg_log2FC)
  merged <- merge(true_lfcs_dt, results_dt, by="gene_names")
  saveRDS(merged, glue("{data_path}/seurat_lfc_results.rds"))
  lfc_results[[(rep_no+1)]] <- merged
}

lfc_results_dt <- rbindlist(lfc_results, idcol = TRUE)
lfc_results_dt$rep <- lfc_results_dt$.id - 1
saveRDS(lfc_results_dt, paste0(results_dir, "seurat_lfc_results.rds"))

# The same table as CSV, for the Seurat LFC figures (citeseq/plots.py).
fwrite(lfc_results_dt, paste0(results_dir, "seurat_lfc_results.csv"))

