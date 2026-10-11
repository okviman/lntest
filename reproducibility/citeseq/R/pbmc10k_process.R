rm(list=ls())
library(data.table)
library(mclust)
library(tidyverse)
library(Seurat)

library(Seurat)

# Paths are relative to this script's directory (citeseq/R/), which is what
# ZILN.Rproj sets as the R working directory. The arm's committed input lives in
# citeseq/data/; everything this chain produces goes to citeseq/results/.
data_dir <- "../data/"
results_dir <- "../results/"
# No figures are drawn in R: citeseq/plots.py draws them from the CSVs this chain writes.

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

MIN_CELLS <- 3

# Was a bare filename, so this only worked if you happened to be standing in the
# directory holding the download. It now reads from citeseq/data/, which is where
# `python -m fetch_data --arm citeseq` puts it.
raw_h5 <- paste0(data_dir, "sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5")
if (!file.exists(raw_h5)) {
  stop(paste0("missing ", raw_h5, "\n",
              "  what:   the 5' PBMC 10k filtered_feature_bc_matrix, raw input to this arm\n",
              "  source: run `python -m fetch_data --arm citeseq` from reproducibility/"))
}
seurat <- Read10X_h5(raw_h5)
output_path <- results_dir
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = T)
}

adt_counts <- seurat$`Antibody Capture`
rownames(adt_counts) <- stringr::str_replace(string = rownames(adt_counts), pattern = "_TotalC", replacement = "")
pbmc10k <- SeuratObject::CreateSeuratObject(counts = seurat$`Gene Expression`, project = "PBMC10k")
pbmc10k[["ADT"]] <- SeuratObject::CreateAssayObject(counts = adt_counts)
# Filtering.
pbmc10k[["percent.mt"]] <- PercentageFeatureSet(pbmc10k, pattern = "^MT-", assay = "RNA")
pbmc10k$ADT_nCount   <- Matrix::colSums(GetAssayData(pbmc10k, assay = "ADT", layer = "counts"))
pbmc10k$ADT_nFeature <- Matrix::colSums(GetAssayData(pbmc10k, assay = "ADT", layer = "counts") > 0)
keep_cells <- with(pbmc10k@meta.data,
                   nFeature_RNA >= 200 & 
                     nFeature_RNA <= 6000 &
                     percent.mt   <= 10 &
                     ADT_nCount   >= 500)
pbmc10k <- subset(pbmc10k, cells = colnames(pbmc10k)[keep_cells])


# Identify cell types.
# Normalize ADT data using CLR.
pbmc10k <- NormalizeData(pbmc10k, assay = "ADT", normalization.method = "CLR")
# slot= was defunct as of SeuratObject 5.0.0; layer= is the same argument.
# Lines 39-40 had already been migrated, this one had not.
adt_clr_transformed <- t(GetAssayData(pbmc10k, assay = "ADT", layer = "data"))
# Fit Gaussian mixture model to identify CD3+ and CD4++ cells.
fit_mclust <- function(x, num_clusters) {
  fit <- mclust::Mclust(x, G = num_clusters)
  # Order the clusters.
  labels <- rep("", num_clusters)
  labels[1] <- "-"
  for (i in 1:(num_clusters - 1))
  {
    labels[i+1] <- paste(rep("+", i), collapse = "")
  }
  cl_id_ordered <- order(fit$parameters$mean, decreasing = FALSE)
  reorder_map <- integer(num_clusters)
  reorder_map[cl_id_ordered] <- 1:num_clusters
  cluster_id_reordered <- reorder_map[fit$classification]
  final_labels <- labels[cluster_id_reordered]
  result_df <- data.frame(
    x = x,
    cluster_id_reordered = cluster_id_reordered,
    label = final_labels
  )
  return(result_df)
}
adt_clr_transformed_dt <- data.table(barcode = rownames(adt_clr_transformed), adt_clr_transformed)
adt_clr_transformed_dt <- adt_clr_transformed_dt %>% 
  mutate(
    CD3_cluster = fit_mclust(CD3, num_clusters = 2)$label,
    CD4_cluster = fit_mclust(CD4, num_clusters = 3)$label,
    CD45RA_cluster = fit_mclust(CD45RA, num_clusters = 2)$label
  )

# The CLR values and GMM gates, for the three gating figures (citeseq/plots.py).
fwrite(select(adt_clr_transformed_dt, barcode, CD3, CD4, CD45RA, CD3_cluster, CD4_cluster, CD45RA_cluster),
       paste0(results_dir, "adt_gating.csv"))

adt_clr_transformed_dt %>% 
  filter(CD3_cluster == "+" & CD4_cluster == "++" & CD45RA_cluster == "-") %>% 
  select(barcode) -> cd4_memory_cells

pbmc10k_cd4_memory <- subset(pbmc10k, cells = cd4_memory_cells$barcode)
dim(pbmc10k_cd4_memory)
saveRDS(pbmc10k_cd4_memory, file = paste0(results_dir, "pbmc10k_cd4_memory.rds"))

# Save all of the data.
saveRDS(pbmc10k, file = paste0(results_dir, "pbmc10k.rds"))

