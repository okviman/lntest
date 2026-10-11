# Abdelfattah et al. 2022 (Nat Commun 13:767): a UMAP of their 83,479 myeloid cells built as the paper's Methods
# describe, with their tools, since their own myeloid embedding is not public. Seurat: library-size normalisation and
# log, the 2,000 most variable genes (vst), scaling, PCA; harmony over patient and sex; UMAP and Louvain clustering on
# the first PCS harmony components, at the resolution of their figure code (0.25). The paper does not state the number
# of components; PCS is ours. Writes, per cell, the UMAP, the authors' cluster, ours and the counts, and the agreement
# between the two clusterings (adjusted Rand index), for published/abdelfattah2022.py to draw.
#
#     Rscript published/R/abdelfattah2022_umap.R      # from reproducibility/, several minutes; needs fetch_data --arm published

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(harmony)
  library(mclust)
})

counts_dir <- "data/GSE182109"
meta_path <- "data/Meta_Data_GBMatlas.txt"
out_dir <- "published/results"
CLUSTERS <- sprintf("MC%02d", 1:9)
MIN_CELLS <- 3
HVG <- 2000
PCS <- 30
RESOLUTION <- 0.25
SEED <- 42

stopifnot(dir.exists(counts_dir), file.exists(meta_path))
meta <- read.delim(meta_path, row.names = 1, check.names = FALSE)
meta <- meta[as.character(meta$SubCluster) %in% CLUSTERS, ]

# Each sample's counts for the authors' labelled myeloid cells, over the genes every sample's reference shares.
parts <- list()
for (gsm in sort(unique(meta$GSMID))) {
  m <- meta[meta$GSMID == gsm, ]
  x <- readMM(Sys.glob(file.path(counts_dir, paste0(gsm, "_*matrix.mtx.gz"))))
  barcodes <- read.delim(Sys.glob(file.path(counts_dir, paste0(gsm, "_*barcodes.tsv.gz"))), header = FALSE)$V1
  genes <- read.delim(Sys.glob(file.path(counts_dir, paste0(gsm, "_*features.tsv.gz"))), header = FALSE)$V2
  first <- !duplicated(genes)
  idx <- match(m$barcode, barcodes)
  keep <- !is.na(idx)
  x <- as(x[first, idx[keep], drop = FALSE], "CsparseMatrix")
  rownames(x) <- genes[first]
  colnames(x) <- rownames(m)[keep]
  parts[[gsm]] <- x
}
common <- Reduce(intersect, lapply(parts, rownames))
x <- do.call(cbind, lapply(parts, function(p) p[common, , drop = FALSE]))
cat(sprintf("  %d cells, %d shared genes\n", ncol(x), length(common)))

obj <- CreateSeuratObject(counts = x, meta.data = meta[colnames(x), c("Patient", "sex", "SubCluster")],
                          min.cells = MIN_CELLS)
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = HVG, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = PCS, seed.use = SEED, verbose = FALSE)
set.seed(SEED)
obj <- RunHarmony(obj, group.by.vars = c("Patient", "sex"), reduction.use = "pca", verbose = FALSE)
obj <- RunUMAP(obj, reduction = "harmony", dims = 1:PCS, seed.use = SEED, verbose = FALSE)
obj <- FindNeighbors(obj, reduction = "harmony", dims = 1:PCS, verbose = FALSE)
obj <- FindClusters(obj, resolution = RESOLUTION, random.seed = SEED, verbose = FALSE)

umap <- Embeddings(obj, "umap")
cells <- data.frame(cell = colnames(obj), umap_1 = umap[, 1], umap_2 = umap[, 2],
                    cluster = as.character(obj$SubCluster), louvain = as.character(obj$seurat_clusters),
                    depth = as.numeric(colSums(LayerData(obj, assay = "RNA", layer = "counts"))))
cells <- cells[order(cells$cell), ]
write.csv(cells, file.path(out_dir, "umap.csv"), row.names = FALSE, quote = FALSE)
agreement <- data.frame(cells = nrow(cells), louvain_clusters = length(unique(cells$louvain)),
                        adjusted_rand_index = adjustedRandIndex(cells$cluster, cells$louvain))
write.csv(agreement, file.path(out_dir, "umap_agreement.csv"), row.names = FALSE, quote = FALSE)
print(agreement)
print(table(cells$cluster, cells$louvain))
