# Abdelfattah et al. 2022 (Nat Commun 13:767): GSEA of each myeloid cluster's gene ranking as the authors ran it (their
# R/Figure4.R): fgsea on msigdbr's Hallmark sets (species Homo sapiens, category H), stats the ranking, minSize 10,
# every other setting fgsea's default (the multilevel method) but nPermSimple, 10,000 in place of 1,000, and 100,000 for
# a ranking whose run at 10,000 still leaves a set without an NES: with 1,000 the skewed rankings (LN's for MC09, the
# t-test's for MC02) leave 15 sets without one, as no random set takes the observed sign, and at 10,000 the Wilcoxon z
# rankings leave 6; more permutations only refine the null the NES is normalised by. Each ranking is
# published/abdelfattah2022.py's: the
# Wilcoxon AUC uncentred, as they ranked genes, the Wilcoxon's tie-corrected z (scanpy's), and LN's and the log1p
# t-test's scores. msigdbr is pinned at 7.5.1,
# the release nearest their code; seeded, serial.
#
#     Rscript published/R/abdelfattah2022_fgsea.R      # from reproducibility/, after python -m published.abdelfattah2022 --rankings

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
  library(BiocParallel)
})

out_dir <- "published/results"
METHODS <- c("wilcoxon", "wilcoxon_z", "ln", "t_test")
MIN_SIZE <- 10
N_PERM_SIMPLE <- c(10000, 100000)
SEED <- 42

ranks <- fread(file.path(out_dir, "rankings.csv"))
ranks <- ranks[method %in% METHODS & is.finite(value)]
hallmark <- msigdbr(species = "Homo sapiens", category = "H")
sets <- split(hallmark$gene_symbol, hallmark$gs_name)
cat(sprintf("  msigdbr %s, fgsea %s: %d Hallmark sets\n", packageVersion("msigdbr"), packageVersion("fgsea"),
            length(sets)))

results <- list()
for (cl in sort(unique(ranks$cluster))) {
  for (m in METHODS) {
    r <- ranks[cluster == cl & method == m][order(-value, gene)]
    stats <- setNames(r$value, r$gene)
    for (n in N_PERM_SIMPLE) {
      set.seed(SEED)
      f <- suppressWarnings(fgsea(sets, stats = stats, minSize = MIN_SIZE, nPermSimple = n, BPPARAM = SerialParam()))
      if (!anyNA(f$NES)) break
    }
    if (n > N_PERM_SIMPLE[1]) cat(sprintf("  %s %s: nPermSimple %d\n", cl, m, n))
    results[[paste(cl, m)]] <- data.table(cluster = cl, method = m, pathway = f$pathway, size = f$size,
                                          nes = f$NES, fdr = f$padj, n_perm_simple = n)
  }
  cat(sprintf("  %s\n", cl))
}
gsea <- rbindlist(results)[order(cluster, method, pathway)]
fwrite(gsea, file.path(out_dir, "gsea.csv"))
cat(sprintf("  wrote %s, %d rows\n", file.path(out_dir, "gsea.csv"), nrow(gsea)))
