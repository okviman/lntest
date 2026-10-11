rm(list=ls())
library(data.table)
library(Seurat)
library(tidyverse)
library(xtable)

# Paths are relative to this script's directory (citeseq/R/), which is what
# ZILN.Rproj sets as the R working directory. The arm's committed input lives in
# citeseq/data/; everything this chain produces goes to citeseq/results/.
data_dir <- "../data/"
results_dir <- "../results/"
# No figures are drawn in R: citeseq/plots.py draws them from the CSVs this chain writes.

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# Method names and order shared with the Python figures (reproducibility/method_colors.py).
method_names <- c("LN" = "LN's t-test", "t-test" = "log1p t-test", "Wilcoxon" = "Wilcoxon")
method_levels <- c("LN's t-test", "log1p t-test", "Wilcoxon")


results <- fread(paste0(results_dir, "CITE_seq_results.csv"))
metrics <- fread(paste0(results_dir, "CITE_seq_metrics.csv"))
names(metrics)[1] <- "Metric"

mean(subset(metrics, Metric == "accuracy")$LN)

tbl <- metrics %>% 
  mutate(Metric = if_else(Metric %in% c("fpr", "tpr", "fnr", "tnr"), 
                          str_to_upper(Metric), 
                          str_to_title(Metric))) %>% 
  pivot_longer(cols = c("LN", "Wilcoxon", "t-test"), names_to = "Method", values_to = "value")
results_tbl <- tbl %>% 
  group_by(Method, Metric) %>% 
  summarise(n = n(), 
            mean = mean(value, na.rm=TRUE), 
            se = sd(value, na.rm=TRUE)/sqrt(n),
            .groups = "drop")

metric_order <- c("Accuracy", "Precision", "Recall", "TPR", "TNR", "FNR", "FPR", "F1")
xt <- results_tbl %>% 
  #mutate(cell = sprintf("%.3f ± %.3f", mean, se)) %>%
  mutate(
    Metric = factor(Metric, metric_order),
    Method = factor(method_names[Method], method_levels),
    cell = sprintf("%.3f", mean)) %>%
  select(Method, Metric, cell) %>%
  pivot_wider(names_from = Method, values_from = cell) %>%
  arrange(Metric) %>% 
  xtable()
print(xt, include.rownames = FALSE, type="latex", file = paste0(results_dir, "CITE_seq_metrics.tex"), timestamp = NULL)

results[is_signal_gene == TRUE,.(mean((ln_lfc - true_lfc)^2),
                                 mean((scanpy_lfc - true_lfc)^2))] 
