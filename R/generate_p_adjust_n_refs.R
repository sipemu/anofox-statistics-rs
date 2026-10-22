# Reference values for p_adjust_n: R's p.adjust(p, method, n) with n >= the
# number of non-NA p-values (tests/posthoc_reference.rs).
# Run with: Rscript R/generate_p_adjust_n_refs.R
f17 <- function(v) paste(ifelse(is.na(v), "NA", sprintf("%.17g", v)), collapse = " ")
cases <- list(
  list(p = c(0.01, 0.02, 0.03, 0.04, 0.05), n = c(5, 6, 10, 100)),
  list(p = c(0.2, 0.001, 0.04, 0.04, 0.9, 0.0001, 0.3, 0.04), n = c(8, 9, 20)),
  list(p = c(NA, 0.03, 0.5, NA, 0.001, 0.02), n = c(4, 5, 6, 12)),
  list(p = c(0, 1, 0.5, 0, 1, 0.025), n = c(6, 7, 15)),
  list(p = c(0.04, 0.01), n = c(2, 3, 4)),
  list(p = c(0.3), n = c(1, 2, 3, 10)),
  list(p = c(NA, 0.3), n = c(1, 2, 5)),
  list(p = c(0.011, 0.012, 0.013, 0.5, 0.6), n = c(5, 7, 50))
)
methods <- c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none")
rows <- list()
for (k in seq_along(cases)) for (n in cases[[k]]$n) for (m in methods)
  rows[[length(rows) + 1]] <- data.frame(case = k, method = m, n = n, p = f17(cases[[k]]$p),
                                         adj = f17(p.adjust(cases[[k]]$p, m, n)))
write.csv(do.call(rbind, rows), file.path("R/data", "p_adjust_n_reference.csv"), row.names = FALSE)
