# Reference values for
#   * partial / semi-partial correlation with 1, 2 and 3 correlated controls,
#     Pearson and Spearman: ppcor::pcor.test / spcor.test (estimate, statistic,
#     p-value, df = n - 2 - k);
#   * the AUC null quantile: qwilcox(p, n_pos, n_neg) / (n_pos * n_neg) over a
#     grid of p (incl. 0, 1e-12, 1 - 1e-12, 1) and sizes up to (50, 40).
#
#   Rscript R/generate_partial_auc_refs.R      (run from the crate root)
#
# Writes R/data/partial_auc_reference.csv with columns case, key, values
# (values: space-separated, "%.17g").
.libPaths(c("~/R/userlib", .libPaths()))
suppressMessages(library(ppcor))

rows <- list()
f17 <- function(x) paste(sprintf("%.17g", as.numeric(x)), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

set.seed(2015)
n <- 40
z1 <- rnorm(n)
z2 <- 0.7 * z1 + rnorm(n, sd = 0.7)
z3 <- 0.5 * z1 - 0.6 * z2 + rnorm(n, sd = 0.6)
x <- 0.8 * z1 + 0.5 * z2 + rnorm(n)
y <- 0.6 * x - 0.7 * z2 + 0.4 * z3 + rnorm(n)
x[5] <- x[7]  # a tie for the Spearman ranks
add("data", "x", x); add("data", "y", y)
add("data", "z1", z1); add("data", "z2", z2); add("data", "z3", z3)
Z <- cbind(z1, z2, z3)
for (k in 1:3) {
  for (m in c("pearson", "spearman")) {
    zz <- Z[, 1:k, drop = FALSE]
    p <- pcor.test(x, y, zz, method = m)
    s <- spcor.test(x, y, zz, method = m)
    add(paste0("pcor_", m), paste0("k", k), c(p$estimate, p$statistic, p$p.value, n - 2 - k))
    add(paste0("spcor_", m), paste0("k", k), c(s$estimate, s$statistic, s$p.value, n - 2 - k))
  }
}

probs <- c(0, 1e-12, 0.001, 0.025, 0.05, 0.3, 0.5, 0.7, 0.95, 0.975, 0.999, 1 - 1e-12, 1)
add("auc_q", "probs", probs)
for (sz in list(c(1, 1), c(3, 4), c(10, 10), c(7, 23), c(25, 25), c(50, 40), c(40, 50))) {
  add("auc_q", sprintf("n%d_%d", sz[1], sz[2]), qwilcox(probs, sz[1], sz[2]) / (sz[1] * sz[2]))
}

out <- do.call(rbind, rows)
write.csv(out, "R/data/partial_auc_reference.csv", row.names = FALSE)
cat("Wrote R/data/partial_auc_reference.csv (", nrow(out), "rows)\n")
print(out[out$case != "data", ])
