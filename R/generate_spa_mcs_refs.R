# Reference values for the forecast-comparison bootstrap tests:
#   * Model Confidence Set  -> R package MCS (MCS::MCSprocedure, >= 0.2.1)
#   * SPA test (Hansen 2005) and the MSPE-adjusted SPA (Clark-West
#     differentials) -> a base-R implementation of Hansen (2005), below.
#
#   Rscript R/generate_spa_mcs_refs.R      (run from the crate root)
#
# Writes R/data/spa_mcs_reference.csv with columns case, key, values
# (values: space-separated, "%.17g"; inputs are in_* keys).
#
# Two kinds of cases:
#   exact_*  the bootstrap index vectors are written out (0-based, in_idx,
#            B * n values, replication by replication) and the Rust tests
#            feed the same indices to the internal cores: the p-values must
#            agree to rounding.
#   mc_*     large-B runs with R's own random numbers; the Rust tests run
#            their own bootstrap with the same scheme and compare within
#            Monte-Carlo tolerance.
.libPaths(c("~/R/userlib", .libPaths()))
suppressMessages(library(MCS))

rows <- list()
f17 <- function(x) paste(ifelse(is.na(x), "NA", sprintf("%.17g", as.numeric(x))), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

ar1 <- function(n, phi) as.numeric(stats::filter(rnorm(n), phi, method = "recursive"))

# ---------------------------------------------------------------------------
# Hansen (2005) SPA in base R
# ---------------------------------------------------------------------------
# omega^2_k = gamma_0 + 2 sum_{i=1}^{n-1} kappa(n, i) gamma_i,
# kappa(n, i) = (n-i)/n (1-q)^i + i/n (1-q)^(n-i), q = 1 / block length.
spa_omega2 <- function(d, l) {
  n <- length(d); q <- 1 / l; cc <- d - mean(d)
  g <- vapply(0:(n - 1), function(i) sum(cc[1:(n - i)] * cc[(1 + i):n]) / n, 0)
  i <- 1:(n - 1)
  kap <- (n - i) / n * (1 - q)^i + i / n * (1 - q)^(n - i)
  g[1] + 2 * sum(kap * g[-1])
}
# D: n x K matrix of differentials (benchmark loss - model loss).
# T_SPA = max(0, max_k sqrt(n) dbar_k / omega_k); bootstrap
# T*_b = max(0, max_k sqrt(n) (dbar*_k - g(dbar_k)) / omega_k) with
# g_l = max(x, 0), g_c = x 1{x >= -sqrt(omega^2/n 2 log log n)}, g_u = x;
# p = mean(T* >= T_SPA): T_SPA and T* have an atom at 0, so with T_SPA = 0
# (no model beats the benchmark) all p-values are 1; for T_SPA > 0 this is
# Hansen's mean(T* > T_SPA).
spa <- function(D, l, idx_list) {
  n <- nrow(D)
  dbar <- colMeans(D)
  om <- sqrt(apply(D, 2, spa_omega2, l = l))
  stat <- max(0, max(sqrt(n) * dbar / om))
  A <- sqrt(om^2 / n * 2 * log(log(n)))
  g <- list(l = pmax(dbar, 0), c = dbar * (dbar >= -A), u = dbar)
  Tb <- vapply(idx_list, function(idx) {
    m <- colMeans(D[idx, , drop = FALSE])
    vapply(g, function(gg) max(0, sqrt(n) * (m - gg) / om), 0)
  }, numeric(3))
  list(stat = stat, p = rowMeans(Tb >= stat), omega = om)
}
# Stationary bootstrap indices (geometric block lengths, wrap-around), as
# boot::tsboot(sim = "geom").
stat_idx <- function(n, l) {
  out <- integer(0)
  while (length(out) < n) {
    s <- sample.int(n, 1); len <- 1 + rgeom(1, 1 / l)
    out <- c(out, ((s - 1 + 0:(len - 1)) %% n) + 1)
  }
  out[1:n]
}

# ---------------------------------------------------------------------------
# SPA data: benchmark + 5 models, n = 200, cross-correlated, autocorrelated
# ---------------------------------------------------------------------------
set.seed(20051)
n <- 200
z <- ar1(n, 0.4)
mu <- c(0.10, 0.08, 0.00, -0.60, -1.00)
D <- sapply(mu, function(m) m + 0.6 * z + 0.6 * ar1(n, 0.3))
L0 <- 5 + 0.5 * ar1(n, 0.5)
L <- L0 - D                       # model losses
add("spa", "in_bench", L0)
for (k in seq_along(mu)) add("spa", paste0("in_model", k - 1), L[, k])
D <- L0 - L                       # exactly as Rust forms it
l_spa <- 4

# Exact case: 300 replications with stored indices.
set.seed(99)
B <- 200
idx <- lapply(1:B, function(b) stat_idx(n, l_spa))
r <- spa(D, l_spa, idx)
add("exact_spa", "in_idx", unlist(idx) - 1)
add("exact_spa", "block_length", l_spa)
add("exact_spa", "statistic", r$stat)
add("exact_spa", "omega", r$omega)
add("exact_spa", "p", r$p)  # lower, consistent, upper

# Monte-Carlo case.
set.seed(7)
B <- 20000
r <- spa(D, l_spa, lapply(1:B, function(b) stat_idx(n, l_spa)))
add("mc_spa", "block_length", l_spa)
add("mc_spa", "statistic", r$stat)
add("mc_spa", "p", r$p)
cat("SPA:", r$stat, r$p, "\n")

# Degenerate case: two clearly worse models (T_SPA = 0, no evidence against
# the null): all three p-values must be 1.
set.seed(2006)
Dw <- sapply(c(-0.8, -1.5), function(m) m + 0.6 * z + 0.6 * ar1(n, 0.3))
add("spa_worse", "in_bench", L0)
for (k in 1:2) add("spa_worse", paste0("in_model", k - 1), L0 - Dw[, k])
Dw <- L0 - (L0 - Dw)
set.seed(9)
r <- spa(Dw, l_spa, lapply(1:2000, function(b) stat_idx(n, l_spa)))
add("mc_spa_worse", "block_length", l_spa)
add("mc_spa_worse", "statistic", r$stat)
add("mc_spa_worse", "p", r$p)
cat("SPA (worse models):", r$stat, r$p, "\n")
stopifnot(r$stat == 0, all(r$p == 1))

# MSPE-adjusted SPA: Clark-West differentials f = e0^2 - ek^2 + (e0 - ek)^2.
set.seed(2012)
n2 <- 150
e0 <- ar1(n2, 0.3)
# Two models that only add noise (CW differential has mean zero) and one
# that shrinks the forecast error slightly.
E <- cbind(e0 + 0.3 * rnorm(n2), e0 + 0.5 * ar1(n2, 0.2), 0.999 * e0 + 0.2 * rnorm(n2))
add("mspe", "in_bench", e0)
for (k in 1:ncol(E)) add("mspe", paste0("in_model", k - 1), E[, k])
Fcw <- sapply(1:ncol(E), function(k) e0^2 - E[, k]^2 + (e0 - E[, k])^2)
set.seed(8)
r <- spa(Fcw, 3, lapply(1:20000, function(b) stat_idx(n2, 3)))
add("mc_mspe", "block_length", 3)
add("mc_mspe", "statistic", r$stat)
add("mc_mspe", "p", r$p)
cat("MSPE-SPA:", r$stat, r$p, "\n")

# ---------------------------------------------------------------------------
# Model Confidence Set (MCS::MCSprocedure)
# ---------------------------------------------------------------------------
set.seed(2011)
n3 <- 150
common <- ar1(n3, 0.5)
mu3 <- c(0.00, 0.06, 0.25, 0.70, 0.12)
Lm <- sapply(mu3, function(m) 3 + m + common + 0.8 * ar1(n3, 0.3))
colnames(Lm) <- paste0("m", 0:4)
for (k in 1:ncol(Lm)) add("mcs", paste0("in_model", k - 1), Lm[, k])

mcs_out <- function(case, fit, B, k) {
  tab <- fit@show
  models <- colnames(Lm)
  add(case, "order", match(fit@Info$elimination.order, models) - 1)
  add(case, "step_p", tab[fit@Info$elimination.order, 2])
  add(case, "mcs_p", tab[models, 3])
  add(case, "B", B); add(case, "k", k)
}
for (stat in c("Tmax", "TR")) {
  # Exact: same indices as MCSprocedure(seed = 31) draws internally.
  B <- 200; k <- 3
  set.seed(31)
  idx <- lapply(1:B, function(b) MCS:::DrawIndices(n3, k))
  fit <- MCSprocedure(Lm, alpha = 0.1, B = B, statistic = stat, k = k, seed = 31, verbose = FALSE)
  case <- paste0("exact_mcs_", stat)
  add(case, "in_idx", unlist(idx) - 1)
  mcs_out(case, fit, B, k)
  # Monte Carlo: moving blocks of length 3, and IID (k = 1).
  for (k in c(1, 3)) {
    fit <- MCSprocedure(Lm, alpha = 0.1, B = 20000, statistic = stat, k = k, seed = 5, verbose = FALSE)
    mcs_out(paste0("mc_mcs_", stat, "_k", k), fit, 20000, k)
    cat("MCS", stat, "k", k, ":", fit@Info$elimination.order, "\n")
    print(fit@show)
  }
}

out <- do.call(rbind, rows)
write.csv(out, "R/data/spa_mcs_reference.csv", row.names = FALSE)
cat("Wrote R/data/spa_mcs_reference.csv (", nrow(out), "rows)\n")
