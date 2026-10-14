# Reference values for timeseries::durbin_watson_test (lmtest::dwtest).
#
#   Rscript R/generate_dwtest_refs.R      (run from the crate root)
#
# Writes R/data/dwtest_reference.csv with columns
#   case, key, values   (values: space-separated, "%.17g")
# Inputs: in_resid (OLS residuals), in_x1..in_xk (design columns, including
# the intercept column when the model has one), in_exact (1, 0 or NA = R's
# default exact = n < 100). Outputs: stat and p_<alternative> for
# alternative = greater, two.sided and less (dwtest with R's defaults).
#
# For the exact cases, pexact_<alternative> is a high-precision exact value:
# P(DW < d) = P(sum_j (lambda_j - d) z_j^2 < 0) with lambda_j the eigenvalues
# of M A (as in dwtest), from Imhof's (1961) formula integrated with
# stats::integrate at a tight tolerance. It is cross-checked against dwtest
# with a long Pan series (iterations = 1000; they agree to ~1e-8). dwtest's
# default 15-term series fails for case e_three (it returns a value outside [0, 1] and dwtest falls
# back to the normal approximation), so there the exact value differs from
# p_<alternative> on purpose.
suppressMessages(library(lmtest))

rows <- list()
f17 <- function(x) paste(ifelse(is.na(x), "NA", sprintf("%.17g", as.numeric(x))), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

ar1 <- function(n, phi) as.numeric(arima.sim(list(ar = phi), n))

imhof_below0 <- function(w) {
  w <- w[w != 0]
  f <- function(u) sapply(u, function(v) {
    sin(0.5 * sum(atan(w * v))) / (v * prod((1 + (w * v)^2)^0.25))
  })
  0.5 - integrate(f, 0, Inf, rel.tol = 1e-13, abs.tol = 1e-15, subdivisions = 100000L)$value / pi
}

exact_p <- function(name, y, X, alt) {
  n <- nrow(X); k <- ncol(X)
  res <- lm.fit(X, y)$residuals
  dw <- sum(diff(res)^2) / sum(res^2)
  Q1 <- chol2inv(qr.R(qr(X)))
  A <- diag(c(1, rep(2, n - 2), 1)); A[abs(row(A) - col(A)) == 1] <- -1
  ev <- Re(eigen((diag(n) - X %*% Q1 %*% t(X)) %*% A)$values[1:(n - k)])
  ev <- ev[ev > 1e-10]
  pdw <- imhof_below0(ev - dw)
  pan <- dwtest(y ~ X - 1, alternative = "greater", exact = TRUE, iterations = 1000)$p.value
  # Long Pan series agree to ~1e-8 (its rounding noise, e.g. e_three).
  if (abs(pdw - pan) > 1e-7) stop(name, ": Imhof ", pdw, " vs Pan(1000) ", pan)
  switch(alt, two.sided = 2 * min(pdw, 1 - pdw), less = 1 - pdw, greater = pdw)
}

case <- function(name, y, X, exact = NULL) {
  n <- nrow(X)
  use_exact <- if (is.null(exact)) n < 100 else exact
  add(name, "in_resid", lm.fit(X, y)$residuals)
  for (j in seq_len(ncol(X))) add(name, paste0("in_x", j), X[, j])
  add(name, "in_exact", if (is.null(exact)) NA else as.numeric(exact))
  for (alt in c("greater", "two.sided", "less")) {
    r <- dwtest(y ~ X - 1, alternative = alt, exact = exact)
    add(name, "stat", r$statistic)
    add(name, paste0("p_", alt), r$p.value)
    if (use_exact) add(name, paste0("pexact_", alt), exact_p(name, y, X, alt))
  }
}

set.seed(177)
# Exact (n < 100): distribution of DW from the eigenvalues of M A.
n <- 30; case("e_int_ar", 1 + ar1(n, 0.7), cbind(1, rep(1, n))[, 1, drop = FALSE])
n <- 25; t <- seq_len(n); case("e_trend", 2 + 0.3 * t + rnorm(n), cbind(1, t))
n <- 50; x1 <- rnorm(n); x2 <- runif(n)
case("e_two_neg", 1 + x1 - x2 + ar1(n, -0.6), cbind(1, x1, x2))
n <- 80; x1 <- rnorm(n); x2 <- rnorm(n); x3 <- seq_len(n) / n
case("e_three", 0.5 * x1 + x2 + x3 + rnorm(n), cbind(1, x1, x2, x3))
n <- 20; x1 <- runif(n, 1, 5); case("e_noint", 2 * x1 + rnorm(n), cbind(x1))
n <- 60; x1 <- rnorm(n); case("e_strong_ar", 1 + x1 + ar1(n, 0.9), cbind(1, x1))
n <- 6; x1 <- c(1.5, 2.1, 0.3, 4.2, 3.3, 2.8)
case("e_tiny", c(2.0, 3.1, 1.2, 5.5, 4.1, 3.0), cbind(1, x1))
n <- 40; x1 <- rnorm(n); case("forced_approx", 1 + x1 + rnorm(n), cbind(1, x1), exact = FALSE)
# Normal approximation (n >= 100).
n <- 150; case("a_int", ar1(n, 0.2), matrix(1, n, 1))
n <- 200; x1 <- rnorm(n); x2 <- rnorm(n)
case("a_two_ar", 1 + x1 + x2 + ar1(n, 0.4), cbind(1, x1, x2))
n <- 120; x1 <- rnorm(n); x2 <- runif(n); x3 <- seq_len(n) / n
case("a_three_neg", x1 + x2 + x3 + ar1(n, -0.3), cbind(1, x1, x2, x3))
n <- 110; x1 <- rnorm(n); case("forced_exact", 1 + x1 + ar1(n, 0.3), cbind(1, x1), exact = TRUE)

out <- do.call(rbind, rows)
out <- out[!duplicated(out[, c("case", "key")]), ]
dir.create("R/data", showWarnings = FALSE)
write.csv(out, "R/data/dwtest_reference.csv", row.names = FALSE)
cat("wrote", nrow(out), "rows\n")
