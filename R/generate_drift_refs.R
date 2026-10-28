# Reference values for the data-drift tests (evidently's stattests; issues
# #30-#39).
#
#   Rscript R/generate_drift_refs.R      (run from the crate root)
#
# Writes R/data/drift_reference.csv with columns
#   case, key, values   (values: space-separated, "%.17g")
# Inputs are stored as in_* keys of a case, outputs under the other keys.
# Every case uses the shared deterministic data of the issues:
#   x  <- 2*sin(0.37*(0:49)); y  <- 2*sin(0.53*(0:39)) + 0.5   # numeric
#   xc <- (0:49) %% 4;        yc <- ((0:39)*3) %% 5             # categorical
# (x / xc the reference sample, y / yc the current sample), plus extra cases
# for ties, unequal sizes and small samples.
suppressMessages({
  library(philentropy)
  library(kSamples)
})

rows <- list()
f17 <- function(x) paste(ifelse(is.na(x), "NA", sprintf("%.17g", as.numeric(x))), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}
check <- function(name, a, b, tol = 1e-10) {
  if (abs(a - b) > tol) stop(name, ": ", a, " vs ", b)
}

x  <- 2 * sin(0.37 * (0:49)); y  <- 2 * sin(0.53 * (0:39)) + 0.5
xc <- (0:49) %% 4;            yc <- ((0:39) * 3) %% 5
add("data", "in_x", x); add("data", "in_y", y)
add("data", "in_xc", xc); add("data", "in_yc", yc)

# Tied numeric data, unequal sizes.
xt <- round(2 * sin(0.37 * (0:59)), 1); yt <- round(2 * sin(0.53 * (0:34)) + 0.4, 1)
add("data", "in_xt", xt); add("data", "in_yt", yt)

# Larger samples (m * n >= 10000: R's default is the asymptotic p-value) and
# two nearly identical samples (small D, high p-value).
xl <- 2 * sin(0.11 * (0:149)); yl <- 2 * sin(0.13 * (0:79)) + 0.3
xs <- sin(0.7 * (0:29));       ys <- sin(0.7 * (0:24) + 0.2)
add("data", "in_xl", xl); add("data", "in_yl", yl)
add("data", "in_xs", xs); add("data", "in_ys", ys)
num_cases <- list(main = list(x, y), tie = list(xt, yt), large = list(xl, yl),
                  close = list(xs, ys))

# ---------------------------------------------------------------------------
# Binning (evidently get_binned_data) and the zero fill
# ---------------------------------------------------------------------------
# Equal-width Sturges bins over the pooled sample (numpy histogram_bin_edges;
# R's hist(breaks = "Sturges") uses pretty() breaks instead).
sturges_props <- function(a, b, pool = c(a, b)) {
  br <- seq(min(pool), max(pool), length.out = nclass.Sturges(pool) + 1)
  list(p = tabulate(findInterval(a, br, rightmost.closed = TRUE), length(br) - 1) / length(a),
       q = tabulate(findInterval(b, br, rightmost.closed = TRUE), length(br) - 1) / length(b))
}
# One bin per distinct pooled value.
cat_props <- function(a, b) {
  lv <- sort(unique(c(a, b)))
  list(p = as.numeric(table(factor(a, lv))) / length(a),
       q = as.numeric(table(factor(b, lv))) / length(b))
}
# Reference-quantile bins (type 7), outer edges open to +-Inf, duplicate
# breaks removed.
quantile_props <- function(a, b, n) {
  br <- unique(quantile(a, 0:n / n, names = FALSE))
  br[1] <- -Inf; br[length(br)] <- Inf
  list(p = tabulate(findInterval(a, br, rightmost.closed = TRUE), length(br) - 1) / length(a),
       q = tabulate(findInterval(b, br, rightmost.closed = TRUE), length(br) - 1) / length(b))
}
fill <- function(a) { m <- min(a[a > 0]); a[a == 0] <- if (m <= 1e-4) m / 1e6 else 1e-4; a }

binnings <- list(
  num = sturges_props(x, y),
  cat = cat_props(xc, yc),
  q10 = quantile_props(x, y, 10),
  tie_q5 = quantile_props(xt, yt, 5),
  tie_cat = cat_props(xt, yt)
)
for (nm in names(binnings)) {
  add(paste0("bins_", nm), "p", binnings[[nm]]$p)
  add(paste0("bins_", nm), "q", binnings[[nm]]$q)
}

# ---------------------------------------------------------------------------
# #35 PSI: sum((P - Q) log(P / Q)) on zero-filled proportions (= Jeffreys)
# ---------------------------------------------------------------------------
psi <- function(b) { P <- fill(b$p); Q <- fill(b$q); sum((P - Q) * log(P / Q)) }
for (nm in names(binnings)) {
  b <- binnings[[nm]]
  v <- psi(b)
  P <- fill(b$p); Q <- fill(b$q)
  check(paste("psi", nm), v,
        suppressMessages(philentropy::distance(rbind(P, Q), method = "jeffreys", unit = "log")))
  add(paste0("psi_", nm), "statistic", v)
}
check("psi num (issue)", psi(binnings$num), 0.570692693084, 1e-11)
check("psi cat (issue)", psi(binnings$cat), 1.56548983794, 1e-10)

# ---------------------------------------------------------------------------
# #36 Kullback-Leibler KL(p || q): zero fill, then renormalize (as
# scipy.stats.entropy), natural log and log2
# ---------------------------------------------------------------------------
kl <- function(b, base = exp(1)) {
  P <- fill(b$p); P <- P / sum(P); Q <- fill(b$q); Q <- Q / sum(Q)
  sum(P * log(P / Q)) / log(base)
}
for (nm in names(binnings)) {
  b <- binnings[[nm]]
  P <- fill(b$p); P <- P / sum(P); Q <- fill(b$q); Q <- Q / sum(Q)
  check(paste("kl", nm), kl(b), suppressMessages(philentropy::KL(rbind(P, Q), unit = "log")))
  check(paste("kl2", nm), kl(b, 2), suppressMessages(philentropy::KL(rbind(P, Q), unit = "log2")))
  add(paste0("kl_", nm), "statistic", kl(b))
  add(paste0("kl_", nm), "statistic_log2", kl(b, 2))
}
check("kl num (issue)", kl(binnings$num), 0.264782319603, 1e-11)
check("kl cat (issue)", kl(binnings$cat), 0.223061363402, 1e-11)

# ---------------------------------------------------------------------------
# #37 Jensen-Shannon distance: sqrt of the JS divergence, no zero fill
# ---------------------------------------------------------------------------
jsd <- function(b, base = exp(1)) {
  p <- b$p / sum(b$p); q <- b$q / sum(b$q); m <- (p + q) / 2
  re <- function(a, m) ifelse(a > 0, a * log(a / m), 0)
  sqrt(sum(re(p, m) + re(q, m)) / 2 / log(base))
}
for (nm in names(binnings)) {
  b <- binnings[[nm]]
  check(paste("js", nm), jsd(b),
        sqrt(suppressMessages(philentropy::JSD(rbind(b$p, b$q), unit = "log"))))
  check(paste("js2", nm), jsd(b, 2),
        sqrt(suppressMessages(philentropy::JSD(rbind(b$p, b$q), unit = "log2"))))
  add(paste0("js_", nm), "statistic", jsd(b))
  add(paste0("js_", nm), "statistic_log2", jsd(b, 2))
}
check("js num (issue)", jsd(binnings$num), 0.257847900615, 1e-11)
check("js cat (issue)", jsd(binnings$cat), 0.27397012217, 1e-10)

# ---------------------------------------------------------------------------
# #38 Hellinger distance sqrt(1 - sum(sqrt(p q))), no zero fill
# ---------------------------------------------------------------------------
hell <- function(b) sqrt(1 - min(1, sum(sqrt(b$p * b$q))))
for (nm in names(binnings)) {
  b <- binnings[[nm]]
  # philentropy's "hellinger" is 2 * sqrt(1 - sum(sqrt(p q))).
  check(paste("hellinger", nm), hell(b),
        suppressMessages(philentropy::distance(rbind(b$p, b$q), method = "hellinger")) / 2)
  add(paste0("hellinger_", nm), "statistic", hell(b))
}
# evidently's numeric rule: Sturges bins whose count comes from the distinct
# pooled values (histogram_bin_edges(unique(ref, cur), "sturges")).
b <- sturges_props(x, y, unique(c(x, y)))
add("hellinger_auto_num", "statistic", hell(b))
check("hellinger num (issue)", hell(b), 0.262287362723, 1e-11)
check("hellinger cat (issue)", hell(binnings$cat), 0.325194993744, 1e-11)
b <- sturges_props(xt, yt, unique(c(xt, yt)))
add("hellinger_auto_tie", "n_bins", length(b$p))
add("hellinger_auto_tie", "statistic", hell(b))
b <- sturges_props(xt, yt)
add("hellinger_sturges_tie", "n_bins", length(b$p))
add("hellinger_sturges_tie", "statistic", hell(b))

# ---------------------------------------------------------------------------
# #30 two-sample Kolmogorov-Smirnov: stats::ks.test, exact = NULL/TRUE/FALSE
# ---------------------------------------------------------------------------
for (nm in names(num_cases)) {
  a <- num_cases[[nm]][[1]]; b <- num_cases[[nm]][[2]]
  for (alt in c("two.sided", "less", "greater")) {
    for (ex in list(NULL, TRUE, FALSE)) {
      r <- suppressWarnings(ks.test(a, b, alternative = alt, exact = ex))
      key <- paste0("ks_", nm, "_", alt, "_",
                    if (is.null(ex)) "auto" else if (ex) "exact" else "asymp")
      add(key, "statistic", unname(r$statistic))
      add(key, "p_value", r$p.value)
      add(key, "exact", as.numeric(r$exact))
    }
  }
}
r <- ks.test(x, y); check("ks exact (issue)", r$p.value, 0.0574636491581, 1e-12)
r <- ks.test(x, y, exact = FALSE); check("ks asymp (issue)", r$p.value, 0.0693904727347, 1e-12)

# ---------------------------------------------------------------------------
# #31 k-sample Anderson-Darling (Scholz & Stephens 1987): kSamples::ad.test.
# ad.test rounds its table, so the unrounded statistics come from its C
# routine and the asymptotic p-values from kSamples::ad.pval.
# ---------------------------------------------------------------------------
ad_k <- function(samples) {
  k <- length(samples); ns <- sapply(samples, length); z <- unlist(samples)
  n <- length(z); Z.star <- sort(unique(z))
  a <- .C("adkTestStat0", ans = double(2), k = as.integer(k), x = as.double(z),
          ns = as.integer(ns), Z.star = as.double(Z.star), L = as.integer(length(Z.star)),
          PACKAGE = "kSamples")$ans
  H <- sum(1 / ns); h <- sum(1 / (1:(n - 1))); g <- 0
  for (i in 1:(n - 2)) g <- g + (1 / (n - i)) * sum(1 / ((i + 1):(n - 1)))
  ca <- (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
  cb <- (2 * g - 4) * k^2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
  cc <- (6 * h + 2 * g - 2) * k^2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
  cd <- (2 * h + 6) * k^2 - 4 * h * k
  sig <- sqrt((ca * n^3 + cb * n^2 + cc * n + cd) / ((n - 1) * (n - 2) * (n - 3)))
  t <- (a - (k - 1)) / sig
  # Cross-check against the (rounded) ad.test table.
  r <- ad.test(samples, method = "asymptotic")$ad
  check("ad T v1", signif(t[1], 5), r[1, 2], 1e-12); check("ad T v2", signif(t[2], 5), r[2, 2], 1e-12)
  list(ad = a, t = t, p = c(ad.pval(t[1], k - 1, 1), ad.pval(t[2], k - 1, 2)))
}
ad_cases <- list(main = list(x, y), tie = list(xt, yt), k3 = list(x, y, xt),
                 close = list(xs, ys), large = list(xl, yl),
                 k4 = list(x, y + 0.6, xt, yl), shift = list(x, y + 1.5))
for (nm in names(ad_cases)) {
  smp <- ad_cases[[nm]]
  for (j in seq_along(smp)) add(paste0("ad_", nm), paste0("in_g", j), smp[[j]])
  r <- ad_k(smp)
  for (v in 1:2) {
    add(paste0("ad_", nm), paste0("ad_v", v), r$ad[v])
    add(paste0("ad_", nm), paste0("statistic_v", v), r$t[v])
    add(paste0("ad_", nm), paste0("p_value_v", v), r$p[v])
  }
}
r <- ad_k(list(x, y))
check("ad T v2 (issue)", r$t[2], 3.61044012836, 1e-10)
check("ad T v1 (issue)", r$t[1], 3.52653061328, 1e-10)
check("ad p v2 (issue)", signif(r$p[2], 5), 0.011626, 1e-12)
# smooth.spline as used by ad.pval, for a direct check of the port.
set.seed(1); sx <- runif(35); sy <- sin(5 * sx) + rnorm(35, sd = 0.2)
add("smooth_spline", "in_x", sx); add("smooth_spline", "in_y", sy)
xp <- c(-0.2, 0.1, 0.5, 0.93, 1.3)
add("smooth_spline", "in_xp", xp)
for (spar in c(0.25, 0.4)) {
  f <- smooth.spline(sx, sy, spar = spar)
  add("smooth_spline", paste0("lambda_", spar), f$lambda)
  add("smooth_spline", paste0("pred_", spar), predict(f, xp)$y)
}

# ---------------------------------------------------------------------------
# #32 two-sample Cramer-von Mises (Anderson 1962), midranks. Asymptotic
# p-value with the limiting CDF goftest::pCvM (as scipy); exact p-value by
# full enumeration of the splits for small samples.
# ---------------------------------------------------------------------------
cvm_u <- function(rk, m, n) {
  rx <- sort(rk[1:m]); ry <- sort(rk[(m + 1):(m + n)])
  m * sum((rx - 1:m)^2) + n * sum((ry - 1:n)^2)
}
cvm2 <- function(x, y) {
  m <- length(x); n <- length(y); N <- m + n
  U  <- cvm_u(rank(c(x, y)), m, n)
  T  <- U / (m * n * N) - (4 * m * n - 1) / (6 * N)
  ET <- (1 + 1 / N) / 6
  VT <- (N + 1) * (4 * m * n * N - 3 * (m^2 + n^2) - 2 * m * n) / (45 * N^2 * 4 * m * n)
  Tn <- 1 / 6 + (T - ET) / sqrt(45 * VT)
  c(statistic = T, p_value = 1 - goftest::pCvM(Tn))
}
for (nm in names(num_cases)) {
  r <- cvm2(num_cases[[nm]][[1]], num_cases[[nm]][[2]])
  add(paste0("cvm_", nm), "statistic", r[1]); add(paste0("cvm_", nm), "p_value", r[2])
}
r <- cvm2(x, y)
check("cvm T (issue)", r[1], 0.4722592592593, 1e-12); check("cvm p (issue)", r[2], 0.0468933196396, 1e-12)
cvm_exact <- function(x, y) {
  m <- length(x); n <- length(y); N <- m + n; rk <- rank(c(x, y))
  U <- cvm_u(rk, m, n)
  us <- combn(N, m, function(ix) cvm_u(c(rk[ix], rk[-ix]), m, n))
  c(statistic = U / (m * n * N) - (4 * m * n - 1) / (6 * N), p_value = mean(us >= U - 1e-9))
}
exact_cases <- list(e54 = c(5, 4), e76 = c(7, 6), e88 = c(8, 8), e107 = c(10, 7), e39 = c(3, 9))
for (nm in names(exact_cases)) {
  m <- exact_cases[[nm]][1]; n <- exact_cases[[nm]][2]
  a <- sin(1.3 * (0:(m - 1))) ; b <- sin(0.9 * (0:(n - 1))) + 0.4
  add(paste0("cvm_", nm), "in_x", a); add(paste0("cvm_", nm), "in_y", b)
  r <- cvm_exact(a, b)
  add(paste0("cvm_", nm), "statistic", r[1]); add(paste0("cvm_", nm), "p_value", r[2])
}
qgrid <- c(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.461, 0.5, 0.75, 1, 1.5, 2, 3)
add("pcvm", "in_q", qgrid); add("pcvm", "p", goftest::pCvM(qgrid))

out <- do.call(rbind, rows)
write.csv(out, "R/data/drift_reference.csv", row.names = FALSE)
cat("wrote", nrow(out), "rows to R/data/drift_reference.csv\n")
