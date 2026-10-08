# Reference estimates / confidence intervals of wilcox.test(conf.int = TRUE)
# for the Hodges-Lehmann regression tests (tests/wilcoxon_ci_reference.rs).
# Run with: Rscript R/generate_wilcox_ci_refs.R
set.seed(20261008)
fmt <- function(v) paste(sprintf("%.17g", v), collapse = " ")
rows <- list()
add <- function(kind, x, y, alternative, correct, exact, conf, mu) {
  r <- if (kind == "mw") {
    wilcox.test(x, y, alternative = alternative, mu = mu, exact = exact,
                correct = correct, conf.int = TRUE, conf.level = conf)
  } else {
    wilcox.test(x, y, paired = TRUE, alternative = alternative, mu = mu,
                exact = exact, correct = correct, conf.int = TRUE,
                conf.level = conf)
  }
  rows[[length(rows) + 1]] <<- data.frame(
    kind = kind, alternative = alternative, correct = correct, exact = exact,
    conf_level = conf, mu = mu, estimate = sprintf("%.17g", unname(r$estimate)),
    lower = sprintf("%.17g", r$conf.int[1]), upper = sprintf("%.17g", r$conf.int[2]),
    x = fmt(x), y = fmt(y))
}
alts <- c("two.sided", "less", "greater")
# Mann-Whitney, normal approximation (with and without ties)
for (sz in list(c(15, 12), c(60, 45), c(200, 150))) {
  x <- rnorm(sz[1], 0.4); y <- rnorm(sz[2])
  xt <- round(x * 2) / 2; yt <- round(y * 2) / 2
  for (a in alts) for (cc in c(TRUE, FALSE)) {
    add("mw", x, y, a, cc, FALSE, 0.95, 0)
    add("mw", xt, yt, a, cc, FALSE, 0.9, 0.25)
  }
}
# Mann-Whitney exact (no ties, small samples)
for (sz in list(c(10, 8), c(20, 25), c(49, 30))) {
  x <- rnorm(sz[1], 0.5); y <- rnorm(sz[2])
  for (a in alts) for (conf in c(0.95, 0.8)) add("mw", x, y, a, FALSE, TRUE, conf, 0)
}
# Signed rank, normal approximation (ties and zeros)
for (n in c(25, 80, 300)) {
  x <- rnorm(n, 0.3); y <- rnorm(n)
  xt <- round(x * 2) / 2; yt <- round(y * 2) / 2
  for (a in alts) for (cc in c(TRUE, FALSE)) {
    add("wsr", x, y, a, cc, FALSE, 0.95, 0)
    add("wsr", xt, yt, a, cc, FALSE, 0.9, 0.5)
  }
}
# Signed rank exact (no ties / zeros)
for (n in c(12, 30, 49)) {
  x <- rnorm(n, 0.3); y <- rnorm(n)
  for (a in alts) for (conf in c(0.95, 0.8)) {
    add("wsr", x, y, a, FALSE, TRUE, conf, 0)
    add("wsr", x, y, a, FALSE, TRUE, conf, 0.1)
  }
}
out <- do.call(rbind, rows)
write.csv(out, "R/data/wilcox_ci_reference.csv", row.names = FALSE)
cat("Generated: R/data/wilcox_ci_reference.csv (", nrow(out), "cases )\n")
