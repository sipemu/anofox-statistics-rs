# Reference values for p_adjust, ptukey/qtukey, pairwise_t_test, tukey_hsd and
# dunn_test (tests/posthoc_reference.rs).
# Run with: Rscript R/generate_posthoc_refs.R   (needs the dunn.test package)
suppressMessages(library(dunn.test))
set.seed(20261009)
f17 <- function(v) paste(ifelse(is.na(v), "NA", sprintf("%.17g", v)), collapse = " ")
out <- function(df, name) write.csv(df, file.path("R/data", name), row.names = FALSE)

# ---- p.adjust ---------------------------------------------------------------
pvecs <- list(
  c(0.01, 0.02, 0.03, 0.04, 0.05),
  c(0.2, 0.001, 0.04, 0.04, 0.9, 0.0001, 0.3, 0.04),
  c(NA, 0.03, 0.5, NA, 0.001, 0.02),
  c(0.04, 0.01),
  c(0.5),
  c(NA, 0.3),
  runif(40)^3,
  c(runif(15)^2, NA, NA, 1, 1, 0)
)
methods <- c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none")
pa <- do.call(rbind, lapply(seq_along(pvecs), function(i) do.call(rbind, lapply(methods, function(m)
  data.frame(case = i, method = m, p = f17(pvecs[[i]]), adj = f17(p.adjust(pvecs[[i]], m)))))))
out(pa, "p_adjust_reference.csv")

# ---- ptukey / qtukey --------------------------------------------------------
pt <- expand.grid(q = c(0.1, 0.5, 1, 2, 3.3, 4, 5.5, 8, 15), nmeans = c(2, 3, 5, 10, 20),
                  df = c(2, 5, 10, 30, 120, 1000, 30000), nranges = c(1, 3))
pt$lower <- sprintf("%.17g", ptukey(pt$q, pt$nmeans, pt$df, pt$nranges))
pt$upper <- sprintf("%.17g", ptukey(pt$q, pt$nmeans, pt$df, pt$nranges, lower.tail = FALSE))
out(pt, "ptukey_reference.csv")
qt <- expand.grid(p = c(0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999), nmeans = c(2, 3, 5, 10, 20),
                  df = c(2, 5, 10, 30, 120, 1000), nranges = c(1, 2))
qt$q <- sprintf("%.17g", qtukey(qt$p, qt$nmeans, qt$df, qt$nranges))
out(qt, "qtukey_reference.csv")

# ---- data sets for the pairwise tests ---------------------------------------
mk <- function(sizes, shifts, round_to = NULL) {
  g <- rep(LETTERS[seq_along(sizes)], sizes)
  x <- rnorm(sum(sizes), rep(shifts, sizes), rep(seq_along(sizes) * 0.5, sizes))
  if (!is.null(round_to)) x <- round(x / round_to) * round_to
  idx <- sample(length(x)); list(x = x[idx], g = g[idx])
}
sets <- list(
  mk(c(8, 8, 8), c(0, 0.5, 1.5)),
  mk(c(5, 12, 7, 9, 15), c(0, 1, 1, 2, -0.5)),
  mk(c(10, 14, 6, 11), c(0, 0.3, 1, 0.6), round_to = 0.5),   # ties
  mk(c(20, 25), c(0, 0.8)),
  mk(c(30, 3, 12, 18, 9, 22), c(0, 2, 1, 1.5, 0.2, -1))
)

rows <- list()
add <- function(...) rows[[length(rows) + 1]] <<- data.frame(...)
for (s in seq_along(sets)) {
  x <- sets[[s]]$x; g <- sets[[s]]$g; lv <- sort(unique(g)); k <- length(lv)
  m <- tapply(x, g, mean); n <- tapply(x, g, length); v <- tapply(x, g, var)
  # pairwise.t.test
  for (pool in c(TRUE, FALSE)) for (alt in c("two.sided", "less", "greater"))
    for (meth in c("holm", "BH", "none", "bonferroni", "hommel")) {
      pr <- pairwise.t.test(x, g, p.adjust.method = "none", pool.sd = pool, alternative = alt)$p.value
      pj <- pairwise.t.test(x, g, p.adjust.method = meth, pool.sd = pool, alternative = alt)$p.value
      sp <- sqrt(sum(v * (n - 1)) / sum(n - 1))
      for (j in 1:(k - 1)) for (i in (j + 1):k) {
        a <- lv[j]; b <- lv[i]
        if (pool) {
          st <- (m[b] - m[a]) / (sp * sqrt(1 / n[a] + 1 / n[b])); df <- sum(n - 1)
        } else {
          tt <- t.test(x[g == b], x[g == a]); st <- tt$statistic; df <- tt$parameter
        }
        add(set = s, test = "pairwise_t", pooled = pool, alternative = alt, option = meth,
            group1 = a, group2 = b, estimate = sprintf("%.17g", m[b] - m[a]),
            statistic = sprintf("%.17g", st), df = sprintf("%.17g", df),
            p_value = sprintf("%.17g", pr[b, a]), p_adj = sprintf("%.17g", pj[b, a]),
            conf_low = "NA", conf_high = "NA")
      }
    }
  # TukeyHSD
  for (cl in c(0.95, 0.9, 0.99)) {
    th <- TukeyHSD(aov(x ~ factor(g)), conf.level = cl)[[1]]
    for (j in 1:(k - 1)) for (i in (j + 1):k) {
      r <- th[paste0(lv[i], "-", lv[j]), ]
      add(set = s, test = "tukey", pooled = NA, alternative = "two.sided", option = cl,
          group1 = lv[j], group2 = lv[i], estimate = sprintf("%.17g", r["diff"]),
          statistic = "NA", df = sprintf("%.17g", length(x) - k),
          p_value = sprintf("%.17g", r["p adj"]), p_adj = sprintf("%.17g", r["p adj"]),
          conf_low = sprintf("%.17g", r["lwr"]), conf_high = sprintf("%.17g", r["upr"]))
    }
  }
  # Dunn (dunn.test, two-sided p; adjusted like FSA::dunnTest via p.adjust).
  # g must be a factor: dunn.test 1.4.2 mis-ranks character group vectors.
  invisible(capture.output(dt <- dunn.test(x, factor(g), method = "none", altp = TRUE, kw = FALSE, table = FALSE)))
  for (meth in c("holm", "BH", "none", "bonferroni", "hochberg", "BY")) {
    padj <- p.adjust(dt$altP, meth)
    for (r in seq_along(dt$comparisons)) {
      pr <- strsplit(dt$comparisons[r], " - ")[[1]]
      add(set = s, test = "dunn", pooled = NA, alternative = "two.sided", option = meth,
          group1 = pr[1], group2 = pr[2], estimate = "NA",
          statistic = sprintf("%.17g", dt$Z[r]), df = "NA",
          p_value = sprintf("%.17g", dt$altP[r]), p_adj = sprintf("%.17g", padj[r]),
          conf_low = "NA", conf_high = "NA")
    }
  }
}
out(do.call(rbind, rows), "posthoc_reference.csv")
out(data.frame(set = seq_along(sets), x = sapply(sets, function(d) f17(d$x)),
               g = sapply(sets, function(d) paste(d$g, collapse = " "))), "posthoc_data.csv")
