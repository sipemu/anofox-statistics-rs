# Reference values for
#   * ICC: all six Shrout-Fleiss types with F test and confidence interval
#     (psych::ICC, ANOVA estimates: lmer = FALSE);
#   * t-tests: one-sample, paired and two-sample t.test with one- and
#     two-sided alternatives (one-sided confidence intervals);
#   * block bootstrap of the mean: boot::tsboot(sim = "geom" / "fixed") and
#     the IID boot::boot (Monte-Carlo reference, B = 20000).
#
#   Rscript R/generate_icc_ttest_boot_refs.R      (run from the crate root)
#
# Writes R/data/icc_ttest_boot_reference.csv with columns case, key, values
# (values: space-separated, "%.17g", Inf/-Inf for infinite bounds).
.libPaths(c("~/R/userlib", .libPaths()))
suppressMessages({
  library(psych)
  library(boot)
})

rows <- list()
f17 <- function(x) paste(ifelse(is.na(x), "NA", ifelse(is.infinite(x), ifelse(x > 0, "Inf", "-Inf"),
                                                        sprintf("%.17g", as.numeric(x)))), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

# ---------------------------------------------------------------------------
# ICC
# ---------------------------------------------------------------------------
# Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges.
sf <- matrix(c(9, 2, 5, 8,
               6, 1, 3, 2,
               8, 4, 6, 8,
               7, 1, 2, 6,
               10, 5, 6, 9,
               6, 2, 4, 7), ncol = 4, byrow = TRUE)
# A second, larger design: 20 subjects x 3 raters with rater bias.
set.seed(1979)
subj <- rnorm(20, 50, 8)
icc2 <- round(sapply(c(0, 2.5, -1.5), function(b) subj + b + rnorm(20, 0, 4)), 1)
# A third with weak agreement (wide intervals, negative lower bounds).
set.seed(1996)
common <- rnorm(12)
icc3 <- round(sapply(1:5, function(j) 0.4 * common + rnorm(12)), 2)

for (nm in c("sf", "icc2", "icc3")) {
  x <- get(nm)
  r <- psych::ICC(x, lmer = FALSE)$results
  add(paste0("icc_", nm), "in_data", t(x))   # row-major
  add(paste0("icc_", nm), "dims", dim(x))
  for (ty in r$type) {
    row <- r[r$type == ty, ]
    add(paste0("icc_", nm), ty, c(row$ICC, row$F, row$df1, row$df2, row$p,
                                  row$`lower bound`, row$`upper bound`))
  }
}

# ---------------------------------------------------------------------------
# t-tests
# ---------------------------------------------------------------------------
set.seed(1908)
x <- round(rnorm(12, 5.4, 1.1), 2)
y <- round(rnorm(15, 4.6, 1.6), 2)
xp <- round(rnorm(10, 3, 1), 2)
yp <- round(xp - 0.4 + rnorm(10, 0, 0.6), 2)
add("ttest", "in_x", x); add("ttest", "in_y", y)
add("ttest", "in_xp", xp); add("ttest", "in_yp", yp)
tt <- function(case, r) add("ttest", case, c(r$statistic, r$parameter, r$p.value,
                                             r$conf.int[1], r$conf.int[2]))
for (alt in c("two.sided", "less", "greater")) {
  for (cl in c(0.95, 0.9)) {
    sfx <- paste0("_", alt, "_", cl)
    tt(paste0("one", sfx), t.test(x, mu = 5, alternative = alt, conf.level = cl))
    tt(paste0("paired", sfx), t.test(xp, yp, paired = TRUE, mu = 0.2, alternative = alt, conf.level = cl))
    tt(paste0("welch", sfx), t.test(x, y, mu = 0.3, alternative = alt, conf.level = cl))
    tt(paste0("student", sfx), t.test(x, y, var.equal = TRUE, mu = 0.3, alternative = alt, conf.level = cl))
  }
}
# Cohen's d for the one-sample test (effectsize::cohens_d(x, mu = 5)).
add("ttest", "cohens_d_one", (mean(x) - 5) / sd(x))

# ---------------------------------------------------------------------------
# Block bootstrap of the mean (Monte-Carlo reference)
# ---------------------------------------------------------------------------
set.seed(1994)
z <- as.numeric(arima.sim(list(ar = 0.6), n = 120))
add("boot", "in_x", z)
B <- 20000
# Percentile bounds as the Rust crate defines them (0-based order statistics
# floor(alpha/2 B) and min(ceil((1 - alpha/2) B), B - 1)).
perc <- function(t, cl = 0.95) {
  s <- sort(t); a <- 1 - cl
  c(s[floor(a / 2 * B) + 1], s[min(ceiling((1 - a / 2) * B), B - 1) + 1])
}
bsum <- function(case, t) add("boot", case, c(sd(t), perc(t)))
set.seed(1)
bsum("iid", boot(z, function(d, i) mean(d[i]), R = B)$t)
for (l in c(1, 5, 10)) {
  set.seed(l + 100)
  bsum(paste0("geom_l", l), tsboot(z, mean, R = B, l = l, sim = "geom")$t)
}
for (l in c(5, 10)) {
  set.seed(l + 200)
  bsum(paste0("fixed_l", l), tsboot(z, mean, R = B, l = l, sim = "fixed")$t)
}

out <- do.call(rbind, rows)
write.csv(out, "R/data/icc_ttest_boot_reference.csv", row.names = FALSE)
cat("Wrote R/data/icc_ttest_boot_reference.csv (", nrow(out), "rows)\n")
print(out[grepl("boot", out$case) & out$key != "in_x", ])
print(psych::ICC(sf, lmer = FALSE)$results)
