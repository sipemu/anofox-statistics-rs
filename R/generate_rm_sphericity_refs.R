# Reference values for the sphericity test and corrections of the one-way
# repeated-measures ANOVA: afex::aov_ez (GG / HF epsilons and corrected
# p-values, Mauchly) and stats::mauchly.test (W, p-value with the second-order
# term). Two datasets each with k = 3 and k = 4 conditions (the existing
# rm_anova_*cond_data.csv and a simulated one violating sphericity).
#
#   Rscript R/generate_rm_sphericity_refs.R      (run from the crate root)
#
# Writes R/data/rm_sphericity_reference.csv with columns case, key, values
# (values: space-separated, "%.17g"; in_data is the n x k matrix row-major).
.libPaths(c("~/R/userlib", .libPaths()))
suppressMessages(library(afex))

rows <- list()
f17 <- function(x) paste(sprintf("%.17g", as.numeric(x)), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

from_long <- function(file) {
  d <- read.csv(file)
  as.matrix(reshape(d, idvar = "subject", timevar = "condition", direction = "wide")[, -1])
}
set.seed(1959)
sim <- function(n, k) {
  # Strongly unequal variances of the differences: violates sphericity.
  s <- rnorm(n)
  sapply(1:k, function(j) 0.4 * j + s * j + rnorm(n, sd = 0.3 * j))
}
cases <- list(k3_a = from_long("R/data/rm_anova_3cond_data.csv"),
              k3_b = sim(14, 3),
              k4_a = from_long("R/data/rm_anova_4cond_data.csv"),
              k4_b = sim(12, 4))

for (nm in names(cases)) {
  Y <- unname(cases[[nm]]); n <- nrow(Y); k <- ncol(Y)
  long <- data.frame(id = factor(rep(1:n, k)), cond = factor(rep(1:k, each = n)), y = as.vector(Y))
  a <- suppressWarnings(suppressMessages(aov_ez("id", "y", long, within = "cond")))
  s <- summary(a)
  adj <- s$pval.adjustments
  sph <- s$sphericity.tests
  m <- mauchly.test(lm(Y ~ 1), X = ~1)
  add(nm, "in_data", t(Y)); add(nm, "dims", c(n, k))
  add(nm, "F", s$univariate.tests["cond", "F value"])
  add(nm, "gg", c(adj["cond", "GG eps"], adj["cond", "Pr(>F[GG])"]))
  add(nm, "hf", c(adj["cond", "HF eps"], adj["cond", "Pr(>F[HF])"]))
  add(nm, "mauchly_afex", c(sph["cond", "Test statistic"], sph["cond", "p-value"]))
  add(nm, "mauchly_stats", c(m$statistic, m$p.value))
  cat(nm, ": GG", adj["cond", "GG eps"], " HF", adj["cond", "HF eps"],
      " W", m$statistic, " p", m$p.value, "\n")
}

out <- do.call(rbind, rows)
write.csv(out, "R/data/rm_sphericity_reference.csv", row.names = FALSE)
cat("Wrote R/data/rm_sphericity_reference.csv (", nrow(out), "rows)\n")
