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

out <- do.call(rbind, rows)
write.csv(out, "R/data/drift_reference.csv", row.names = FALSE)
cat("wrote", nrow(out), "rows to R/data/drift_reference.csv\n")
