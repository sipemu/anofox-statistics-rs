# Reference values for
#   * multivariate::pca_with_options(scale = true): prcomp(scale. = TRUE) and
#     predict.prcomp (component signs: largest |loading| positive), a tall
#     (n > d) and a wide (n <= d, Gram path) design;
#   * the MMD median-heuristic bandwidths: median(dist(rbind(x, y))) (pooled,
#     floored at 0.01; also 1-D), and evidently's embedding MMD bandwidth
#     median(cross distances) / sqrt(2); the unbiased MMD^2 with the pooled
#     median bandwidth (mmd_test with Kernel::GaussianMedianHeuristic).
#
#   Rscript R/generate_pca_mmd_refs.R      (run from the crate root)
#
# Writes R/data/pca_mmd_reference.csv (case, key, values; "%.17g";
# matrices row-major).
rows <- list()
f17 <- function(x) paste(sprintf("%.17g", as.numeric(x)), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

pca_ref <- function(key, X, newdata) {
  p <- prcomp(X, center = TRUE, scale. = TRUE)
  k <- min(nrow(X) - 1, ncol(X))
  rot <- p$rotation[, 1:k, drop = FALSE]
  sg <- apply(rot, 2, function(v) sign(v[which.max(abs(v))]))
  rot <- sweep(rot, 2, sg, "*")
  add(key, "in_x", t(X)); add(key, "dims", dim(X))
  add(key, "in_new", t(newdata)); add(key, "new_rows", nrow(newdata))
  add(key, "center", p$center)
  add(key, "scale", p$scale)
  add(key, "sdev", p$sdev[1:k])
  add(key, "rotation", t(rot))
  add(key, "explained", p$sdev[1:k]^2 / sum(p$sdev^2))
  add(key, "scores", t(sweep(predict(p, newdata)[, 1:k, drop = FALSE], 2, sg, "*")))
}

set.seed(1901)
n <- 40
z <- rnorm(n)
tall <- cbind(z + rnorm(n, sd = 0.5), 10 * (z + rnorm(n)), 0.01 * rnorm(n), 3 + 2 * z)
pca_ref("pca_tall", tall, tall[1:7, ] + 0.1)
wide <- matrix(sin(0.7 * (1:60)) * rep(c(1, 5, 20, 0.5, 2, 9, 1, 3, 4, 7), each = 6) +
                 0.3 * cos(0.11 * (1:60)^1.3), nrow = 6)
pca_ref("pca_wide", wide, wide)

# MMD bandwidths and statistic.
mmd2 <- function(X, Y, bw) {
  K <- function(A, B) exp(-as.matrix(dist(rbind(A, B)))^2 / (2 * bw^2))[1:nrow(A), nrow(A) + 1:nrow(B), drop = FALSE]
  m <- nrow(X); nn <- nrow(Y)
  kxx <- K(X, X); kyy <- K(Y, Y); kxy <- K(X, Y)
  (sum(kxx) - sum(diag(kxx))) / (m * (m - 1)) + (sum(kyy) - sum(diag(kyy))) / (nn * (nn - 1)) -
    2 * sum(kxy) / (m * nn)
}
set.seed(77)
X <- matrix(rnorm(25 * 3), 25); Y <- matrix(rnorm(20 * 3, mean = 0.4), 20)
add("mmd", "in_x", t(X)); add("mmd", "in_y", t(Y)); add("mmd", "dim", 3)
bw <- max(median(dist(rbind(X, Y))), 0.01)
add("mmd", "bandwidth", bw)
add("mmd", "statistic", mmd2(X, Y, bw))
D <- as.matrix(dist(rbind(X, Y)))
add("mmd", "embedding_bandwidth", median(D[1:25, 25 + 1:20]) / sqrt(2))
x1 <- rnorm(30); y1 <- rnorm(24, 0.5)
add("mmd1", "in_x", x1); add("mmd1", "in_y", y1)
add("mmd1", "bandwidth", max(median(dist(c(x1, y1))), 0.01))

out <- do.call(rbind, rows)
write.csv(out, "R/data/pca_mmd_reference.csv", row.names = FALSE)
cat("Wrote R/data/pca_mmd_reference.csv (", nrow(out), "rows)\n")
