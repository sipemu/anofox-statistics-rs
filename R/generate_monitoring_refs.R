# R reference values for the monitoring and metric modules:
#   * isotonic calibration: isoreg() + linear interpolation (approx, rule = 2)
#     on distinct scores;
#   * Platt calibration: glm(t ~ score, quasibinomial) on Platt's smoothed
#     targets (a = -slope, b = -intercept);
#   * Pca::reconstruct / reconstruction_error: prcomp(scale. = TRUE), first k
#     components;
#   * classification metrics: yardstick (binary, macro, micro, macro_weighted);
#   * regression metrics: base R and yardstick::smape.
#
#   Rscript R/generate_monitoring_refs.R      (from the crate root)
#
# Writes R/data/monitoring_r_reference.csv (case, key, values; "%.17g", space
# separated; matrices row-major). The NannyML / scikit-learn / evidently
# references are in R/generate_monitoring_refs.py and R/generate_metrics_refs.py.
suppressPackageStartupMessages(library(yardstick))
rows <- list()
f17 <- function(x) paste(sprintf("%.17g", as.numeric(x)), collapse = " ")
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

# Isotonic calibration (distinct scores).
set.seed(365)
s <- runif(300)^1.5
y <- as.integer(runif(300) < sqrt(s))
fit <- isoreg(s, y)
xs <- sort(s)
yf <- fit$yf
newx <- c(-0.5, seq(0, 1, length.out = 41), 1.5)
add("isotonic", "x", s); add("isotonic", "y", y); add("isotonic", "new", newx)
add("isotonic", "pred", approx(xs, yf, xout = newx, rule = 2, ties = "ordered")$y)

# Platt calibration.
f <- rnorm(250)
yp <- as.integer(runif(250) < plogis(1.7 * f - 0.4))
n1 <- sum(yp); n0 <- length(yp) - n1
t <- ifelse(yp == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
g <- suppressWarnings(glm(t ~ f, family = quasibinomial, control = glm.control(epsilon = 1e-14, maxit = 100)))
add("platt", "x", f); add("platt", "y", yp); add("platt", "ab", -rev(coef(g)))

# PCA reconstruction.
n <- 50
z <- rnorm(n)
X <- cbind(z + rnorm(n, sd = 0.3), 5 * z + rnorm(n), rnorm(n), 2 - z + rnorm(n, sd = 0.5))
newX <- X[1:9, ] + 0.2 * matrix(rnorm(36), 9)
p <- prcomp(X, center = TRUE, scale. = TRUE)
add("reconstruction", "in_x", t(X)); add("reconstruction", "dims", dim(X))
add("reconstruction", "in_new", t(newX)); add("reconstruction", "new_rows", nrow(newX))
for (k in 1:3) {
  sc <- predict(p, newX)[, 1:k, drop = FALSE]
  zhat <- sc %*% t(p$rotation[, 1:k, drop = FALSE])
  zs <- scale(newX, center = p$center, scale = p$scale)
  add("reconstruction", paste0("error_k", k), sqrt(rowSums((zs - zhat)^2)))
  rec <- sweep(sweep(zhat, 2, p$scale, "*"), 2, p$center, "+")
  add("reconstruction", paste0("reconstruct_k", k), t(rec))
}
# Unscaled.
pu <- prcomp(X, center = TRUE, scale. = FALSE)
sc <- predict(pu, newX)[, 1:2]
add("reconstruction", "error_unscaled_k2",
    sqrt(rowSums((sweep(newX, 2, pu$center) - sc %*% t(pu$rotation[, 1:2]))^2)))

# Classification metrics (yardstick). Labels 0..K-1; yardstick's event is the
# first factor level, so the binary case uses levels c("1", "0").
cls <- function(case, yt, yp, levels, estimators) {
  d <- data.frame(truth = factor(yt, levels = levels), estimate = factor(yp, levels = levels))
  add(case, "y_true", yt); add(case, "y_pred", yp)
  add(case, "accuracy", accuracy(d, truth, estimate)$.estimate)
  add(case, "mcc", mcc(d, truth, estimate)$.estimate)
  add(case, "kappa", kap(d, truth, estimate)$.estimate)
  for (e in estimators) {
    v <- c(precision(d, truth, estimate, estimator = e)$.estimate,
           recall(d, truth, estimate, estimator = e)$.estimate,
           f_meas(d, truth, estimate, estimator = e)$.estimate,
           spec(d, truth, estimate, estimator = e)$.estimate)
    add(case, e, v)
  }
}
yt <- as.integer(runif(200) < 0.4)
ypb <- ifelse(runif(200) < 0.75, yt, 1 - yt)
cls("binary", yt, ypb, c("1", "0"), c("binary", "macro", "micro", "macro_weighted"))
yt <- sample(0:3, 160, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1))
ypm <- ifelse(runif(160) < 0.6, yt, sample(0:3, 160, replace = TRUE))
cls("multiclass", yt, ypm, as.character(0:3), c("macro", "micro", "macro_weighted"))

# Regression metrics.
y <- rnorm(40, 5, 2)
yh <- y + rnorm(40, 0, 0.7)
add("regression", "y_true", y); add("regression", "y_pred", yh)
e <- y - yh
add("regression", "mae", mean(abs(e)))
add("regression", "rmse", sqrt(mean(e^2)))
add("regression", "mape", mean(abs(e / y)))
add("regression", "smape", smape_vec(y, yh) / 100)
add("regression", "median_ae", median(abs(e)))
add("regression", "r2", 1 - sum(e^2) / sum((y - mean(y))^2))
add("regression", "explained_variance", 1 - var(e) / var(y))
add("regression", "max_error", max(abs(e)))

write.csv(do.call(rbind, rows), "R/data/monitoring_r_reference.csv", row.names = FALSE)
cat(length(rows), "rows\n")
