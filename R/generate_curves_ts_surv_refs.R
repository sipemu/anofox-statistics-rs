# Reference values for the classification, timeseries and survival modules.
#
#   Rscript R/generate_curves_ts_surv_refs.R      (run from the crate root)
#
# Writes R/data/curves_ts_surv_reference.csv with columns
#   case, key, values   (values: space-separated, "%.17g", NA -> "NA")
# Inputs are stored under keys starting with "in_".
#
# Sources:
#   ROC        pROC::roc / coords(ret = "all") (fpr, tpr) and pROC::auc
#   PR         brute force in base R: for every distinct score t,
#              precision = mean(y[s >= t]), recall = sum(y[s >= t]) / sum(y)
#              (the definition of sklearn's precision_recall_curve)
#   calibration sklearn's uniform binning re-done in base R
#              (bin = #inner edges < p), Wilson CI = prop.test(correct = FALSE)
#   brier/log  base R formulas (sklearn brier_score_loss / log_loss)
#   acf/pacf   stats::acf / stats::pacf
#   Box tests  stats::Box.test
#   DW         lmtest::dwtest(lm)$statistic
#   KM         survival::survfit (conf.type log / log-log / plain)
#   log-rank   survival::survdiff
suppressMessages({
  library(pROC)
  library(survival)
  library(lmtest)
})

rows <- list()
f17 <- function(x) {
  x <- as.numeric(x)
  paste(ifelse(is.na(x) & !is.nan(x), "NA", ifelse(is.nan(x), "NaN", sprintf("%.17g", x))),
        collapse = " ")
}
add <- function(case, key, values) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, key = key, values = f17(values))
}

# ---------------------------------------------------------------- classification
set.seed(153)
x <- rnorm(200)
p1 <- plogis(0.3 + 1.5 * x)
y1 <- rbinom(200, 1, p1)
p2 <- round(runif(150), 1)               # heavy ties, includes edges 0.3, 0.5, 1
y2 <- rbinom(150, 1, p2)
p3 <- c(0.1, 0.4, 0.35, 0.8, 0.4, 0.4, 0.9, 0.1, 0.2, 0.5, 0.3, 1.0, 0.0)
y3 <- c(0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0)
p4 <- rep(0.5, 10)
y4 <- c(1, 0, 1, 1, 0, 0, 1, 0, 1, 1)
cls <- list(c1 = list(p1, y1), c2 = list(p2, y2), c3 = list(p3, y3), c4 = list(p4, y4))
for (nm in names(cls)) {
  s <- cls[[nm]][[1]]; y <- cls[[nm]][[2]]
  add(nm, "in_scores", s); add(nm, "in_labels", y)
  r <- roc(y, s, levels = c(0, 1), direction = "<", quiet = TRUE)
  co <- coords(r, "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
  # pROC orders by increasing threshold; sklearn by decreasing.
  add(nm, "roc_fpr", rev(1 - co$specificity))
  add(nm, "roc_tpr", rev(co$sensitivity))
  add(nm, "auc", as.numeric(auc(r)))
  th <- sort(unique(s))                  # increasing, as sklearn PR thresholds
  add(nm, "pr_thresholds", th)
  add(nm, "pr_precision", c(sapply(th, function(t) mean(y[s >= t])), 1))
  add(nm, "pr_recall", c(sapply(th, function(t) sum(y[s >= t]) / sum(y)), 0))
  add(nm, "brier", mean((s - y)^2))
  eps <- .Machine$double.eps
  pc <- pmin(pmax(s, eps), 1 - eps)
  add(nm, "log_loss", -mean(y * log(pc) + (1 - y) * log(1 - pc)))
  for (nb in c(5, 10)) {
    for (cl in c(0.95, 0.9)) {
      edges <- seq(0, 1, length.out = nb + 1)[2:nb]
      b <- sapply(s, function(v) sum(edges < v))
      ub <- sort(unique(b))
      ci <- t(sapply(ub, function(k) {
        prop.test(sum(y[b == k]), sum(b == k), conf.level = cl, correct = FALSE)$conf.int
      }))
      key <- sprintf("cal_%d_%g_", nb, cl)
      add(nm, paste0(key, "bin"), ub)
      add(nm, paste0(key, "mean_pred"), sapply(ub, function(k) mean(s[b == k])))
      add(nm, paste0(key, "frac_pos"), sapply(ub, function(k) mean(y[b == k])))
      add(nm, paste0(key, "n"), sapply(ub, function(k) sum(b == k)))
      add(nm, paste0(key, "low"), ci[, 1])
      add(nm, paste0(key, "high"), ci[, 2])
    }
  }
}

# ---------------------------------------------------------------- timeseries
set.seed(155)
t1 <- as.numeric(arima.sim(list(ar = 0.6), n = 120))
t2 <- rnorm(50)
t3 <- c(1.2, 3.4, 2.2, 5.1, 4.0, 6.3, 5.5, 7.9, 6.6, 9.1, 8.0, 10.4, 9.9, 12.2, 11.0)
xreg <- seq_len(80)
yreg <- 2 + 0.5 * xreg + as.numeric(arima.sim(list(ar = 0.5), n = 80))
res4 <- residuals(lm(yreg ~ xreg))
ts_cases <- list(t1 = t1, t2 = t2, t3 = t3, t4 = res4)
for (nm in names(ts_cases)) {
  v <- ts_cases[[nm]]
  n <- length(v)
  ml <- min(20, n - 1)
  add(nm, "in_x", v)
  add(nm, "max_lag", ml)
  add(nm, "acf", acf(v, lag.max = ml, plot = FALSE)$acf[, 1, 1])
  add(nm, "pacf", pacf(v, lag.max = ml, plot = FALSE)$acf[, 1, 1])
  add(nm, "bound", qnorm(0.975) / sqrt(n))
  for (lag in c(1, 5, 10)) for (fitdf in c(0, 1, 2)) {
    if (fitdf >= lag || lag >= n) next
    lb <- Box.test(v, lag = lag, type = "Ljung-Box", fitdf = fitdf)
    bp <- Box.test(v, lag = lag, type = "Box-Pierce", fitdf = fitdf)
    add(nm, sprintf("lb_%d_%d", lag, fitdf), c(lb$statistic, lb$parameter, lb$p.value))
    add(nm, sprintf("bp_%d_%d", lag, fitdf), c(bp$statistic, bp$parameter, bp$p.value))
  }
}
add("t4", "dw", dwtest(yreg ~ xreg)$statistic)
add("t2", "dw", dwtest(lm(t2 ~ 1))$statistic)
add("t1", "dw", dwtest(lm(t1 ~ seq_along(t1)))$statistic)
add("t1", "in_dw_resid", residuals(lm(t1 ~ seq_along(t1))))
add("t2", "in_dw_resid", residuals(lm(t2 ~ 1)))
add("t4", "in_dw_resid", res4)

# ---------------------------------------------------------------- survival
lung2 <- lung[!is.na(lung$ph.ecog), ]
set.seed(156)
ns <- 90
g3 <- sample(c("a", "b", "c"), ns, replace = TRUE)
rate <- c(a = 0.10, b = 0.15, c = 0.25)[g3]
tt <- round(rexp(ns, rate))               # many ties
cc <- round(runif(ns, 2, 25))
s3time <- pmax(pmin(tt, cc), 1)
s3ev <- as.numeric(tt <= cc)
surv_cases <- list(
  s1 = list(lung2$time, as.numeric(lung2$status == 2), lung2$ph.ecog),
  s2 = list(c(1, 2, 2, 2, 3, 3, 4, 5, 5, 6), c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
            c(1, 2, 1, 2, 1, 2, 1, 2, 2, 1)),
  s3 = list(s3time, s3ev, match(g3, c("a", "b", "c"))),
  s4 = list(lung$time, as.numeric(lung$status == 2), lung$sex)
)
for (nm in names(surv_cases)) {
  tm <- surv_cases[[nm]][[1]]; ev <- surv_cases[[nm]][[2]]; gr <- surv_cases[[nm]][[3]]
  add(nm, "in_time", tm); add(nm, "in_event", ev); add(nm, "in_group", gr)
  for (ct in c("log", "log-log", "plain")) for (cl in c(0.95, 0.9)) {
    f <- survfit(Surv(tm, ev) ~ 1, conf.type = ct, conf.int = cl)
    key <- sprintf("km_%s_%g_", ct, cl)
    if (ct == "log" && cl == 0.95) {
      add(nm, "km_time", f$time); add(nm, "km_n_risk", f$n.risk)
      add(nm, "km_n_event", f$n.event); add(nm, "km_n_censor", f$n.censor)
      add(nm, "km_surv", f$surv); add(nm, "km_std_err_log", f$std.err)
      add(nm, "km_std_err", summary(f, censored = TRUE)$std.err)
    }
    add(nm, paste0(key, "low"), f$lower); add(nm, paste0(key, "high"), f$upper)
  }
  sd <- survdiff(Surv(tm, ev) ~ gr)
  add(nm, "lr_obs", sd$obs); add(nm, "lr_exp", sd$exp); add(nm, "lr_n", sd$n)
  add(nm, "lr_stat", c(sd$chisq, length(sd$n) - 1, sd$pvalue))
}

out <- do.call(rbind, rows)
write.csv(out, "R/data/curves_ts_surv_reference.csv", row.names = FALSE)
cat("wrote", nrow(out), "rows\n")
