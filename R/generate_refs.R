# Reference Data Generator for libanostat TDD
# Run with: Rscript R/generate_refs.R

# Set seed for reproducibility
set.seed(42)

# Ensure output directory exists
dir.create("R/data", showWarnings = FALSE, recursive = TRUE)

# Helper function to save reference data
save_ref <- function(filename, data_list) {
  df <- as.data.frame(lapply(data_list, unlist))
  filepath <- file.path("R/data", filename)
  write.csv(df, filepath, row.names = FALSE)
  cat(paste("Generated:", filepath, "\n"))
}

# Helper to save vectors (one per row)
save_vector <- function(filename, vec, col_name = "value") {
  df <- data.frame(value = vec)
  names(df) <- col_name
  filepath <- file.path("R/data", filename)
  write.csv(df, filepath, row.names = FALSE)
  cat(paste("Generated:", filepath, "\n"))
}

# ============================================
# PHASE 1: Math Primitives References
# ============================================

cat("=== Phase 1: Math Primitives ===\n")

# Test vectors
x_short <- c(1.2, 2.3, 3.4, 4.5, 5.6)
x_long <- rnorm(100, mean = 10, sd = 2)
x_outlier <- c(1, 1, 1, 1, 100)
x_even <- c(1.0, 2.0, 3.0, 4.0)  # Even length for median edge case
x_single <- c(42.0)  # Single element

# Save test vectors for Rust to use
save_vector("vec_short.csv", x_short)
save_vector("vec_long.csv", x_long)
save_vector("vec_outlier.csv", x_outlier)
save_vector("vec_even.csv", x_even)

# Basic statistics
save_ref("math_basic.csv", list(
  # Short vector
  mean_short = mean(x_short),
  var_short = var(x_short),
  median_short = median(x_short),
  trim_20_short = mean(x_short, trim = 0.2),

  # Long vector (100 elements)
  mean_long = mean(x_long),
  var_long = var(x_long),
  median_long = median(x_long),
  trim_20_long = mean(x_long, trim = 0.2),

  # Outlier vector - tests robustness
  mean_outlier = mean(x_outlier),
  var_outlier = var(x_outlier),
  median_outlier = median(x_outlier),
  trim_20_outlier = mean(x_outlier, trim = 0.2),

  # Even-length vector for median
  median_even = median(x_even),

  # Single element
  mean_single = mean(x_single),
  median_single = median(x_single)
))

# ============================================
# PHASE 2: Parametric Tests References
# ============================================

cat("\n=== Phase 2: Parametric Tests ===\n")

# Generate two groups for t-tests
# Group 1: n=20, mean=5, sd=1
# Group 2: n=25, mean=5.5, sd=1.5 (unequal variance)
g1 <- rnorm(20, mean = 5, sd = 1)
g2 <- rnorm(25, mean = 5.5, sd = 1.5)

# Save the test vectors
save_vector("ttest_g1.csv", g1)
save_vector("ttest_g2.csv", g2)

# Equal-sized groups for paired test
g1_paired <- rnorm(15, mean = 10, sd = 2)
g2_paired <- g1_paired + rnorm(15, mean = 0.5, sd = 0.5)  # Correlated

save_vector("ttest_g1_paired.csv", g1_paired)
save_vector("ttest_g2_paired.csv", g2_paired)

# Welch t-test (unequal variances)
welch_two <- t.test(g1, g2, var.equal = FALSE, alternative = "two.sided")
welch_less <- t.test(g1, g2, var.equal = FALSE, alternative = "less")
welch_greater <- t.test(g1, g2, var.equal = FALSE, alternative = "greater")
# t-test with mu parameter (testing against non-zero null hypothesis)
welch_mu <- t.test(g1, g2, var.equal = FALSE, alternative = "two.sided", mu = 0.5)

# Confidence intervals at different levels
welch_ci_95 <- t.test(g1, g2, var.equal = FALSE, conf.level = 0.95)
welch_ci_90 <- t.test(g1, g2, var.equal = FALSE, conf.level = 0.90)
welch_ci_99 <- t.test(g1, g2, var.equal = FALSE, conf.level = 0.99)

save_ref("ttest_welch.csv", list(
  # Two-sided
  statistic_two = welch_two$statistic,
  df_two = welch_two$parameter,
  p_value_two = welch_two$p.value,
  mean_x_two = welch_two$estimate[1],
  mean_y_two = welch_two$estimate[2],
  # Less (g1 < g2)
  statistic_less = welch_less$statistic,
  df_less = welch_less$parameter,
  p_value_less = welch_less$p.value,
  # Greater (g1 > g2)
  statistic_greater = welch_greater$statistic,
  df_greater = welch_greater$parameter,
  p_value_greater = welch_greater$p.value,
  # With mu parameter
  statistic_mu = welch_mu$statistic,
  p_value_mu = welch_mu$p.value,
  # Confidence intervals
  conf_low_95 = welch_ci_95$conf.int[1],
  conf_high_95 = welch_ci_95$conf.int[2],
  conf_low_90 = welch_ci_90$conf.int[1],
  conf_high_90 = welch_ci_90$conf.int[2],
  conf_low_99 = welch_ci_99$conf.int[1],
  conf_high_99 = welch_ci_99$conf.int[2]
))

# Student t-test (equal variances assumed)
student_two <- t.test(g1, g2, var.equal = TRUE, alternative = "two.sided")
student_less <- t.test(g1, g2, var.equal = TRUE, alternative = "less")
student_greater <- t.test(g1, g2, var.equal = TRUE, alternative = "greater")

save_ref("ttest_student.csv", list(
  statistic_two = student_two$statistic,
  df_two = student_two$parameter,
  p_value_two = student_two$p.value,
  mean_x_two = student_two$estimate[1],
  mean_y_two = student_two$estimate[2],
  statistic_less = student_less$statistic,
  df_less = student_less$parameter,
  p_value_less = student_less$p.value,
  statistic_greater = student_greater$statistic,
  df_greater = student_greater$parameter,
  p_value_greater = student_greater$p.value
))

# Paired t-test
paired_two <- t.test(g1_paired, g2_paired, paired = TRUE, alternative = "two.sided")
paired_less <- t.test(g1_paired, g2_paired, paired = TRUE, alternative = "less")
paired_greater <- t.test(g1_paired, g2_paired, paired = TRUE, alternative = "greater")

save_ref("ttest_paired.csv", list(
  statistic_two = paired_two$statistic,
  df_two = paired_two$parameter,
  p_value_two = paired_two$p.value,
  mean_diff_two = paired_two$estimate,
  statistic_less = paired_less$statistic,
  df_less = paired_less$parameter,
  p_value_less = paired_less$p.value,
  statistic_greater = paired_greater$statistic,
  df_greater = paired_greater$parameter,
  p_value_greater = paired_greater$p.value
))

# ============================================
# PHASE 2b: Robust Parametric Tests
# ============================================

cat("\n=== Phase 2b: Robust Parametric Tests ===\n")

library(WRS2)
library(car)

# Yuen's test (robust t-test using trimmed means)
# Uses same g1, g2 from above with outliers added for robustness testing
g1_outlier <- c(g1, 50)  # Add outlier to g1
g2_outlier <- c(g2, -20) # Add outlier to g2

save_vector("yuen_g1.csv", g1_outlier)
save_vector("yuen_g2.csv", g2_outlier)

# Create data frame for yuen (requires formula interface)
yuen_data <- data.frame(
  value = c(g1_outlier, g2_outlier),
  group = factor(c(rep(1, length(g1_outlier)), rep(2, length(g2_outlier))))
)

# Yuen's test with 20% trimming (default)
yuen_20 <- yuen(value ~ group, data = yuen_data, tr = 0.2)
# Yuen's test with 10% trimming
yuen_10 <- yuen(value ~ group, data = yuen_data, tr = 0.1)
# Yuen's test alternatives computed manually from t-distribution
# (WRS2::yuen doesn't correctly implement alternatives)
t_stat <- yuen_20$test
df <- yuen_20$df
p_less <- pt(t_stat, df)
p_greater <- pt(t_stat, df, lower.tail = FALSE)

save_ref("yuen.csv", list(
  # 20% trim (default)
  statistic_20 = yuen_20$test,
  df_20 = yuen_20$df,
  p_value_20 = yuen_20$p.value,
  diff_20 = yuen_20$diff,
  # 10% trim
  statistic_10 = yuen_10$test,
  df_10 = yuen_10$df,
  p_value_10 = yuen_10$p.value,
  diff_10 = yuen_10$diff,
  # alternatives (20% trim) - computed from t-distribution
  p_value_less = p_less,
  p_value_greater = p_greater
))

# Brown-Forsythe test (Levene's test with median)
# Need data in long format for leveneTest
bf_data <- data.frame(
  value = c(g1, g2),
  group = factor(c(rep("A", length(g1)), rep("B", length(g2))))
)

# Brown-Forsythe (center = median)
bf_result <- leveneTest(value ~ group, data = bf_data, center = median)

# Also test with 3 groups
g3 <- rnorm(22, mean = 6, sd = 2)
save_vector("levene_g3.csv", g3)

bf_data_3 <- data.frame(
  value = c(g1, g2, g3),
  group = factor(c(rep("A", length(g1)), rep("B", length(g2)), rep("C", length(g3))))
)
bf_result_3 <- leveneTest(value ~ group, data = bf_data_3, center = median)

save_ref("brown_forsythe.csv", list(
  # 2 groups
  statistic_2 = bf_result$`F value`[1],
  df1_2 = bf_result$Df[1],
  df2_2 = bf_result$Df[2],
  p_value_2 = bf_result$`Pr(>F)`[1],
  # 3 groups
  statistic_3 = bf_result_3$`F value`[1],
  df1_3 = bf_result_3$Df[1],
  df2_3 = bf_result_3$Df[2],
  p_value_3 = bf_result_3$`Pr(>F)`[1]
))

# ============================================
# PHASE 3: Nonparametric Tests References
# ============================================

cat("\n=== Phase 3: Nonparametric Tests ===\n")

# Ranking tests
rank_simple <- c(3, 1, 4, 1, 5, 9, 2, 6)
rank_ties <- c(1, 2, 2, 3, 3, 3, 4)  # Has ties

save_vector("rank_simple.csv", rank_simple)
save_vector("rank_ties.csv", rank_ties)

# R's rank with average tie handling
save_ref("ranks.csv", list(
  # Simple ranking (no ties in this specific data)
  rank_simple_1 = rank(rank_simple, ties.method = "average")[1],
  rank_simple_2 = rank(rank_simple, ties.method = "average")[2],
  rank_simple_3 = rank(rank_simple, ties.method = "average")[3],
  rank_simple_4 = rank(rank_simple, ties.method = "average")[4],
  rank_simple_5 = rank(rank_simple, ties.method = "average")[5],
  rank_simple_6 = rank(rank_simple, ties.method = "average")[6],
  rank_simple_7 = rank(rank_simple, ties.method = "average")[7],
  rank_simple_8 = rank(rank_simple, ties.method = "average")[8],
  # Ranking with ties
  rank_ties_1 = rank(rank_ties, ties.method = "average")[1],
  rank_ties_2 = rank(rank_ties, ties.method = "average")[2],
  rank_ties_3 = rank(rank_ties, ties.method = "average")[3],
  rank_ties_4 = rank(rank_ties, ties.method = "average")[4],
  rank_ties_5 = rank(rank_ties, ties.method = "average")[5],
  rank_ties_6 = rank(rank_ties, ties.method = "average")[6],
  rank_ties_7 = rank(rank_ties, ties.method = "average")[7]
))

# Mann-Whitney U test (Wilcoxon rank-sum)
# Using larger samples for normal approximation
mw_x <- rnorm(30, mean = 5, sd = 1)
mw_y <- rnorm(35, mean = 5.8, sd = 1.2)

save_vector("mw_x.csv", mw_x)
save_vector("mw_y.csv", mw_y)

# exact=FALSE forces normal approximation
mw_result <- wilcox.test(mw_x, mw_y, exact = FALSE, correct = FALSE)
mw_less <- wilcox.test(mw_x, mw_y, alternative = "less", exact = FALSE, correct = FALSE)
mw_greater <- wilcox.test(mw_x, mw_y, alternative = "greater", exact = FALSE, correct = FALSE)
mw_corrected <- wilcox.test(mw_x, mw_y, exact = FALSE, correct = TRUE)

# Confidence interval (requires conf.int=TRUE)
mw_ci <- wilcox.test(mw_x, mw_y, exact = FALSE, correct = FALSE, conf.int = TRUE, conf.level = 0.95)

# Test with mu parameter (null hypothesis: location shift = mu)
mw_mu <- wilcox.test(mw_x, mw_y, mu = 0.5, exact = FALSE, correct = FALSE)
mw_mu_less <- wilcox.test(mw_x, mw_y, mu = 0.5, alternative = "less", exact = FALSE, correct = FALSE)
mw_mu_greater <- wilcox.test(mw_x, mw_y, mu = 0.5, alternative = "greater", exact = FALSE, correct = FALSE)

save_ref("mann_whitney.csv", list(
  statistic = mw_result$statistic,
  p_value = mw_result$p.value,
  p_value_less = mw_less$p.value,
  p_value_greater = mw_greater$p.value,
  p_value_corrected = mw_corrected$p.value,
  estimate = mw_ci$estimate,
  conf_low = mw_ci$conf.int[1],
  conf_high = mw_ci$conf.int[2],
  p_value_mu = mw_mu$p.value,
  p_value_mu_less = mw_mu_less$p.value,
  p_value_mu_greater = mw_mu_greater$p.value
))

# Mann-Whitney with exact p-value (smaller samples, no ties)
# Use integer data to avoid ties
mw_x_exact <- c(1.1, 2.3, 3.5, 4.7, 5.9, 6.2, 7.4, 8.6)
mw_y_exact <- c(2.2, 3.4, 4.6, 5.8, 7.0, 8.2, 9.4, 10.6, 11.8)

save_vector("mw_x_exact.csv", mw_x_exact)
save_vector("mw_y_exact.csv", mw_y_exact)

mw_exact_two <- wilcox.test(mw_x_exact, mw_y_exact, exact = TRUE)
mw_exact_less <- wilcox.test(mw_x_exact, mw_y_exact, exact = TRUE, alternative = "less")
mw_exact_greater <- wilcox.test(mw_x_exact, mw_y_exact, exact = TRUE, alternative = "greater")
mw_exact_ci <- wilcox.test(mw_x_exact, mw_y_exact, exact = TRUE, conf.int = TRUE, conf.level = 0.95)
mw_exact_ci90 <- wilcox.test(mw_x_exact, mw_y_exact, exact = TRUE, conf.int = TRUE, conf.level = 0.90)

save_ref("mann_whitney_exact.csv", list(
  statistic = mw_exact_two$statistic,
  p_value = mw_exact_two$p.value,
  p_value_less = mw_exact_less$p.value,
  p_value_greater = mw_exact_greater$p.value,
  estimate = mw_exact_ci$estimate,
  conf_low_95 = mw_exact_ci$conf.int[1],
  conf_high_95 = mw_exact_ci$conf.int[2],
  conf_low_90 = mw_exact_ci90$conf.int[1],
  conf_high_90 = mw_exact_ci90$conf.int[2]
))

# Wilcoxon Signed-Rank test (paired)
wsr_x <- rnorm(25, mean = 10, sd = 2)
wsr_y <- wsr_x + rnorm(25, mean = 0.5, sd = 0.8)

save_vector("wsr_x.csv", wsr_x)
save_vector("wsr_y.csv", wsr_y)

wsr_result <- wilcox.test(wsr_x, wsr_y, paired = TRUE, exact = FALSE, correct = FALSE)
wsr_less <- wilcox.test(wsr_x, wsr_y, paired = TRUE, alternative = "less", exact = FALSE, correct = FALSE)
wsr_greater <- wilcox.test(wsr_x, wsr_y, paired = TRUE, alternative = "greater", exact = FALSE, correct = FALSE)
wsr_corrected <- wilcox.test(wsr_x, wsr_y, paired = TRUE, exact = FALSE, correct = TRUE)

# Confidence interval
wsr_ci <- wilcox.test(wsr_x, wsr_y, paired = TRUE, exact = FALSE, correct = FALSE, conf.int = TRUE, conf.level = 0.95)

# Test with mu parameter (null hypothesis: median difference = mu)
wsr_mu <- wilcox.test(wsr_x, wsr_y, paired = TRUE, mu = 0.3, exact = FALSE, correct = FALSE)
wsr_mu_less <- wilcox.test(wsr_x, wsr_y, paired = TRUE, mu = 0.3, alternative = "less", exact = FALSE, correct = FALSE)
wsr_mu_greater <- wilcox.test(wsr_x, wsr_y, paired = TRUE, mu = 0.3, alternative = "greater", exact = FALSE, correct = FALSE)

save_ref("wilcoxon_signed_rank.csv", list(
  statistic = wsr_result$statistic,
  p_value = wsr_result$p.value,
  p_value_less = wsr_less$p.value,
  p_value_greater = wsr_greater$p.value,
  p_value_corrected = wsr_corrected$p.value,
  estimate = wsr_ci$estimate,
  conf_low = wsr_ci$conf.int[1],
  conf_high = wsr_ci$conf.int[2],
  p_value_mu = wsr_mu$p.value,
  p_value_mu_less = wsr_mu_less$p.value,
  p_value_mu_greater = wsr_mu_greater$p.value
))

# Wilcoxon Signed-Rank with exact p-value (smaller samples, no ties)
# Differences should have no ties and no zeros
# Creating data with unique differences: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1
wsr_x_exact <- seq(10, 20, length.out = 10)
wsr_y_exact <- wsr_x_exact - c(0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1)

save_vector("wsr_x_exact.csv", wsr_x_exact)
save_vector("wsr_y_exact.csv", wsr_y_exact)

wsr_exact_two <- wilcox.test(wsr_x_exact, wsr_y_exact, paired = TRUE, exact = TRUE)
wsr_exact_less <- wilcox.test(wsr_x_exact, wsr_y_exact, paired = TRUE, exact = TRUE, alternative = "less")
wsr_exact_greater <- wilcox.test(wsr_x_exact, wsr_y_exact, paired = TRUE, exact = TRUE, alternative = "greater")
wsr_exact_ci <- wilcox.test(wsr_x_exact, wsr_y_exact, paired = TRUE, exact = TRUE, conf.int = TRUE, conf.level = 0.95)
wsr_exact_ci90 <- wilcox.test(wsr_x_exact, wsr_y_exact, paired = TRUE, exact = TRUE, conf.int = TRUE, conf.level = 0.90)

save_ref("wilcoxon_signed_rank_exact.csv", list(
  statistic = wsr_exact_two$statistic,
  p_value = wsr_exact_two$p.value,
  p_value_less = wsr_exact_less$p.value,
  p_value_greater = wsr_exact_greater$p.value,
  estimate = wsr_exact_ci$estimate,
  conf_low_95 = wsr_exact_ci$conf.int[1],
  conf_high_95 = wsr_exact_ci$conf.int[2],
  conf_low_90 = wsr_exact_ci90$conf.int[1],
  conf_high_90 = wsr_exact_ci90$conf.int[2]
))

# Kruskal-Wallis test
kw_a <- rnorm(20, mean = 5, sd = 1)
kw_b <- rnorm(22, mean = 6, sd = 1.2)
kw_c <- rnorm(18, mean = 5.5, sd = 0.9)

save_vector("kw_a.csv", kw_a)
save_vector("kw_b.csv", kw_b)
save_vector("kw_c.csv", kw_c)

kw_data <- data.frame(
  value = c(kw_a, kw_b, kw_c),
  group = factor(c(rep("A", length(kw_a)), rep("B", length(kw_b)), rep("C", length(kw_c))))
)

kw_result <- kruskal.test(value ~ group, data = kw_data)

save_ref("kruskal_wallis.csv", list(
  statistic = kw_result$statistic,
  df = kw_result$parameter,
  p_value = kw_result$p.value
))

# ============================================
# PHASE 4: Distributional Tests References
# ============================================

cat("\n=== Phase 4: Distributional Tests ===\n")

# Shapiro-Wilk test for normality
# Test with various sample sizes and distributions

# Small sample from normal distribution
sw_normal_small <- rnorm(20, mean = 5, sd = 2)
save_vector("sw_normal_small.csv", sw_normal_small)

sw_small_result <- shapiro.test(sw_normal_small)

# Medium sample from normal distribution
sw_normal_medium <- rnorm(50, mean = 10, sd = 3)
save_vector("sw_normal_medium.csv", sw_normal_medium)

sw_medium_result <- shapiro.test(sw_normal_medium)

# Large sample from normal distribution
sw_normal_large <- rnorm(100, mean = 0, sd = 1)
save_vector("sw_normal_large.csv", sw_normal_large)

sw_large_result <- shapiro.test(sw_normal_large)

# Non-normal data (uniform distribution - should reject normality)
sw_uniform <- runif(30, min = 0, max = 10)
save_vector("sw_uniform.csv", sw_uniform)

sw_uniform_result <- shapiro.test(sw_uniform)

# Non-normal data (exponential - should reject normality)
sw_exp <- rexp(40, rate = 1)
save_vector("sw_exp.csv", sw_exp)

sw_exp_result <- shapiro.test(sw_exp)

save_ref("shapiro_wilk.csv", list(
  # Normal small (n=20)
  w_normal_small = sw_small_result$statistic,
  p_normal_small = sw_small_result$p.value,
  # Normal medium (n=50)
  w_normal_medium = sw_medium_result$statistic,
  p_normal_medium = sw_medium_result$p.value,
  # Normal large (n=100)
  w_normal_large = sw_large_result$statistic,
  p_normal_large = sw_large_result$p.value,
  # Uniform (non-normal)
  w_uniform = sw_uniform_result$statistic,
  p_uniform = sw_uniform_result$p.value,
  # Exponential (non-normal)
  w_exp = sw_exp_result$statistic,
  p_exp = sw_exp_result$p.value
))

# ============================================
# PHASE 5: Forecast Tests References
# ============================================

cat("\n=== Phase 5: Forecast Tests ===\n")

library(forecast)

# Generate actual values and two competing forecasts
set.seed(123)
actual <- cumsum(rnorm(100))  # Random walk as actual values

# Forecast 1: Slightly lagged actual + noise (better forecast)
forecast1 <- c(0, actual[-100]) + rnorm(100, sd = 0.5)

# Forecast 2: More noise (worse forecast)
forecast2 <- c(0, actual[-100]) + rnorm(100, sd = 1.5)

# Compute forecast errors
e1 <- actual - forecast1
e2 <- actual - forecast2

save_vector("dm_e1.csv", e1)
save_vector("dm_e2.csv", e2)

# Diebold-Mariano test with squared error loss, h=1
dm_se_h1 <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 2)

# Diebold-Mariano test with absolute error loss, h=1
dm_ae_h1 <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 1)

# Diebold-Mariano test with squared error loss, h=3
dm_se_h3 <- dm.test(e1, e2, alternative = "two.sided", h = 3, power = 2)

# Diebold-Mariano test with alternative hypotheses
dm_less <- dm.test(e1, e2, alternative = "less", h = 1, power = 2)
dm_greater <- dm.test(e1, e2, alternative = "greater", h = 1, power = 2)

# Diebold-Mariano test with Bartlett variance estimator
dm_se_h1_bartlett <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 2, varestimator = "bartlett")
dm_se_h3_bartlett <- dm.test(e1, e2, alternative = "two.sided", h = 3, power = 2, varestimator = "bartlett")
dm_less_bartlett <- dm.test(e1, e2, alternative = "less", h = 1, power = 2, varestimator = "bartlett")
dm_greater_bartlett <- dm.test(e1, e2, alternative = "greater", h = 1, power = 2, varestimator = "bartlett")

save_ref("diebold_mariano.csv", list(
  # Squared error, h=1, ACF (default)
  statistic_se_h1 = dm_se_h1$statistic,
  p_value_se_h1 = dm_se_h1$p.value,
  # Absolute error, h=1, ACF
  statistic_ae_h1 = dm_ae_h1$statistic,
  p_value_ae_h1 = dm_ae_h1$p.value,
  # Squared error, h=3, ACF
  statistic_se_h3 = dm_se_h3$statistic,
  p_value_se_h3 = dm_se_h3$p.value,
  # Alternative hypotheses (ACF)
  p_value_less_se_h1 = dm_less$p.value,
  p_value_greater_se_h1 = dm_greater$p.value,
  # Bartlett variance estimator
  statistic_se_h1_bartlett = dm_se_h1_bartlett$statistic,
  p_value_se_h1_bartlett = dm_se_h1_bartlett$p.value,
  statistic_se_h3_bartlett = dm_se_h3_bartlett$statistic,
  p_value_se_h3_bartlett = dm_se_h3_bartlett$p.value,
  p_value_less_bartlett = dm_less_bartlett$p.value,
  p_value_greater_bartlett = dm_greater_bartlett$p.value
))

# ============================================
# PHASE 1b: Additional Math Primitives (std_dev, skewness, kurtosis)
# ============================================

cat("\n=== Phase 1b: Additional Math Primitives ===\n")

library(e1071)

# Standard deviation (matches R's sd())
save_ref("math_extended.csv", list(
  # Standard deviation
  sd_short = sd(x_short),
  sd_long = sd(x_long),
  sd_outlier = sd(x_outlier),

  # Skewness (Fisher's, type 2)
  skew_short = skewness(x_short, type = 2),
  skew_long = skewness(x_long, type = 2),
  skew_outlier = skewness(x_outlier, type = 2),

  # Kurtosis (excess, Fisher's, type 2)
  kurt_short = kurtosis(x_short, type = 2),
  kurt_long = kurtosis(x_long, type = 2),
  kurt_outlier = kurtosis(x_outlier, type = 2)
))

# Skewed data for testing
x_skewed <- c(1, 1, 1, 1, 1, 2, 2, 3, 5, 10, 20)
save_vector("vec_skewed.csv", x_skewed)

save_ref("math_skewed.csv", list(
  skew_skewed = skewness(x_skewed, type = 2),
  kurt_skewed = kurtosis(x_skewed, type = 2)
))

# ============================================
# PHASE 4b: Shapiro-Wilk Edge Cases
# ============================================

cat("\n=== Phase 4b: Shapiro-Wilk Edge Cases ===\n")

# Very small samples (n=3, 4, 5)
sw_n3 <- rnorm(3)
sw_n4 <- rnorm(4)
sw_n5 <- rnorm(5)
sw_n10 <- rnorm(10)

save_vector("sw_n3.csv", sw_n3)
save_vector("sw_n4.csv", sw_n4)
save_vector("sw_n5.csv", sw_n5)
save_vector("sw_n10.csv", sw_n10)

sw_n3_result <- shapiro.test(sw_n3)
sw_n4_result <- shapiro.test(sw_n4)
sw_n5_result <- shapiro.test(sw_n5)
sw_n10_result <- shapiro.test(sw_n10)

save_ref("shapiro_wilk_edge.csv", list(
  # n=3 (exact formula)
  w_n3 = sw_n3_result$statistic,
  p_n3 = sw_n3_result$p.value,
  # n=4 (small n path)
  w_n4 = sw_n4_result$statistic,
  p_n4 = sw_n4_result$p.value,
  # n=5 (small n path)
  w_n5 = sw_n5_result$statistic,
  p_n5 = sw_n5_result$p.value,
  # n=10 (4 <= n <= 11 path)
  w_n10 = sw_n10_result$statistic,
  p_n10 = sw_n10_result$p.value
))

# ============================================
# PHASE 6: D'Agostino's K-Squared Test
# ============================================

cat("\n=== Phase 6: D'Agostino's K-Squared Test ===\n")

library(moments)

# Normal-ish data (should not reject)
dag_normal <- rnorm(50, mean = 10, sd = 2)
save_vector("dag_normal.csv", dag_normal)

# Skewed data (should reject)
dag_skewed <- rexp(50, rate = 0.5)
save_vector("dag_skewed.csv", dag_skewed)

# Test results
dag_skew_test <- agostino.test(dag_normal)
dag_kurt_test <- anscombe.test(dag_normal)

dag_skew_test_skewed <- agostino.test(dag_skewed)
dag_kurt_test_skewed <- anscombe.test(dag_skewed)

save_ref("dagostino.csv", list(
  # Normal data - skewness test
  z_skew_normal = dag_skew_test$statistic[2],
  p_skew_normal = dag_skew_test$p.value,
  # Normal data - kurtosis test
  z_kurt_normal = dag_kurt_test$statistic[2],
  p_kurt_normal = dag_kurt_test$p.value,
  # Skewed data - skewness test
  z_skew_skewed = dag_skew_test_skewed$statistic[2],
  p_skew_skewed = dag_skew_test_skewed$p.value,
  # Skewed data - kurtosis test
  z_kurt_skewed = dag_kurt_test_skewed$statistic[2],
  p_kurt_skewed = dag_kurt_test_skewed$p.value
))

# ============================================
# PHASE 7: Brunner-Munzel Test
# ============================================

cat("\n=== Phase 7: Brunner-Munzel Test ===\n")

library(lawstat)

# Test data with unequal variances
bm_x <- c(1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1)
bm_y <- c(3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4)

save_vector("bm_x.csv", bm_x)
save_vector("bm_y.csv", bm_y)

bm_result <- brunner.munzel.test(bm_x, bm_y)
bm_less <- brunner.munzel.test(bm_x, bm_y, alternative = "less")
bm_greater <- brunner.munzel.test(bm_x, bm_y, alternative = "greater")

# Confidence intervals at different alpha levels
bm_ci_95 <- brunner.munzel.test(bm_x, bm_y, alpha = 0.05)
bm_ci_90 <- brunner.munzel.test(bm_x, bm_y, alpha = 0.10)
bm_ci_99 <- brunner.munzel.test(bm_x, bm_y, alpha = 0.01)

save_ref("brunner_munzel.csv", list(
  statistic = bm_result$statistic,
  df = bm_result$parameter,
  p_value = bm_result$p.value,
  estimate = bm_result$estimate,
  p_value_less = bm_less$p.value,
  p_value_greater = bm_greater$p.value,
  # Confidence intervals (alpha=0.05 -> 95% CI)
  conf_low_95 = bm_ci_95$conf.int[1],
  conf_high_95 = bm_ci_95$conf.int[2],
  conf_low_90 = bm_ci_90$conf.int[1],
  conf_high_90 = bm_ci_90$conf.int[2],
  conf_low_99 = bm_ci_99$conf.int[1],
  conf_high_99 = bm_ci_99$conf.int[2]
))

# ============================================
# PHASE 8: Clark-West Test
# ============================================

cat("\n=== Phase 8: Clark-West Test ===\n")

# Clark-West test for nested model comparison
# Uses the same forecast errors as Diebold-Mariano

# Compute Clark-West adjusted statistic manually
# d_t = e1_t^2 - e2_t^2 + (e1_t - e2_t)^2
cw_d <- e1^2 - e2^2 + (e1 - e2)^2

n <- length(cw_d)
d_bar <- mean(cw_d)

# HAC variance (h=1, so no autocorrelation adjustment)
gamma_0 <- var(cw_d) * (n - 1) / n  # Population variance
var_d_bar <- gamma_0 / n

cw_stat <- d_bar / sqrt(var_d_bar)
cw_p_one <- 1 - pnorm(cw_stat)  # One-sided
cw_p_two <- 2 * (1 - pnorm(abs(cw_stat)))  # Two-sided

# Also compute with h=3 (HAC adjustment)
# Autocovariances for h=3 (lags 0, 1, 2)
d_centered <- cw_d - mean(cw_d)
gamma_0_h3 <- sum(d_centered^2) / n
gamma_1 <- sum(d_centered[2:n] * d_centered[1:(n-1)]) / n
gamma_2 <- sum(d_centered[3:n] * d_centered[1:(n-2)]) / n

acov_sum_h3 <- gamma_0_h3 + 2 * gamma_1 + 2 * gamma_2
var_d_bar_h3 <- acov_sum_h3 / n

cw_stat_h3 <- d_bar / sqrt(var_d_bar_h3)
cw_p_one_h3 <- 1 - pnorm(cw_stat_h3)
cw_p_two_h3 <- 2 * (1 - pnorm(abs(cw_stat_h3)))

save_ref("clark_west.csv", list(
  # h=1
  statistic_h1 = cw_stat,
  p_value_one_h1 = cw_p_one,
  p_value_two_h1 = cw_p_two,
  # h=3
  statistic_h3 = cw_stat_h3,
  p_value_one_h3 = cw_p_one_h3,
  p_value_two_h3 = cw_p_two_h3
))

# ============================================
# PHASE 9: SPA Test (Superior Predictive Ability)
# ============================================

cat("\n=== Phase 9: SPA Test ===\n")

# Generate benchmark and model losses for SPA test
set.seed(456)
spa_n <- 100

# Benchmark: moderate losses
spa_benchmark <- abs(rnorm(spa_n, 2, 0.5))
save_vector("spa_benchmark.csv", spa_benchmark)

# Model 1: Clearly better (lower losses)
spa_model1 <- abs(rnorm(spa_n, 1, 0.3))
save_vector("spa_model1.csv", spa_model1)

# Model 2: Similar to benchmark
spa_model2 <- abs(rnorm(spa_n, 2.1, 0.5))
save_vector("spa_model2.csv", spa_model2)

# Model 3: Worse than benchmark
spa_model3 <- abs(rnorm(spa_n, 3, 0.6))
save_vector("spa_model3.csv", spa_model3)

# Compute SPA statistic manually (without bootstrap)
# d_k = L_benchmark - L_model_k (positive = model better)
d1 <- spa_benchmark - spa_model1
d2 <- spa_benchmark - spa_model2
d3 <- spa_benchmark - spa_model3

# Sample means
d1_bar <- mean(d1)
d2_bar <- mean(d2)
d3_bar <- mean(d3)

# Sample variances of the mean
var1 <- var(d1) / spa_n
var2 <- var(d2) / spa_n
var3 <- var(d3) / spa_n

# Standardized statistics
t1 <- d1_bar * sqrt(spa_n) / sqrt(var1)
t2 <- d2_bar * sqrt(spa_n) / sqrt(var2)
t3 <- d3_bar * sqrt(spa_n) / sqrt(var3)

# SPA statistic = max of standardized stats
spa_stat <- max(t1, t2, t3)
best_idx <- which.max(c(t1, t2, t3)) - 1  # 0-indexed

save_ref("spa.csv", list(
  statistic = spa_stat,
  t1 = t1,
  t2 = t2,
  t3 = t3,
  best_idx = best_idx,
  d1_bar = d1_bar,
  d2_bar = d2_bar,
  d3_bar = d3_bar
))

# ============================================
# PHASE 10: Model Confidence Set (MCS)
# ============================================

cat("\n=== Phase 10: Model Confidence Set ===\n")

# Generate loss series for MCS test
set.seed(789)
mcs_n <- 200

# Model 0: Best (lowest losses)
mcs_model0 <- abs(rnorm(mcs_n, 1.0, 0.3))
save_vector("mcs_model0.csv", mcs_model0)

# Model 1: Good (similar to model 0)
mcs_model1 <- abs(rnorm(mcs_n, 1.1, 0.35))
save_vector("mcs_model1.csv", mcs_model1)

# Model 2: Medium (clearly worse than 0,1)
mcs_model2 <- abs(rnorm(mcs_n, 2.0, 0.5))
save_vector("mcs_model2.csv", mcs_model2)

# Model 3: Worst (much worse)
mcs_model3 <- abs(rnorm(mcs_n, 3.5, 0.8))
save_vector("mcs_model3.csv", mcs_model3)

# Compute pairwise loss differentials for MCS validation
# d_ij = L_i - L_j (positive means model i has higher loss = worse)
d_01 <- mcs_model0 - mcs_model1
d_02 <- mcs_model0 - mcs_model2
d_03 <- mcs_model0 - mcs_model3
d_12 <- mcs_model1 - mcs_model2
d_13 <- mcs_model1 - mcs_model3
d_23 <- mcs_model2 - mcs_model3

# Mean loss differentials
mean_d_01 <- mean(d_01)
mean_d_02 <- mean(d_02)
mean_d_03 <- mean(d_03)
mean_d_12 <- mean(d_12)
mean_d_13 <- mean(d_13)
mean_d_23 <- mean(d_23)

# t-statistics for each pair (two-sample t-test on paired differences)
t_01 <- mean(d_01) / sqrt(var(d_01) / mcs_n)
t_02 <- mean(d_02) / sqrt(var(d_02) / mcs_n)
t_03 <- mean(d_03) / sqrt(var(d_03) / mcs_n)
t_12 <- mean(d_12) / sqrt(var(d_12) / mcs_n)
t_13 <- mean(d_13) / sqrt(var(d_13) / mcs_n)
t_23 <- mean(d_23) / sqrt(var(d_23) / mcs_n)

# T_R statistic = max |t_ij|
t_range <- max(abs(c(t_01, t_02, t_03, t_12, t_13, t_23)))

save_ref("mcs.csv", list(
  t_01 = t_01,
  t_02 = t_02,
  t_03 = t_03,
  t_12 = t_12,
  t_13 = t_13,
  t_23 = t_23,
  t_range = t_range,
  mean_d_01 = mean_d_01,
  mean_d_02 = mean_d_02,
  mean_d_03 = mean_d_03
))

# ============================================
# PHASE 11: Energy Distance Test
# ============================================

cat("\n=== Phase 11: Energy Distance Test ===\n")

# Try to load energy package
energy_available <- require(energy, quietly = TRUE)

if (energy_available) {
  set.seed(321)

  # Generate two clearly different samples
  ed_x <- rnorm(30, mean = 0, sd = 1)
  ed_y <- rnorm(30, mean = 3, sd = 1)

  save_vector("ed_x.csv", ed_x)
  save_vector("ed_y.csv", ed_y)

  # Energy distance test
  # Note: eqdist.etest returns different things
  # We'll compute the energy distance statistic manually

  # Energy statistic (E = 2*E|X-Y| - E|X-X'| - E|Y-Y'|)
  n_x <- length(ed_x)
  n_y <- length(ed_y)

  # Mean distance X-Y
  mean_xy <- 0
  for (i in 1:n_x) {
    for (j in 1:n_y) {
      mean_xy <- mean_xy + abs(ed_x[i] - ed_y[j])
    }
  }
  mean_xy <- mean_xy / (n_x * n_y)

  # Mean distance X-X'
  mean_xx <- 0
  for (i in 1:n_x) {
    for (j in 1:n_x) {
      if (i != j) {
        mean_xx <- mean_xx + abs(ed_x[i] - ed_x[j])
      }
    }
  }
  mean_xx <- mean_xx / (n_x * (n_x - 1))

  # Mean distance Y-Y'
  mean_yy <- 0
  for (i in 1:n_y) {
    for (j in 1:n_y) {
      if (i != j) {
        mean_yy <- mean_yy + abs(ed_y[i] - ed_y[j])
      }
    }
  }
  mean_yy <- mean_yy / (n_y * (n_y - 1))

  # Energy statistic
  ed_stat <- 2 * mean_xy - mean_xx - mean_yy

  save_ref("energy_distance.csv", list(
    statistic = ed_stat,
    mean_xy = mean_xy,
    mean_xx = mean_xx,
    mean_yy = mean_yy
  ))

  cat("Energy distance reference generated\n")
} else {
  cat("Note: 'energy' package not available, skipping energy distance reference\n")

  # Generate basic reference without the package
  set.seed(321)
  ed_x <- rnorm(30, mean = 0, sd = 1)
  ed_y <- rnorm(30, mean = 3, sd = 1)

  save_vector("ed_x.csv", ed_x)
  save_vector("ed_y.csv", ed_y)

  # Manual energy statistic calculation
  n_x <- length(ed_x)
  n_y <- length(ed_y)

  mean_xy <- mean(outer(ed_x, ed_y, function(a, b) abs(a - b)))
  mean_xx <- mean(as.dist(outer(ed_x, ed_x, function(a, b) abs(a - b))))
  mean_yy <- mean(as.dist(outer(ed_y, ed_y, function(a, b) abs(a - b))))

  ed_stat <- 2 * mean_xy - mean_xx - mean_yy

  save_ref("energy_distance.csv", list(
    statistic = ed_stat,
    mean_xy = mean_xy,
    mean_xx = mean_xx,
    mean_yy = mean_yy
  ))
}

# ============================================
# PHASE 12: Permutation T-Test
# ============================================

cat("\n=== Phase 12: Permutation T-Test ===\n")

set.seed(555)

# Generate two samples with different means
perm_x <- rnorm(20, mean = 5, sd = 1)
perm_y <- rnorm(25, mean = 7, sd = 1.2)

save_vector("perm_x.csv", perm_x)
save_vector("perm_y.csv", perm_y)

# Compute t-statistic manually (for comparison)
# t = (mean_x - mean_y) / sqrt(var_x/n_x + var_y/n_y)
mean_x <- mean(perm_x)
mean_y <- mean(perm_y)
var_x <- var(perm_x)
var_y <- var(perm_y)
n_x <- length(perm_x)
n_y <- length(perm_y)

t_stat <- (mean_x - mean_y) / sqrt(var_x / n_x + var_y / n_y)

save_ref("permutation_t.csv", list(
  t_statistic = t_stat,
  mean_x = mean_x,
  mean_y = mean_y,
  var_x = var_x,
  var_y = var_y
))

cat("\n=== Reference generation complete ===\n")
cat("Run 'cargo test' to verify Rust implementation against these references.\n")
