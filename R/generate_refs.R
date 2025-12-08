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
  p_value_greater = welch_greater$p.value
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
  diff_10 = yuen_10$diff
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

save_ref("mann_whitney.csv", list(
  statistic = mw_result$statistic,
  p_value = mw_result$p.value
))

# Wilcoxon Signed-Rank test (paired)
wsr_x <- rnorm(25, mean = 10, sd = 2)
wsr_y <- wsr_x + rnorm(25, mean = 0.5, sd = 0.8)

save_vector("wsr_x.csv", wsr_x)
save_vector("wsr_y.csv", wsr_y)

wsr_result <- wilcox.test(wsr_x, wsr_y, paired = TRUE, exact = FALSE, correct = FALSE)

save_ref("wilcoxon_signed_rank.csv", list(
  statistic = wsr_result$statistic,
  p_value = wsr_result$p.value
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

save_ref("diebold_mariano.csv", list(
  # Squared error, h=1
  statistic_se_h1 = dm_se_h1$statistic,
  p_value_se_h1 = dm_se_h1$p.value,
  # Absolute error, h=1
  statistic_ae_h1 = dm_ae_h1$statistic,
  p_value_ae_h1 = dm_ae_h1$p.value,
  # Squared error, h=3
  statistic_se_h3 = dm_se_h3$statistic,
  p_value_se_h3 = dm_se_h3$p.value
))

cat("\n=== Reference generation complete ===\n")
cat("Run 'cargo test' to verify Rust implementation against these references.\n")
