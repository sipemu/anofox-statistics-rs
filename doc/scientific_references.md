# Scientific References

This document provides the scientific references for all statistical tests implemented in this library.

## Parametric Tests

### Student's t-Test

- **Student (1908).** "The Probable Error of a Mean." *Biometrika*, 6(1), 1–25. [DOI: 10.2307/2331554](https://doi.org/10.2307/2331554)

  *Note: "Student" was the pseudonym of William Sealy Gosset, who worked at the Guinness Brewery.*

### Welch's t-Test

- **Welch, B. L. (1947).** "The Generalization of 'Student's' Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1–2), 28–35. [DOI: 10.2307/2332510](https://doi.org/10.2307/2332510)

### Yuen's Test (Trimmed Means t-Test)

- **Yuen, K. K. (1974).** "The Two-Sample Trimmed t for Unequal Population Variances." *Biometrika*, 61(1), 165–170. [DOI: 10.2307/2334299](https://doi.org/10.2307/2334299)

### Brown-Forsythe Test (Homogeneity of Variances)

- **Brown, M. B., & Forsythe, A. B. (1974).** "Robust Tests for the Equality of Variances." *Journal of the American Statistical Association*, 69(346), 364–367. [DOI: 10.2307/2285659](https://doi.org/10.2307/2285659)

---

## Nonparametric Tests

### Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

- **Wilcoxon, F. (1945).** "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6), 80–83. [DOI: 10.2307/3001968](https://doi.org/10.2307/3001968)

- **Mann, H. B., & Whitney, D. R. (1947).** "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics*, 18(1), 50–60. [DOI: 10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491)

### Wilcoxon Signed-Rank Test

- **Wilcoxon, F. (1945).** "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6), 80–83. [DOI: 10.2307/3001968](https://doi.org/10.2307/3001968)

### Kruskal-Wallis Test

- **Kruskal, W. H., & Wallis, W. A. (1952).** "Use of Ranks in One-Criterion Variance Analysis." *Journal of the American Statistical Association*, 47(260), 583–621. [DOI: 10.2307/2280779](https://doi.org/10.2307/2280779) (Errata: ibid. 48, 907–911.)

### Brunner-Munzel Test (Stochastic Equality)

- **Brunner, E., & Munzel, U. (2000).** "The Nonparametric Behrens-Fisher Problem: Asymptotic Theory and a Small-Sample Approximation." *Biometrical Journal*, 42(1), 17–25. [DOI: 10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U](https://doi.org/10.1002/(SICI)1521-4036(200001)42:1<17::AID-BIMJ17>3.0.CO;2-U)

- **Neubert, K., & Brunner, E. (2007).** "A Studentized Permutation Test for the Non-parametric Behrens-Fisher Problem." *Computational Statistics & Data Analysis*, 51(10), 5192–5204. [DOI: 10.1016/j.csda.2006.05.024](https://doi.org/10.1016/j.csda.2006.05.024)

  *Note: The 2007 paper contains corrections to a typo in the original 2000 paper.*

---

## Distributional Tests (Normality)

### Shapiro-Wilk Test

- **Shapiro, S. S., & Wilk, M. B. (1965).** "An Analysis of Variance Test for Normality (Complete Samples)." *Biometrika*, 52(3–4), 591–611. [DOI: 10.1093/biomet/52.3-4.591](https://doi.org/10.1093/biomet/52.3-4.591)

- **Royston, J. P. (1995).** "Remark AS R94: A Remark on Algorithm AS 181: The W-test for Normality." *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, 44(4), 547–551. [DOI: 10.2307/2986146](https://doi.org/10.2307/2986146)

  *Note: The implementation uses Royston's Algorithm AS R94.*

### D'Agostino's K-Squared Test

- **D'Agostino, R. B. (1971).** "An Omnibus Test of Normality for Moderate and Large Sample Size." *Biometrika*, 58(2), 341–348. [DOI: 10.2307/2334522](https://doi.org/10.2307/2334522)

- **D'Agostino, R. B., & Pearson, E. S. (1973).** "Tests for Departure from Normality. Empirical Results for the Distributions of b₂ and √b₁." *Biometrika*, 60(3), 613–622. [DOI: 10.2307/2335012](https://doi.org/10.2307/2335012)

- **D'Agostino, R. B., Belanger, A., & D'Agostino, R. B. Jr. (1990).** "A Suggestion for Using Powerful and Informative Tests of Normality." *The American Statistician*, 44(4), 316–321. [DOI: 10.2307/2684359](https://doi.org/10.2307/2684359)

---

## Resampling Methods

### Permutation Test

- **Fisher, R. A. (1935).** *The Design of Experiments.* Oliver and Boyd. [Wikipedia](https://en.wikipedia.org/wiki/The_Design_of_Experiments)

- **Pitman, E. J. G. (1937).** "Significance Tests Which May be Applied to Samples from Any Populations." *Supplement to the Journal of the Royal Statistical Society*, 4(1), 119–130. [DOI: 10.2307/2984124](https://doi.org/10.2307/2984124)

### Stationary Bootstrap

- **Politis, D. N., & Romano, J. P. (1994).** "The Stationary Bootstrap." *Journal of the American Statistical Association*, 89(428), 1303–1313. [DOI: 10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)

### Circular Block Bootstrap

- **Künsch, H. R. (1989).** "The Jackknife and the Bootstrap for General Stationary Observations." *Annals of Statistics*, 17(3), 1217–1241. [DOI: 10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)

- **Politis, D. N., & Romano, J. P. (1992).** "A Circular Block-Resampling Procedure for Stationary Data." In R. LePage & L. Billard (Eds.), *Exploring the Limits of Bootstrap* (pp. 263–270). Wiley.

---

## Modern Distribution Tests

### Energy Distance Test

- **Székely, G. J., & Rizzo, M. L. (2004).** "Testing for Equal Distributions in High Dimension." *InterStat*, November (5). [PDF](https://pages.stat.wisc.edu/~wahba/stat860public/pdf4/Energy/JSPI5102.pdf)

- **Székely, G. J., & Rizzo, M. L. (2013).** "Energy Statistics: A Class of Statistics Based on Distances." *Journal of Statistical Planning and Inference*, 143(8), 1249–1272. [DOI: 10.1016/j.jspi.2013.03.018](https://doi.org/10.1016/j.jspi.2013.03.018)

### Maximum Mean Discrepancy (MMD) Test

- **Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).** "A Kernel Two-Sample Test." *Journal of Machine Learning Research*, 13, 723–773. [PDF](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)

---

## Forecast Evaluation Tests

### Diebold-Mariano Test

- **Diebold, F. X., & Mariano, R. S. (1995).** "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253–263. [DOI: 10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)

### Clark-West Test (Nested Model Comparison)

- **Clark, T. E., & West, K. D. (2006).** "Using Out-of-Sample Mean Squared Prediction Errors to Test the Martingale Difference Hypothesis." *Journal of Econometrics*, 135(1–2), 155–186. [DOI: 10.1016/j.jeconom.2005.07.014](https://doi.org/10.1016/j.jeconom.2005.07.014)

- **Clark, T. E., & West, K. D. (2007).** "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *Journal of Econometrics*, 138(1), 291–311. [DOI: 10.1016/j.jeconom.2006.05.023](https://doi.org/10.1016/j.jeconom.2006.05.023)

### Superior Predictive Ability (SPA) Test

- **Hansen, P. R. (2005).** "A Test for Superior Predictive Ability." *Journal of Business & Economic Statistics*, 23(4), 365–380. [DOI: 10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063)

### Model Confidence Set (MCS)

- **Hansen, P. R., Lunde, A., & Nason, J. M. (2011).** "The Model Confidence Set." *Econometrica*, 79(2), 453–497. [DOI: 10.3982/ECTA5771](https://doi.org/10.3982/ECTA5771)

---

## R Implementation References

The implementations in this library have been validated against the following R packages:

| Test | R Package | R Function | Documentation |
|------|-----------|------------|---------------|
| t-Tests | stats | `t.test()` | [R Documentation](https://rdrr.io/r/stats/t.test.html) |
| Yuen's Test | WRS2, DescTools | `yuen()`, `YuenTTest()` | [DescTools](https://andrisignorell.github.io/DescTools/reference/YuenTTest.html) |
| Brown-Forsythe | car | `leveneTest()` | [car Package](https://rdrr.io/cran/car/man/leveneTest.html) |
| Mann-Whitney U | stats | `wilcox.test()` | [R Documentation](https://rdrr.io/r/stats/wilcox.test.html) |
| Wilcoxon Signed-Rank | stats | `wilcox.test(..., paired=TRUE)` | [R Documentation](https://rdrr.io/r/stats/wilcox.test.html) |
| Kruskal-Wallis | stats | `kruskal.test()` | [R Documentation](https://rdrr.io/r/stats/kruskal.test.html) |
| Brunner-Munzel | lawstat, brunnermunzel | `brunner.munzel.test()` | [lawstat](https://rdrr.io/cran/lawstat/man/brunner.munzel.test.html) |
| Shapiro-Wilk | stats | `shapiro.test()` | [R Documentation](https://rdrr.io/r/stats/shapiro.test.html) |
| D'Agostino | moments, fBasics | `agostino.test()` | [moments](https://rdrr.io/cran/moments/man/agostino.test.html) |
| Energy Distance | energy | `eqdist.etest()` | [energy Package](https://cran.r-project.org/web/packages/energy/energy.pdf) |
| MMD | kernlab | `kmmd()` | [kernlab](https://rdrr.io/cran/kernlab/man/kmmd.html) |
| Diebold-Mariano | forecast | `dm.test()` | [forecast Package](https://pkg.robjhyndman.com/forecast/reference/dm.test.html) |
| Model Confidence Set | MCS | `MCSprocedure()` | [MCS Package](https://cran.r-project.org/web/packages/MCS/MCS.pdf) |
