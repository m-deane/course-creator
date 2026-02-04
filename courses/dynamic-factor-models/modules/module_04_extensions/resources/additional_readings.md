# Additional Readings: Module 04 - Advanced Extensions

## Time-Varying Parameters

### Essential

1. **Primiceri, G. E. (2005)**
   "Time Varying Structural Vector Autoregressions and Monetary Policy"
   *Review of Economic Studies, 72*(3), 821-852.

   **Why read:** Foundational TVP-VAR paper, directly extends to DFM context
   **Key contribution:** Bayesian estimation with stochastic volatility
   **Link:** https://doi.org/10.1111/j.1467-937X.2005.00353.x

2. **Del Negro, M., & Otrok, C. (2008)**
   "Dynamic Factor Models with Time-Varying Parameters"
   *Federal Reserve Bank of New York Staff Report 326*

   **Why read:** First comprehensive TVP-DFM treatment
   **Application:** International business cycle synchronization changes
   **Link:** https://www.newyorkfed.org/research/staff_reports/sr326

3. **Koop, G., & Korobilis, D. (2013)**
   "Large Time-Varying Parameter VARs"
   *Journal of Econometrics, 177*(2), 185-198.

   **Why read:** Computational methods for high-dimensional TVP models
   **Key technique:** Forgetting factors and limited information methods
   **Link:** https://doi.org/10.1016/j.jeconom.2013.04.007

### Recommended

4. **Stock, J. H., & Watson, M. W. (1996)**
   "Evidence on Structural Instability in Macroeconomic Time Series Relations"
   *Journal of Business & Economic Statistics, 14*(1), 11-30.

   **Why read:** Empirical evidence for time-varying parameters in macro
   **Finding:** Most relationships unstable over post-WWII period

5. **Cogley, T., & Sargent, T. J. (2005)**
   "Drifts and Volatilities: Monetary Policies and Outcomes in the Post WWII US"
   *Review of Economic Dynamics, 8*(2), 262-302.

   **Why read:** Joint modeling of parameter drift and volatility changes
   **Method:** Bayesian TVP-VAR with stochastic volatility

## Mixed-Frequency Models

### Essential

6. **Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004)**
   "The MIDAS Touch: Mixed Data Sampling Regression Models"
   *UCLA/UNC Working Paper*

   **Why read:** Original MIDAS paper
   **Key innovation:** Parsimonious distributed lag parameterization
   **Link:** Available on SSRN

7. **Mariano, R. S., & Murasawa, Y. (2003)**
   "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series"
   *Journal of Applied Econometrics, 18*(4), 427-443.

   **Why read:** State-space mixed-frequency DFM
   **Application:** Monthly-quarterly business cycle indicators
   **Link:** https://doi.org/10.1002/jae.695

8. **Schorfheide, F., & Song, D. (2015)**
   "Real-Time Forecasting with a Mixed-Frequency VAR"
   *Journal of Business & Economic Statistics, 33*(3), 366-380.

   **Why read:** Practical implementation for forecasting
   **Code:** Replication files available
   **Link:** https://doi.org/10.1080/07350015.2014.954707

### Recommended

9. **Andreou, E., Ghysels, E., & Kourtellos, A. (2013)**
   "Should Macroeconomic Forecasters Use Daily Financial Data and How?"
   *Journal of Business & Economic Statistics, 31*(2), 240-251.

   **Why read:** Extreme mixed-frequency (daily → quarterly)
   **Finding:** Daily financial data improves macro forecasts

10. **Foroni, C., & Marcellino, M. (2014)**
    "A Comparison of Mixed Frequency Approaches for Nowcasting Euro Area Macroeconomic Aggregates"
    *International Journal of Forecasting, 30*(3), 554-568.

    **Why read:** Comprehensive comparison of MIDAS vs state-space
    **Result:** State-space slightly better for nowcasting

## Large Datasets

### Essential

11. **Doz, C., Giannone, D., & Reichlin, L. (2011)**
    "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering"
    *Journal of Econometrics, 164*(1), 188-205.

    **Why read:** **THE** paper on two-step estimation
    **Proof:** Asymptotic equivalence to full MLE
    **Link:** https://doi.org/10.1016/j.jeconom.2011.02.012

12. **Bai, J., & Ng, S. (2002)**
    "Determining the Number of Factors in Approximate Factor Models"
    *Econometrica, 70*(1), 191-221.

    **Why read:** Information criteria for factor number selection
    **Contribution:** Consistent estimators for r (number of factors)
    **Link:** https://doi.org/10.1111/1468-0262.00273

13. **Bai, J., & Ng, S. (2008)**
    "Large Dimensional Factor Analysis"
    *Foundations and Trends in Econometrics, 3*(2), 89-163.

    **Why read:** Comprehensive monograph on large-dimensional DFM
    **Coverage:** Theory, estimation, inference, applications
    **Link:** https://doi.org/10.1561/0800000002

### Recommended

14. **Stock, J. H., & Watson, M. W. (2002)**
    "Forecasting Using Principal Components from a Large Number of Predictors"
    *Journal of the American Statistical Association, 97*(460), 1167-1179.

    **Why read:** Empirical evidence that PCA factors forecast well
    **Dataset:** 215 macroeconomic and financial series

15. **Boivin, J., & Ng, S. (2006)**
    "Are More Data Always Better for Factor Analysis?"
    *Journal of Econometrics, 132*(1), 169-194.

    **Why read:** Not all variables are informative!
    **Finding:** Targeted predictor selection can improve performance

### Sparse Methods

16. **Bai, J., & Ng, S. (2008)**
    "Forecasting Economic Time Series Using Targeted Predictors"
    *Journal of Econometrics, 146*(2), 304-317.

    **Why read:** Pre-screening variables before DFM estimation
    **Method:** Select predictors correlated with target
    **Link:** https://doi.org/10.1016/j.jeconom.2008.08.010

17. **Kristensen, J. T. (2017)**
    "Diffusion Indexes with Sparse Loadings"
    *Journal of Business & Economic Statistics, 35*(3), 435-451.

    **Why read:** LASSO-penalized factor loadings
    **Advantage:** Interpretable factors (few variables per factor)

## Software & Code

### R Packages

18. **dfms: Dynamic Factor Models**
    CRAN: https://cran.r-project.org/package=dfms

    **Features:** TVP-DFM, mixed-frequency, Bayesian estimation
    **Maintainer:** Active development

19. **midasr: Mixed Data Sampling Regression**
    CRAN: https://cran.r-project.org/package=midasr

    **Features:** All MIDAS variants, lag selection, forecasting
    **Documentation:** Extensive vignettes

### Python Resources

20. **statsmodels DynamicFactor Documentation**
    URL: https://www.statsmodels.org/stable/statespace.html#dynamic-factor-models

    **Examples:** Handles missing data, constraints, large N

21. **scikit-learn PCA Documentation**
    URL: https://scikit-learn.org/stable/modules/decomposition.html#pca

    **Fast:** Randomized SVD for very large N

### Replication Code

22. **NY Fed DSGE Model**
    GitHub: https://github.com/FRBNY-DSGE

    **Language:** Julia (concepts portable to Python)
    **Includes:** Time-varying parameters, mixed-frequency

23. **Giannone et al. (2008) Replication**
    URL: http://www.lucrezia-reichlin.eu/research.php

    **Code:** MATLAB (adaptable)
    **Data:** FRED-MD subset

## Datasets

### High-Dimensional

24. **FRED-MD: Monthly Database for Macroeconomic Research**
    URL: https://research.stlouisfed.org/econ/mccracken/fred-databases/

    **Variables:** 127 monthly US macroeconomic series
    **Updates:** Monthly, free download
    **Format:** CSV with transformation codes

25. **FRED-QD: Quarterly Database**
    URL: Same as FRED-MD

    **Variables:** 248 quarterly US series
    **Use:** Mixed-frequency applications

26. **ECB Statistical Data Warehouse**
    URL: https://sdw.ecb.europa.eu/

    **Coverage:** Euro area monthly/quarterly data
    **API:** Programmable access

### Real-Time Vintages

27. **Philadelphia Fed Real-Time Data Set**
    URL: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists

    **What:** Vintages of key macro variables
    **Why:** Test TVP models on actual data evolution

## Advanced Topics

### Bayesian TVP-DFM

28. **Chan, J. C., Koop, G., Poirier, D. J., & Tobias, J. L. (2019)**
    *Bayesian Econometric Methods* 2nd Edition. Cambridge University Press.
    **Chapter 14:** Time-varying parameter models

29. **Kalli, M., & Griffin, J. E. (2014)**
    "Time-Varying Sparsity in Dynamic Regression Models"
    *Journal of Econometrics, 178*, 779-793.

    **Innovation:** Allow sparsity pattern to change over time

### Machine Learning Integration

30. **Coulombe, P. G., Leroux, M., Stevanovic, D., & Surprenant, S. (2021)**
    "How is Machine Learning Useful for Macroeconomic Forecasting?"
    *Journal of Applied Econometrics*

    **Topics:** Neural networks for factor extraction, ensemble methods

## Study Plan Recommendations

### Week 1: Time-Varying Parameters
- Read Primiceri (2005), Del Negro & Otrok (2008)
- Implement rolling window estimation
- Test on FRED-MD data with COVID break

### Week 2: Mixed-Frequency
- Read Ghysels et al. (2004), Mariano & Murasawa (2003)
- Implement MIDAS and state-space approaches
- Compare nowcast accuracy

### Week 3: Large Datasets
- Read Doz et al. (2011), Bai & Ng (2002, 2008)
- Download FRED-MD (127 series)
- Two-step estimation and factor interpretation

### Week 4: Integration
- Combine TVP + mixed-frequency + large N
- Build production nowcasting system
- Backtest on historical data

---

*Last updated: 2024*
