# Additional Readings: Module 03 - Applications

## Essential Papers (Must Read)

### Nowcasting Foundations

1. **Giannone, D., Reichlin, L., & Small, D. (2008)**
   "Nowcasting: The Real-Time Informational Content of Macroeconomic Data"
   *Journal of Monetary Economics, 55*(4), 665-676.

   **Why read:** Foundational paper introducing DFM-based nowcasting with ragged edges.
   **Key contribution:** Shows how to optimally combine timely but noisy indicators.
   **Link:** https://doi.org/10.1016/j.jmoneco.2008.05.010

2. **Bańbura, M., & Modugno, M. (2014)**
   "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data"
   *Journal of Applied Econometrics, 29*(1), 133-160.

   **Why read:** State-of-the-art treatment of missing data in DFMs.
   **Key contribution:** EM algorithm for ragged edges and irregular patterns.
   **Link:** https://doi.org/10.1002/jae.2306

3. **Diebold, F. X., & Mariano, R. S. (1995)**
   "Comparing Predictive Accuracy"
   *Journal of Business & Economic Statistics, 13*(3), 253-263.

   **Why read:** The standard method for forecast comparison.
   **Key contribution:** Asymptotically valid test for equal forecast accuracy.
   **Link:** https://doi.org/10.1080/07350015.1995.10524599

## Recommended Papers (High Value)

### Nowcasting Applications

4. **Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., & Tambalotti, A. (2018)**
   "Macroeconomic Nowcasting and Forecasting with Big Data"
   *Annual Review of Economics, 10*, 615-643.

   **Why read:** Comprehensive review of modern nowcasting methods.
   **Covers:** NY Fed Nowcast methodology, big data integration, real-time applications.
   **Link:** https://doi.org/10.1146/annurev-economics-080217-053214

5. **Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013)**
   "Now-Casting and the Real-Time Data Flow"
   *Handbook of Economic Forecasting, Vol 2*, 195-237.

   **Why read:** Authoritative handbook chapter on real-time nowcasting.
   **Covers:** Publication lags, revision analysis, operational systems.
   **Link:** https://doi.org/10.1016/B978-0-444-53683-9.00004-9

### Forecast Evaluation

6. **Gneiting, T., & Raftery, A. E. (2007)**
   "Strictly Proper Scoring Rules, Prediction, and Estimation"
   *Journal of the American Statistical Association, 102*(477), 359-378.

   **Why read:** Definitive treatment of proper scoring rules (CRPS, log score).
   **Key insight:** Why and when to use distributional forecasts.
   **Link:** https://doi.org/10.1198/016214506000001437

7. **Clark, T. E., & McCracken, M. W. (2013)**
   "Advances in Forecast Evaluation"
   *Handbook of Economic Forecasting, Vol 2*, 1107-1201.

   **Why read:** Comprehensive review of modern forecast evaluation.
   **Covers:** Out-of-sample tests, conditional evaluation, real-time issues.
   **Link:** https://doi.org/10.1016/B978-0-444-62731-5.00020-8

8. **Harvey, D., Leybourne, S., & Newbold, P. (1997)**
   "Testing the Equality of Prediction Mean Squared Errors"
   *International Journal of Forecasting, 13*(2), 281-291.

   **Why read:** Small-sample corrections for Diebold-Mariano test.
   **Practical:** Use these corrections in finite samples (T < 100).
   **Link:** https://doi.org/10.1016/S0169-2070(96)00719-4

## Advanced Topics (For Deeper Exploration)

### Missing Data Theory

9. **Durbin, J., & Koopman, S. J. (2012)**
   *Time Series Analysis by State Space Methods.* 2nd Edition. Oxford University Press.
   **Chapter 4: Missing Observations**

   **Why read:** Rigorous treatment of Kalman filtering with missing data.
   **Math level:** Advanced (matrix calculus, state-space theory).

10. **Doz, C., Giannone, D., & Reichlin, L. (2011)**
    "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering"
    *Journal of Econometrics, 164*(1), 188-205.

    **Why read:** Practical two-step approach (PCA + Kalman smoother).
    **Computational:** Much faster than full MLE for large N.
    **Link:** https://doi.org/10.1016/j.jeconom.2011.02.012

### Forecast Combination

11. **Timmermann, A. (2006)**
    "Forecast Combinations"
    *Handbook of Economic Forecasting, Vol 1*, 135-196.

    **Why read:** Theory and practice of combining forecasts.
    **Covers:** Optimal weights, simple averaging, Bayesian approaches.
    **Link:** https://doi.org/10.1016/S1574-0706(05)01004-9

12. **Bates, J. M., & Granger, C. W. J. (1969)**
    "The Combination of Forecasts"
    *Operations Research Quarterly, 20*(4), 451-468.

    **Why read:** Classic paper showing simple averaging often optimal.
    **Historical:** Still highly relevant 50+ years later.
    **Link:** https://doi.org/10.1057/jors.1969.103

### Real-Time Data Issues

13. **Croushore, D., & Stark, T. (2001)**
    "A Real-Time Data Set for Macroeconomists"
    *Journal of Econometrics, 105*(1), 111-130.

    **Why read:** Introduction to real-time vintage data (Philadelphia Fed).
    **Data source:** ALFRED database origins.
    **Link:** https://doi.org/10.1016/S0304-4076(01)00072-0

14. **Aruoba, S. B., Diebold, F. X., & Scotti, C. (2009)**
    "Real-Time Measurement of Business Conditions"
    *Journal of Business & Economic Statistics, 27*(4), 417-427.

    **Why read:** High-frequency nowcasting with extreme publication lags.
    **Application:** Daily business conditions index.
    **Link:** https://doi.org/10.1198/jbes.2009.07205

## Software & Code Resources

### Python Implementations

15. **statsmodels.tsa.statespace.DynamicFactor**
    Documentation: https://www.statsmodels.org/stable/statespace.html

    **What:** Python implementation of DFM with Kalman filter.
    **Features:** Handles missing data automatically, supports constraints.

16. **NY Fed DSGE Model GitHub**
    Repository: https://github.com/FRBNY-DSGE/DSGE.jl

    **What:** Production nowcasting code from NY Fed (Julia, but concepts portable).
    **Learn:** Real-world implementation patterns.

### R Implementations

17. **nowcasting R package**
    CRAN: https://cran.r-project.org/package=nowcasting

    **What:** Dedicated package for DFM-based nowcasting.
    **Features:** Ragged edge handling, real-time evaluation.

### Datasets

18. **FRED-MD Database**
    Source: https://research.stlouisfed.org/econ/mccracken/fred-databases/

    **What:** 127 monthly US macroeconomic indicators (standardized).
    **Updates:** Monthly, free download.

19. **ALFRED (Archival FRED)**
    Source: https://alfred.stlouisfed.org/

    **What:** Real-time vintages of macroeconomic data.
    **Critical:** For proper nowcast evaluation with publication lags.

20. **Survey of Professional Forecasters**
    Source: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters

    **What:** Consensus forecasts from professional economists.
    **Use:** Benchmark for forecast evaluation.

## Operational Nowcasting Systems (Learn from Practitioners)

21. **Federal Reserve Bank of New York Nowcast**
    URL: https://www.newyorkfed.org/research/policy/nowcast

    **What:** Operational GDP nowcast, updated weekly.
    **Code:** GitHub repository with full methodology.

22. **Federal Reserve Bank of Atlanta GDPNow**
    URL: https://www.atlantafed.org/cqer/research/gdpnow

    **What:** Alternative nowcasting approach (bottom-up from national accounts).
    **Compare:** Different methodology, often diverges from NY Fed.

23. **European Central Bank Nowcasting Resources**
    URL: https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/experimental-statistics.en.html

    **What:** Euro area nowcasting research and tools.
    **Coverage:** Multi-country applications.

## Blogs & Tutorials

24. **Macroeconomic Musings (by David Beckworth)**
    Focus: Real-time monetary policy and nowcasting applications.

25. **Liberty Street Economics (NY Fed Blog)**
    URL: https://libertystreeteconomics.newyorkfed.org/

    **Topic searches:** "Nowcasting", "Real-time data", "GDP forecasting"

## Study Plan Recommendations

### Week 1: Nowcasting Foundations
- Read papers 1, 2 (Giannone et al., Bańbura & Modugno)
- Explore NY Fed Nowcast website (#21)
- Download FRED-MD data (#18)

### Week 2: Forecast Evaluation
- Read papers 3, 6, 7 (Diebold-Mariano, Gneiting-Raftery, Clark-McCracken)
- Implement DM test from scratch
- Compare your nowcast to SPF consensus (#20)

### Week 3: Missing Data & Real-Time Issues
- Read papers 10, 13, 14 (Doz et al., Croushore-Stark, Aruoba et al.)
- Download ALFRED vintages (#19)
- Backtest with real-time data

### Week 4: Production Systems
- Study NY Fed Nowcast code (#21)
- Read comprehensive review (#4: Bok et al.)
- Build your own automated nowcasting pipeline

## Citation Management

All papers listed are peer-reviewed and published. Recommended citation manager: Zotero (free, integrates with FRED and NBER).

## Questions or Suggestions?

If you find additional valuable resources, please share them with the course community.

---

*Last updated: 2024*
