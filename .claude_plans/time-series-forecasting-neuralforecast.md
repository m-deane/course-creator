# Course Plan: Modern Time Series Forecasting with NeuralForecast

## Status: In Progress
## Created: 2026-04-03

## Course Overview
Build mastery in probabilistic time series forecasting using the neuralforecast ecosystem.
Covers sample paths for uncertainty quantification, model explainability, and state-of-the-art
architectures (XLinear). Based on four technical articles from datasciencewithmarco.com and
minimizeregret.com.

## Source Articles
1. "Use Sample Paths Instead of Quantiles" — minimizeregret.com (theory)
2. "Sample Paths for Uncertainty Quantification" — datasciencewithmarco.com (implementation)
3. "Explainability for Deep Learning Models in Time Series" — datasciencewithmarco.com
4. "Discover XLinear for State-of-the-Art Forecasting" — datasciencewithmarco.com

## Key Libraries
- neuralforecast (v3.1.6+), datasetsforecast, utilsforecast
- captum, shap (explainability)
- pandas, numpy, matplotlib

## Datasets (Real Only)
1. French Bakery Daily Sales (Kaggle) — baguette sales
2. Blog Traffic — daily visitors 2020-2023
3. ETTm1 — 7-variable multivariate, 15-min intervals

## Module Plan

| Module | Topic | Guides | Notebooks | Slides |
|--------|-------|--------|-----------|--------|
| 00 | Foundations & Prerequisites | 2 | 2 | 2 |
| 01 | Point Forecasting with Neural Models | 2 | 2 | 2 |
| 02 | Probabilistic Forecasting — Why Quantiles Aren't Enough | 2 | 2 | 2 |
| 03 | Sample Paths — The Correct Uncertainty Framework | 2 | 2 | 2 |
| 04 | Explainability for Neural Forecasting | 2 | 2 | 2 |
| 05 | XLinear — State-of-the-Art Architecture | 2 | 2 | 2 |
| 06 | Production Patterns & Integration | 2 | 2 | 2 |
| 07 | Portfolio Project | 1 | 1 | 0 |

## Success Criteria
- [ ] All 8 modules have guides + companion slides + notebooks
- [ ] All notebooks use real datasets (no mocks)
- [ ] All code uses neuralforecast API (.fit, .predict, .simulate, .explain)
- [ ] Every guide has a companion _slides.md with Marp frontmatter
- [ ] Quick-start notebook runs in <2 min
- [ ] Production template is copy-paste ready
- [ ] All slides render with course theme
- [ ] No quiz.md, grading_rubric.md, or formal assessments
