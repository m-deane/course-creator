# Dynamic Factor Models: Extract Signal from Noisy Economic Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/dynamic-factor-models)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-repo/dynamic-factor-models/main)

**Get working code in 2 minutes.** Extract latent factors from 100+ economic indicators, build nowcasting models for GDP before official releases, and handle mixed-frequency data with ragged edges.

## Start Here: Quick Wins

**New to factor models?** Start with these working examples:

1. **[Your First Factor Model](quick-starts/00_hello_dfm.ipynb)** (2 min)
   - Extract 3 factors from 20 indicators with 5 lines of code
   - Visualize what each factor represents
   - Copy-paste ready for your own data

2. **[Forecast Economic Data](quick-starts/01_your_first_forecast.ipynb)** (5 min)
   - Predict industrial production using multiple indicators
   - Works with live FRED API data
   - Update predictions as new data arrives

3. **[Handle Missing Data & Multiple Series](quick-starts/02_multivariate_example.ipynb)** (7 min)
   - Deal with publication lags automatically
   - Extract factors from 10+ time series
   - Production-ready patterns for real-time forecasting

**Have experience?** Grab templates:
- [Production Factor Model Pipeline](templates/dfm_pipeline_template.py) - End-to-end data → factors → forecasts
- [Nowcasting Template](templates/nowcasting_template.py) - State-space DFM with EM estimation

## What You'll Build

By the end of this course, you'll have a portfolio of working projects:

- **Real-Time Nowcasting Dashboard** - Predict GDP, inflation, unemployment before official releases
- **Multi-Asset Factor Extractor** - Find common drivers across 100+ financial time series
- **Production Forecasting Pipeline** - Automated data → factors → predictions with monitoring

## Learning Path

```
START → Quick-Starts (Get it working)
          ↓
       Templates (Copy for your projects)
          ↓
       Recipes (Patterns for common tasks)
          ↓
       Modules (Structured deep-dive) ← Optional, if you want theory
          ↓
       Projects (Build portfolio pieces)
```

### Quick-Starts: Working Examples (Start Here!)
Get something running in minutes:
- `00_hello_dfm.ipynb` - Extract factors in 5 lines
- `01_your_first_forecast.ipynb` - Real-time forecasting with FRED data
- `02_multivariate_example.ipynb` - Handle 10+ series with missing data

### Templates: Production-Ready Code (Copy These!)
Scaffolds for real projects:
- `dfm_pipeline_template.py` - Complete data → model → forecast pipeline
- `nowcasting_template.py` - Real-time nowcasting system

### Recipes: Copy-Paste Patterns
Solve specific problems:
- Load FRED-MD data with transformations
- Select optimal number of factors
- Handle ragged edges in real-time data
- Compute factor contributions to forecasts
- Evaluate nowcast accuracy with vintages

### Concepts: Visual Guides (60-Second Summaries)
One-page visual explanations:
- What is a factor? (diagram + code)
- How Kalman filtering works (animation)
- Why PCA finds factors (visual proof)
- When to use DFMs vs alternatives

### Modules: Structured Learning (Optional)
Full course path for deep understanding:
- **Module 0**: Foundations (state-space models, Kalman filter)
- **Module 1**: DFM Theory (factor models, specification, identification)
- **Module 2**: Estimation (ML, EM algorithm, Bayesian methods)
- **Module 3**: Applications (nowcasting, forecasting, missing data)
- **Module 4**: Extensions (time-varying, mixed frequency, large datasets)

### Projects: Portfolio Builders
Deploy these to showcase your skills:
- **Beginner**: GDP Nowcasting Dashboard (visualize factor contributions)
- **Intermediate**: Multi-Asset Factor Monitor (track common shocks)
- **Advanced**: Real-Time Forecasting Platform (auto-update with new data)

## Prerequisites

**Required:**
- Python basics (functions, loops, NumPy arrays)
- Linear algebra intuition (matrix multiplication, eigenvalues)
- Basic time series (AR processes, stationarity)

**Helpful but not required:**
- State-space models
- Maximum likelihood estimation
- Bayesian statistics

**Don't have these?** Start with Module 0 quick refreshers.

## Technology Stack

**Core:**
- NumPy/SciPy for matrix operations
- Pandas for data wrangling
- Statsmodels for state-space models

**Optional (for specific use cases):**
- PyMC for Bayesian estimation
- JAX for fast Kalman filtering
- Scikit-learn for PCA and validation

**No installation required:** All notebooks run in Google Colab with zero setup.

## Quick Setup

### Option 1: Colab (Recommended - Zero Setup)
Click any notebook's Colab badge → Run all cells. That's it.

### Option 2: Local Installation
```bash
pip install numpy pandas statsmodels matplotlib scikit-learn
pip install pandas-datareader  # For FRED data access
```

See [detailed setup guide](resources/setup.md) for advanced options.

## Real-World Applications

**When to use Dynamic Factor Models:**
- ✅ 50+ correlated time series with common drivers
- ✅ Need to forecast before slow-releasing official data
- ✅ Data at different frequencies (daily + monthly + quarterly)
- ✅ Missing observations or ragged edges
- ✅ Want to understand what drives comovement

**When NOT to use:**
- ❌ Only 5-10 series (use VAR or direct modeling)
- ❌ Series are independent (no common factors)
- ❌ Need instant predictions (factor extraction takes time)
- ❌ Perfect data, same frequency, no lags (simpler methods work)

See [decision flowchart](resources/cheat_sheet.md) for detailed guidance.

## Data Sources

All examples use real, publicly available data:
- **FRED-MD** - 127 monthly US macroeconomic indicators (updated monthly)
- **FRED-QD** - Quarterly macroeconomic database
- **ALFRED** - Real-time vintage data for proper backtesting
- **Yahoo Finance** - Financial market data
- **Your own data** - Templates work with any CSV/DataFrame

Free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

## Course Philosophy

**Practical-First Learning:**
1. See it work (run the code)
2. Understand what it does (visualize outputs)
3. Modify it (parameter playground)
4. Copy it (use in your projects)
5. Master it (optional theory deep-dives)

**No Grades, Build Portfolio:**
- Every project is a real, deployable tool
- Self-check exercises with instant feedback
- "Modify This" challenges to extend examples
- Share your nowcasting dashboard with potential employers

## Key References

**Foundational Papers** (optional reading):
- Stock & Watson (2002) - Principal components approach
- Bai & Ng (2002) - Determining number of factors
- Giannone et al. (2008) - Nowcasting with real-time data

**Practical Resources** (read these first):
- [Visual Guide to Factor Models](concepts/visual_guides/what_are_factors.md)
- [Kalman Filter Intuition](concepts/visual_guides/kalman_filter.md)
- [Cheat Sheet](resources/cheat_sheet.md) - 1-page reference

## What Makes This Course Different

**Traditional Approach:**
1. Study factor model theory (3 weeks)
2. Learn state-space math (2 weeks)
3. Derive Kalman filter (2 weeks)
4. Finally code something (week 8)

**Our Approach:**
1. Extract factors from FRED data (minute 2)
2. Build GDP nowcast (minute 10)
3. Deploy as dashboard (minute 30)
4. Understand why it works (when you're curious)

## Success Path

**Week 1:** Run all quick-starts → Pick a template → Use your own data
**Week 2:** Build beginner project → Deploy it → Share the link
**Week 3:** Dig into concepts you're curious about
**Week 4+:** Build intermediate/advanced projects for your portfolio

## Support & Community

- **Cheat Sheet** - [1-page reference](resources/cheat_sheet.md)
- **Glossary** - [Key terms with examples](resources/glossary.md)
- **Recipes** - [Common patterns](recipes/common_patterns.py)
- **Troubleshooting** - [Error fixes](recipes/troubleshooting.md)

## Quick Links

| Resource | Description |
|----------|-------------|
| [Quick-Starts](quick-starts/) | 5-10 min working examples |
| [Templates](templates/) | Production-ready scaffolds |
| [Recipes](recipes/) | Copy-paste code patterns |
| [Projects](projects/) | Portfolio builders |
| [Concepts](concepts/) | Visual guides + theory |
| [Modules](modules/) | Structured learning path |
| [Cheat Sheet](resources/cheat_sheet.md) | 1-page reference |
| [Setup](resources/setup.md) | Environment configuration |
| [Glossary](resources/glossary.md) | Key terms |

## License & Attribution

All course materials are open source. If you build something cool with these tools, we'd love to hear about it!

---

**Ready to start?** → [Open your first notebook](quick-starts/00_hello_world.ipynb) and extract factors in 2 minutes.

*"The best way to learn factor models is to run factor models."*
