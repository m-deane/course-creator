# Additional Readings and Resources

Curated references for deeper learning on bandit-based commodity trading systems.

## Core Papers on Multi-Armed Bandits for Portfolio Selection

### Foundational Research

1. **Sani, A., Lazaric, A., & Munos, R. (2012)**
   "Risk-Aversion in Multi-armed Bandits"
   *Advances in Neural Information Processing Systems (NeurIPS)*
   - Risk-sensitive bandit algorithms
   - Variance-penalized rewards
   - Relevant for commodity trading with downside protection

2. **Huo, X., & Fu, F. (2017)**
   "Risk-Aware Multi-Armed Bandit Problem with Application to Portfolio Selection"
   *Royal Society Open Science*
   - Combines mean-variance optimization with bandits
   - Direct application to portfolio allocation
   - [Link](https://royalsocietypublishing.org/doi/10.1098/rsos.171377)

3. **Shen, W., Wang, J., Jiang, Y., & Zha, H. (2015)**
   "Portfolio Choices with Orthant Ordering of Assets"
   *ICML 2015*
   - Contextual bandits for portfolio selection
   - Regime-dependent allocation strategies
   - [Link](http://proceedings.mlr.press/v37/shenb15.pdf)

### Thompson Sampling for Finance

4. **Russo, D., & Van Roy, B. (2014)**
   "Learning to Optimize via Posterior Sampling"
   *Mathematics of Operations Research*
   - Theoretical foundations of Thompson Sampling
   - Regret bounds and optimality
   - [Link](https://arxiv.org/abs/1301.2609)

5. **Agrawal, S., & Goyal, N. (2013)**
   "Thompson Sampling for Contextual Bandits with Linear Payoffs"
   *ICML 2013*
   - Contextual Thompson Sampling (used in regime-aware allocation)
   - Theoretical guarantees
   - [Link](http://proceedings.mlr.press/v28/agrawal13.pdf)

## Commodity Trading Systems

### Systematic Commodity Strategies

6. **Erb, C., & Harvey, C. (2006)**
   "The Strategic and Tactical Value of Commodity Futures"
   *Financial Analysts Journal*
   - Momentum and carry in commodity futures
   - Portfolio construction insights
   - [Link](https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P74_The_strategic_and.pdf)

7. **Gorton, G., & Rouwenhorst, K. (2006)**
   "Facts and Fantasies about Commodity Futures"
   *Financial Analysts Journal*
   - Long-term commodity return drivers
   - Diversification benefits
   - Classic reference for commodity portfolios

8. **Fuertes, A., Miffre, J., & Rallis, G. (2010)**
   "Tactical Allocation in Commodity Futures Markets: Combining Momentum and Term Structure Signals"
   *Journal of Banking & Finance*
   - Combining multiple commodity signals
   - Relevant for multi-arm bandit feature engineering
   - [Link](https://www.sciencedirect.com/science/article/abs/pii/S0378426609003124)

### Risk Management in Commodities

9. **Kat, H., & Oomen, R. (2007)**
   "What Every Investor Should Know About Commodities, Part II: Multivariate Return Analysis"
   *Journal of Investment Management*
   - Correlation structures in commodity markets
   - Regime-switching behavior
   - Informs correlation guardrails

10. **Basu, D., & Miffre, J. (2013)**
    "Capturing the Risk Premium of Commodity Futures: The Role of Hedging Pressure"
    *Journal of Banking & Finance*
    - Supply/demand imbalances in commodity pricing
    - Relevant for inventory-based features

## Regime Detection and Market Microstructure

### Regime-Switching Models

11. **Ang, A., & Bekaert, G. (2002)**
    "Regime Switches in Interest Rates"
    *Journal of Business & Economic Statistics*
    - Hidden Markov Models for regime detection
    - Statistical methods for identifying market states

12. **Guidolin, M., & Timmermann, A. (2007)**
    "Asset Allocation under Multivariate Regime Switching"
    *Journal of Economic Dynamics and Control*
    - Portfolio optimization under regime uncertainty
    - Theoretical foundation for regime-aware allocation
    - [Link](https://www.sciencedirect.com/science/article/abs/pii/S0165188906001230)

### Volatility and Term Structure

13. **Fama, E., & French, K. (1987)**
    "Commodity Futures Prices: Some Evidence on Forecast Power, Premiums, and the Theory of Storage"
    *Journal of Business*
    - Term structure of commodity futures
    - Backwardation vs contango regimes
    - Classic reference

14. **Trolle, A., & Schwartz, E. (2009)**
    "Unspanned Stochastic Volatility and the Pricing of Commodity Derivatives"
    *Review of Financial Studies*
    - Volatility dynamics in commodities
    - Relevant for volatility-based regime features

## Guardrails and Risk Constraints

### Position Sizing and Constraints

15. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**
    "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"
    *Review of Financial Studies*
    - Case for simple constraints (equal-weight core)
    - Robustness of naive diversification
    - [Link](https://academic.oup.com/rfs/article/22/5/1915/1592901)

16. **Jagannathan, R., & Ma, T. (2003)**
    "Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps"
    *Journal of Finance*
    - Benefits of position limits and constraints
    - Why guardrails improve out-of-sample performance

### Transaction Costs and Turnover

17. **Gârleanu, N., & Pedersen, L. (2013)**
    "Dynamic Trading with Predictable Returns and Transaction Costs"
    *Journal of Finance*
    - Optimal turnover under transaction costs
    - Relevant for tilt speed limits
    - [Link](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12012)

## Connections to Other Courses in This Repository

### Bayesian Methods for Commodities

**Bayesian Commodity Forecasting Course**
- Enhanced regime detection using Bayesian state-space models
- Prior specification for commodity-specific beliefs
- Uncertainty quantification for allocation decisions
- **Integration point:** Use Bayesian forecasts as contextual features for bandits

**Relevant modules:**
- Module 3: State-space models for commodity prices
- Module 5: Regime-switching Bayesian models
- Module 7: Combining multiple information sources

### Regime Detection with HMMs

**Hidden Markov Models Course**
- Probabilistic regime detection (vs rule-based)
- Regime transition probabilities
- Viterbi algorithm for most likely regime sequence
- **Integration point:** HMM states as contexts for contextual bandits

**Relevant modules:**
- Module 2: Gaussian HMMs for continuous observations
- Module 4: Regime identification in financial time series
- Module 6: Online filtering and regime prediction

### GenAI for Enhanced Features

**GenAI for Commodities Course**
- LLM-based sentiment analysis from news
- Supply shock detection from text data
- Inventory surprise extraction
- **Integration point:** LLM outputs as additional contextual features

**Relevant modules:**
- Module 3: Commodity-specific news sentiment
- Module 5: Supply/demand narrative extraction
- Module 7: Combining fundamental and technical signals

### Panel Methods for Cross-Commodity Patterns

**Panel Regression Course**
- Cross-commodity momentum and mean-reversion
- Identifying common factors across commodities
- Sectoral patterns (energy, metals, grains)
- **Integration point:** Panel regression factors as regime features

**Relevant modules:**
- Module 2: Fixed effects for commodity-specific patterns
- Module 4: Dynamic panel models
- Module 6: Factor structures in commodity returns

## Practical Implementation Resources

### Software Libraries

**Python Libraries:**
- `scipy.stats`: For statistical distributions (Thompson Sampling)
- `scikit-learn`: For clustering (regime detection via K-means)
- `hmmlearn`: For Hidden Markov Models (advanced regime detection)
- `yfinance`: For historical commodity futures data
- `pandas`: For time series manipulation
- `numpy`: For numerical computations

**R Libraries (Alternative):**
- `bandit`: Multi-armed bandit implementations
- `depmixS4`: Hidden Markov Models and regime-switching
- `quantmod`: Financial data and technical indicators

### Data Sources

**Free Commodity Data:**
- Yahoo Finance (`yfinance` library): Futures data (limited history)
- Quandl: Commodity prices and fundamentals (free tier available)
- EIA (Energy Information Administration): Oil, gas inventories and prices
- USDA: Agricultural commodity data

**Premium Data Providers:**
- Bloomberg Terminal: Professional-grade commodity data
- Refinitiv Eikon: Futures, term structure, fundamentals
- CME Group: Direct futures exchange data

### Backtesting Frameworks

- `zipline`: Algorithmic trading backtesting (Python)
- `backtrader`: Event-driven backtesting with live trading support
- `vectorbt`: Fast vectorized backtesting for research
- Custom implementation (as shown in notebooks): Full control, educational

## Industry Applications and Case Studies

### Commodity Trading Firms Using Systematic Strategies

1. **AQR Capital Management**
   - Momentum and carry strategies across asset classes including commodities
   - Academic research on factor investing
   - [Research](https://www.aqr.com/Insights/Research)

2. **Winton Group**
   - Statistical arbitrage and systematic trading in commodities
   - Machine learning for pattern recognition
   - Similar to bandit-based adaptive allocation

3. **Man Group (AHL)**
   - Trend-following and diversified systematic strategies
   - Commodities as part of multi-asset portfolios

### Institutional Applications

- **Commodity Producers**: Hedging with adaptive position sizing
- **Endowments**: Commodity allocation in alternative portfolios
- **Hedge Funds**: Tactical commodity tilts based on market regimes
- **Pension Funds**: Strategic commodity exposure with guardrails

## Recommended Learning Path

### For Beginners
1. Start with Module 1 (Multi-Armed Bandits basics)
2. Read Thompson Sampling papers (Russo & Van Roy, Agrawal & Goyal)
3. Review commodity fundamentals (Erb & Harvey, Gorton & Rouwenhorst)
4. Complete Module 5 notebooks

### For Intermediate Practitioners
1. Study contextual bandits (Shen et al., Agrawal & Goyal)
2. Implement regime detection (Ang & Bekaert, Guidolin & Timmermann)
3. Add transaction costs (Gârleanu & Pedersen)
4. Backtest on historical data with multiple guardrails

### For Advanced Researchers
1. Integrate HMMs for regime detection (see HMM course)
2. Combine with Bayesian forecasting (see Bayesian course)
3. Add LLM-based features (see GenAI course)
4. Deploy with live data and monitoring

## Open Questions and Future Research

### Active Research Areas

1. **Multi-objective bandits for trading**
   - Balancing multiple conflicting goals (return, risk, ESG)
   - Pareto-optimal allocation strategies

2. **Non-stationary bandits**
   - Adapting to structural breaks in commodity markets
   - Distinguishing regime shifts from noise

3. **Hierarchical bandits**
   - Sector-level vs commodity-level allocation
   - Learning at multiple timescales

4. **Safe reinforcement learning for trading**
   - Formal guarantees on maximum loss
   - Constrained optimization with bandits

### Unanswered Practical Questions

- **Optimal core/bandit split**: Is 80/20 universal or commodity-specific?
- **Regime persistence**: How to incorporate regime duration in contextual bandits?
- **Feature engineering**: Which commodity-specific features matter most?
- **Guardrail tuning**: Data-driven methods to set position limits?

## Contributing to the Field

If this module inspired research ideas:
- Implement and backtest novel reward functions
- Test on broader commodity universes (50+ commodities)
- Compare bandit methods to traditional optimization
- Publish results and share with the community

## Contact and Community

**Questions or want to share your implementations?**
- Open an issue in this repository
- Join commodity trading forums (EliteTrader, QuantConnect)
- Share backtests on Twitter/X with #commoditytrading #bandits

---

**This is a living document.** If you find valuable resources not listed here, please contribute!

**Last updated:** 2024-01-01
