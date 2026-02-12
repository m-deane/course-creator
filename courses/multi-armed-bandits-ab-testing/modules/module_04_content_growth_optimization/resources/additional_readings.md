# Additional Readings: Bandits for Content & Growth

## Academic Papers

### Multi-Armed Bandits for Experimentation

**"A Modern Bayesian Look at the Multi-Armed Bandit" (2010)**
- Authors: Steven L. Scott
- Source: Applied Stochastic Models in Business and Industry
- Key insight: Thompson Sampling is optimal for many practical problems
- Link: https://proceedings.mlr.press/v23/scott12.html
- Why read: Foundational paper on why Thompson Sampling works so well

**"Thompson Sampling for Contextual Bandits with Linear Payoffs" (2013)**
- Authors: Agrawal & Goyal
- Source: ICML 2013
- Key insight: Extends Thompson Sampling to contextual settings
- Relevance: When conversion rates depend on user features
- Link: https://arxiv.org/abs/1209.3352

**"Online Experimentation at Microsoft" (2013)**
- Authors: Kohavi, Henne, Sommerfield
- Source: ACM SIGKDD
- Key insight: Real-world challenges of A/B testing at scale
- Why read: Explains when traditional A/B testing breaks down
- Link: Microsoft Research publications
- Quote: "We waste half our traffic on inferior variants for weeks"

### Content Optimization

**"Practical Guide to Controlled Experiments on the Web" (2007)**
- Authors: Kohavi et al.
- Key insight: A/B testing best practices from Google, Microsoft, Amazon
- Relevance: Shows limitations that bandits address
- Link: https://ai.stanford.edu/~ronnyk/

**"Multi-Armed Bandits in the Wild: Pitfalls and Strategies" (2019)**
- Authors: Bouneffouf & Féraud
- Source: Applied Data Science track
- Key insight: Common mistakes in production bandit systems
- Why read: Learn from others' failures before deploying
- Link: https://arxiv.org/abs/1904.10040

## Industry Blog Posts & Case Studies

### Engineering Blogs

**Optimizely Engineering Blog**
- "Stats Engine: Multi-Armed Bandit Algorithm"
- Real-world implementation of Thompson Sampling for web optimization
- Link: https://www.optimizely.com/optimization-glossary/multi-armed-bandit/
- Takeaway: How to transition from A/B tests to bandits

**VWO (Visual Website Optimizer) Blog**
- "The Ultimate Guide to A/B Testing Stats"
- Comparison of frequentist A/B testing vs Bayesian bandits
- Link: https://vwo.com/ab-testing/
- Practical advice on choosing between approaches

**Netflix Tech Blog**
- "It's All A/Bout Testing: The Netflix Experimentation Platform"
- How Netflix runs thousands of experiments simultaneously
- Relevance: Multi-armed bandits for content recommendation
- Link: https://netflixtechblog.com/
- Key insight: Bandits for personalized experiences

**Booking.com Engineering**
- "How Booking.com Uses Machine Learning to Improve Customer Experience"
- Real-time optimization using contextual bandits
- Scale: Millions of daily decisions
- Link: Booking.com tech blog

**Stitch Fix Algorithms Blog**
- "Multi-Armed Bandits and the Stitch Fix Experimentation Platform"
- Fashion personalization using Thompson Sampling
- Link: https://algorithms-tour.stitchfix.com/
- Relevance: Shows bandits work beyond tech companies

### Creator Economy & Content Strategy

**"The Creator Bandit Playbook" (2023)**
- Author: Shenggang Li
- Source: Medium
- The original article that inspired this module
- Framework: 6 arms (3 topics × 2 formats), 3-week exploration, quarterly retirement
- Link: Search Medium for "Creator Bandit Playbook"
- Why read: Direct inspiration for content optimization strategies

**"How We Grew Our Newsletter to 100K Subscribers Using Bandits"**
- Various creator case studies
- Platforms: Substack, ConvertKit, beehiiv
- Pattern: Test subject lines, send times, content mix adaptively
- Typical lift: 15-25% in engagement

**"Content Marketing with Multi-Armed Bandits" (ConversionXL)**
- Practical guide to applying bandits to blog topics
- Tools: Google Optimize, Convert.com
- Case studies with real lift numbers
- Link: https://conversionxl.com/

### E-Commerce & Conversion Optimization

**"Dynamic Pricing and Multi-Armed Bandits" (2020)**
- Context: E-commerce pricing optimization
- Key insight: Price elasticity changes over time → non-stationary bandits
- Link: Search for papers on dynamic pricing with MAB

**"How We Increased Conversions by 20% with Thompson Sampling"**
- Various e-commerce case studies
- Common pattern: Landing page headlines, CTAs, product images
- Tools: Optimizely, VWO, Google Optimize
- Result: Faster convergence than A/B tests

## Books

### Core References

**"Bandit Algorithms" (2020)**
- Authors: Lattimore & Szepesvári
- Publisher: Cambridge University Press
- Level: Advanced (mathematical)
- Free online: https://tor-lattimore.com/downloads/book/book.pdf
- Chapters 1-3, 6: Essential theory
- Chapters 11-14: Contextual bandits
- Why read: The definitive mathematical treatment

**"Introduction to Multi-Armed Bandits" (2019)**
- Authors: Slivkins
- Publisher: Foundations and Trends in Machine Learning
- Level: Intermediate
- Link: https://arxiv.org/abs/1904.07272
- Why read: More accessible than Lattimore, still rigorous

**"Trustworthy Online Controlled Experiments" (2020)**
- Authors: Kohavi, Tang, Xu
- Publisher: Cambridge University Press
- Focus: A/B testing at scale (Microsoft, Google, Amazon)
- Relevance: Explains when to use bandits instead of A/B tests
- Practical: Lots of real-world examples

### Business & Application

**"Lean Analytics" (2013)**
- Authors: Croll & Yoskovitz
- Relevant chapters: Metrics selection, experimentation
- Why read: How to choose the right reward metric
- Takeaway: Avoid vanity metrics, optimize for business value

**"Hacking Growth" (2017)**
- Authors: Ellis & Brown
- Relevant sections: Experimentation frameworks
- Connection: Bandits fit naturally into growth hacking workflows
- Case studies: Airbnb, Uber, Facebook growth teams

## Technical Resources

### Open-Source Libraries

**Vowpal Wabbit**
- Fast contextual bandit implementation
- Used in production at Microsoft, Yahoo
- Link: https://github.com/VowpalWabbit/vowpal_wabbit
- Tutorial: Contextual bandits for content recommendation

**PyMC (for Thompson Sampling)**
- Bayesian inference library
- Perfect for implementing custom Thompson Sampling bandits
- Link: https://www.pymc.io/
- Example: Beta-Bernoulli for conversion optimization

**Epsilon (Facebook Research)**
- Full-featured bandit framework
- Link: Search for "Facebook Epsilon bandits"
- Use case: Large-scale experimentation

**scikit-learn contrib**
- Contains some bandit implementations
- Link: https://github.com/scikit-learn-contrib

### Tutorials & Courses

**CS234: Reinforcement Learning (Stanford)**
- Instructor: Emma Brunskill
- Lectures 3-5: Multi-armed bandits
- Free online: http://web.stanford.edu/class/cs234/
- Why watch: Clear explanations with proofs

**David Silver's RL Course (UCL/DeepMind)**
- Lecture 9: Exploration and exploitation
- YouTube: Search "David Silver RL"
- Level: Intermediate
- Covers: UCB, Thompson Sampling, Gittins index

**Fast.ai Practical Deep Learning**
- Philosophy: Working code first, theory contextually
- Relevant: Approach to experimentation and iteration
- Link: https://www.fast.ai/
- Mindset: Ship and learn, not perfect then ship

## Commodity Trading Specific

### Algorithmic Trading & Bandits

**"Multi-Armed Bandits for Dynamic Trading Strategies"**
- Various papers on strategy selection
- Key insight: Market regimes are non-stationary → discounted bandits
- Application: Rotate between strategies as markets change

**"Online Learning in Algorithmic Trading" (2018)**
- Context: Portfolio allocation with bandits
- Challenges: Non-stationarity, delayed feedback, risk constraints
- Techniques: Contextual bandits with regime features

### Research Reports & Analysis

**CME Group Research**
- "Adaptive Execution Strategies"
- Relevance: Similar explore-exploit tradeoff in execution
- Link: CME Institute publications

**Energy Information Administration (EIA)**
- Weekly crude oil reports
- Relevance: Example of content optimization problem
  - "Which format drives the most trader action?"
- Link: https://www.eia.gov/

## Blogs & Newsletters to Follow

**Towards Data Science (Medium)**
- Search for: "multi-armed bandits", "Thompson Sampling"
- Quality varies, but good practical tutorials
- Filter: Look for articles with working code

**Chip Huyen's Blog**
- ML systems design
- Relevant posts on experimentation in production
- Link: https://huyenchip.com/

**Eugene Yan's Blog**
- Applied ML
- Several posts on recommendation systems using bandits
- Link: https://eugeneyan.com/

**Lenny's Newsletter**
- Product growth strategies
- Occasional deep-dives on experimentation
- Link: https://www.lennysnewsletter.com/

## Video Lectures

**"Thompson Sampling: Simple, Scalable, and Effective" (NIPS 2015)**
- Speaker: Ben Van Roy (Stanford)
- 45 minutes
- Link: Search YouTube for "Ben Van Roy Thompson Sampling"
- Why watch: Intuitive explanation from the master

**"Multi-Armed Bandits in Production" (Various Tech Talks)**
- Google TechTalks
- Netflix Tech Talks
- Microsoft Research talks
- Search: "Multi-armed bandits [company name]"

## Tools & Platforms

### Commercial Experimentation Platforms

**Optimizely**
- Full-featured experimentation platform
- Includes MAB support
- Link: https://www.optimizely.com/

**VWO (Visual Website Optimizer)**
- A/B testing + bandits
- Strong for e-commerce
- Link: https://vwo.com/

**Google Optimize** (Being deprecated 2023)
- Was: Free A/B testing tool
- Now: Migrating to Google Analytics 4
- Lesson: Don't rely solely on third-party tools

**Statsig**
- Modern experimentation platform
- Strong bandit support
- Link: https://statsig.com/

### Analytics & Monitoring

**Mixpanel**
- User analytics
- Can track bandit performance
- Link: https://mixpanel.com/

**Amplitude**
- Product analytics
- Experimentation features
- Link: https://amplitude.com/

## Research Groups to Follow

**Microsoft Research**
- Experimentation Platform team
- Many papers on large-scale A/B testing and bandits
- Link: https://www.microsoft.com/en-us/research/

**Google Research**
- Search & recommendation teams
- Cutting-edge bandit applications
- Link: https://research.google/

**Meta (Facebook) Research**
- Large-scale experimentation
- Epsilon platform papers
- Link: https://research.facebook.com/

## Practical Implementation Guides

**"Building a Multi-Armed Bandit from Scratch"**
- Various blog tutorials
- Best: Those with complete working code
- Look for: Beta-Bernoulli Thompson Sampling examples

**"Production Bandit Checklist"**
- Logging and monitoring
- A/A tests (sanity checks)
- Gradual rollout strategies
- Fallback mechanisms

## Advanced Topics (For Later)

**Contextual Bandits**
- LinUCB algorithm
- Neural bandits
- When features matter (user demographics, market conditions)

**Non-Stationary Bandits**
- Discounted Thompson Sampling
- Sliding window estimates
- Change detection algorithms

**Adversarial Bandits**
- EXP3 algorithm
- When rewards are chosen by an adversary
- Relevant for competitive markets

**Constrained Bandits**
- Budget constraints
- Fairness constraints
- Risk constraints (for trading)

## Communities & Forums

**r/MachineLearning (Reddit)**
- Active discussions on bandits
- Search for: "multi-armed bandits", "A/B testing"

**Cross Validated (StackExchange)**
- Statistical questions
- Good for: "When should I use bandits vs A/B test?"

**Kaggle**
- Some competitions use bandit-like evaluation
- Learn from winner solutions

## Key Takeaways for This Module

**Must-Read (Start Here):**
1. "The Creator Bandit Playbook" (Medium) — practical framework
2. Optimizely blog on Thompson Sampling — implementation guide
3. Scott (2010) "Modern Bayesian Look" — why Thompson Sampling works

**Deep Dive (If You Have Time):**
4. Lattimore & Szepesvári book (Chapters 1-3, 6) — theory
5. Kohavi et al. "Online Experimentation at Microsoft" — real-world scale
6. VWO blog on metric selection — avoid vanity metrics

**For Commodity Traders:**
7. Any paper on "strategy selection with bandits"
8. EIA reports (as examples of content optimization)
9. Case studies on non-stationary bandits (regime changes)

**Practical Implementation:**
10. PyMC tutorial on Beta-Bernoulli models
11. Any blog with complete Thompson Sampling code
12. Experimentation platform documentation (Optimizely, VWO)

---

**Next Steps:**
- Pick ONE paper/blog post from "Must-Read" section
- Implement the example code in your own project
- Share learnings with your team
- Consider: "What's one decision we make repeatedly that could be a bandit?"
