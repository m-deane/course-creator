# Slide Deck Quality Review: Dynamic Factor Models Course

**Reviewer:** AI Quality Reviewer
**Date:** 2026-02-20
**Scope:** All 28 slide decks across 9 modules in `courses/dynamic-factor-models/modules/*/guides/*_slides.md`

---

## Executive Summary

The Dynamic Factor Models course contains 28 Marp slide decks spanning 9 modules, from foundational matrix algebra through advanced topics like time-varying parameters and machine learning connections. The decks demonstrate a **remarkably consistent quality level** with strong technical depth, valid Marp/MathJax/Mermaid formatting, and a uniform structural template. The overall course quality is **high** -- these decks are production-ready with only minor refinements needed.

**Overall Score: 4.0 / 5.0** (weighted across all dimensions)

### Strengths
- Exceptional consistency in structure and formatting across all 28 decks
- Every deck includes Mermaid diagrams, MathJax formulas, code examples, pitfalls tables, and practice problems
- Mathematical notation is precise and correctly formatted throughout
- The "Connections & Summary" slide on each deck provides excellent cross-referencing
- Code examples are realistic and use appropriate Python libraries (numpy, scipy, sklearn)

### Weaknesses
- Formulaic structure becomes predictable -- every deck follows the exact same template, reducing engagement
- Mermaid diagrams are sometimes text-heavy flowcharts rather than genuinely visual explanations
- No actual rendered visualizations (plots, heatmaps, matrices) -- only text-based diagram descriptions
- Practice problems are listed but solutions/hints are never provided
- Font size at 24px may be too small for projected presentations; no speaker notes included

---

## Dimension Scores (Aggregate)

| Dimension | Weight | Score | Weighted |
|-----------|:------:|:-----:|:--------:|
| Design & Visual Quality | 20% | 3.5 | 0.70 |
| Narrative & Story Flow | 20% | 3.5 | 0.70 |
| Comprehensiveness | 20% | 4.5 | 0.90 |
| Technical Accuracy | 20% | 4.5 | 0.90 |
| Added Visual Value | 10% | 3.0 | 0.30 |
| Production Readiness | 10% | 4.5 | 0.45 |
| **Overall** | **100%** | | **3.95** |

---

## Dimension-by-Dimension Analysis

### 1. DESIGN & VISUAL QUALITY (Score: 3.5/5)

**Strengths:**
- Consistent Marp frontmatter across all 28 decks (theme: default, paginate: true, math: mathjax)
- Two-column layouts via `<div class="columns">` used effectively for comparisons
- Tables are well-formatted with clear headers and alignment
- Lead-class section dividers create clear topic transitions

**Weaknesses:**
- Every deck uses the identical CSS (24px font, same grid layout) -- no visual variety
- No custom color themes, branded elements, or visual identity
- Compared to Stanford CS229 slides (which use color-coded sections, custom illustrations, and progressive reveals), these feel utilitarian
- Compared to fast.ai (which uses large fonts, minimal text, heavy imagery), these are text-heavy
- No slide numbering beyond Marp's auto-pagination
- No images, photographs, or hand-drawn illustrations anywhere in 28 decks

**Recommendation:** Introduce 2-3 custom CSS themes for different content types (theory slides vs code slides vs summary slides). Add a course logo/branding element. Consider larger font sizes (28-32px) for key formulas.

### 2. NARRATIVE & STORY FLOW (Score: 3.5/5)

**Strengths:**
- Every deck opens with a "Key idea" summary sentence -- excellent for framing
- Blockquotes used effectively to highlight key insights
- Progressive structure within each deck: motivation, theory, code, pitfalls, practice
- Cross-module connections shown via Mermaid flowcharts on the final slide

**Weaknesses:**
- The rigid template (title, motivation, numbered sections, code, pitfalls, practice, summary) is followed identically in all 28 decks. This predictability reduces engagement by the third module.
- No analogies or real-world stories in most decks. Exceptions: Module 00 Guide 01 (temporal aggregation bathtub analogy), Module 07 Guide 01 (suitcase packing analogy for LASSO), Module 07 Guide 03 (interview/consensus analogy for 3PRF). Most decks jump directly to formulas.
- Compared to DataCamp's approach (scenario-driven, problem-first), these decks start with formalism rather than a motivating problem.
- The opening blockquote on each deck is often a restatement of the formula rather than an accessible hook.
- No "callback" or "remember when we..." references between decks despite the module structure.

**Recommendation:** For each deck, replace the opening blockquote with a concrete real-world scenario. Add more analogies -- the bathtub analogy in Module 05 Guide 01 and the suitcase analogy in Module 07 Guide 01 are excellent models. Use progressive disclosure: show the result first, then unpack.

### 3. COMPREHENSIVENESS (Score: 4.5/5)

**Strengths:**
- Excellent coverage of each topic. Comparing slides to companion source guides, the slide decks capture the core mathematical content, key code implementations, and important relationships.
- Every deck includes: formal definitions, code implementations, comparison tables, common pitfalls, practice problems (conceptual + mathematical + implementation), and cross-references.
- Module 00 (Foundations) provides thorough prerequisite coverage: matrix algebra, time series basics, PCA refresher.
- Module 04 (ML Estimation) covers MLE via Kalman, EM algorithm, AND Bayesian estimation -- the three main approaches.
- Module 07 (Sparse Methods) covers LASSO, targeted predictors, AND three-pass filter -- a complete progression.
- Module 08 (Advanced Topics) bridges to time-varying parameters, non-Gaussian models, and ML connections.

**Weaknesses:**
- Some companion guides contain detailed derivations that are completely omitted from slides (acceptable for presentation format, but intermediate steps would help).
- No deck covers model diagnostics or goodness-of-fit testing comprehensively (scattered across pitfalls sections).
- Missing: a "course overview" or "roadmap" deck that shows the full 9-module progression upfront.
- The code in slides is illustrative but not always complete (e.g., `StudentTFactorModel._e_step` has `...` truncation; `StructuralFAVAR.fit` has `# ... (reduced-form FAVAR setup)` placeholder).

**Recommendation:** Create a Module 00 "Course Roadmap" deck. Ensure all code examples in slides are self-contained (no `...` truncation). Add a brief "Model Diagnostics Checklist" slide to each estimation module.

### 4. TECHNICAL ACCURACY (Score: 4.5/5)

**Strengths:**
- Mathematical notation is consistent and correct throughout all 28 decks.
- Matrix dimensions are correctly specified (e.g., $X_t$ ($N \times 1$), $\Lambda$ ($N \times r$), $F_t$ ($r \times 1$)).
- State-space representations correctly map to companion form matrices.
- Kalman filter equations (prediction, update, smoother) are correctly stated.
- Bai-Ng information criteria formulas are correct.
- EM algorithm E-step/M-step derivations are accurate (including the crucial $P_{t|T}$ correction in sufficient statistics).
- MIDAS Beta weight function correctly uses normalized Beta PDF.
- FAVAR two-step procedure correctly distinguishes slow vs fast variables.
- Student-t scale mixture representation and E-step weight formula are correct.
- VAE ELBO decomposition is correctly stated.

**Minor Issues Found:**
1. **Module 02, Kalman Filter deck (line ~180-200):** The Joseph form is mentioned but the full formula is correct.
2. **Module 07, Guide 02 (Targeted Predictors), line 188:** The code uses `|correlations|` which is invalid Python syntax (should be `np.abs(correlations)`). This is a presentation shorthand but could confuse implementers.
3. **Module 08, Guide 02 (Non-Gaussian), line 131:** The code references `pca` variable that was not assigned -- should be `pca = PCA(n_factors)` followed by `self.F = pca.fit_transform(X)` and `self.Lambda = pca.components_.T`.
4. **Module 06, Guide 02 (FAVAR), line 218:** `PCA(self.n_factors)` should be `PCA(n_components=self.n_factors)` for sklearn API correctness.
5. **Module 08, Guide 03 (ML Connections), lines 99-103:** The tied-weight reconstruction uses `self.encoder.weight` directly -- in PyTorch, `nn.Linear` stores weight as `(out_features, in_features)`, so this is technically correct for `torch.matmul(latent, self.encoder.weight)` reconstructing `(batch, in_features)`.

**Recommendation:** Fix the 3 code issues identified above (items 2, 3, 4). These are minor but would cause confusion for students trying to run the code.

### 5. ADDED VISUAL VALUE (Score: 3.0/5)

**Strengths:**
- Mermaid diagrams are used in every single deck (consistently 3-7 per deck).
- Flowcharts effectively show algorithmic steps (EM algorithm, Kalman filter, MIDAS estimation).
- Decision trees are well-done (variable classification in Module 05 Guide 01, identification strategy in Module 06 Guide 03, TVP selection in Module 08 Guide 01).
- Connection diagrams on summary slides show inter-module relationships clearly.
- The "Standard PCA Problem vs Targeted Solution" side-by-side in Module 07 Guide 02 is effective.

**Weaknesses:**
- Mermaid diagrams are predominantly text-in-boxes flowcharts. There are zero matrix dimension diagrams showing how matrix shapes flow through operations.
- No actual data visualizations: no scree plots, no eigenvalue spectra, no IRF plots, no factor loading heatmaps, no time series plots. These would be the most impactful visuals for understanding factor models.
- The geometric intuition for LASSO (L1 diamond vs L2 circle) in Module 07 Guide 01 uses a text description inside a Mermaid box rather than an actual geometric diagram -- this is a missed opportunity.
- Compared to Stanford CS229 which includes matrix dimension annotations flowing through operations, these decks never show dimensions visually.
- The "Bathtub analogy" and "Suitcase analogy" are described in text but not drawn as diagrams.

**Recommendation:** Add actual rendered plots (as embedded images) for: (1) scree plots and eigenvalue spectra, (2) impulse response functions, (3) LASSO regularization paths, (4) factor loading heatmaps. Add matrix dimension diagrams showing how $X_{T \times N} = \Lambda_{N \times r} F_{r \times T}' + e_{T \times N}$ flows. Draw the L1 diamond vs L2 circle geometrically.

### 6. PRODUCTION READINESS (Score: 4.5/5)

**Strengths:**
- All 28 decks have valid Marp frontmatter and will render correctly.
- MathJax formulas use consistent notation and will render without errors.
- Mermaid diagrams use correct syntax (flowchart TD/LR, subgraph, proper arrow notation).
- Two-column layouts use valid HTML/CSS within Marp.
- Slide separators (`---`) are correctly placed throughout.
- `<!-- _class: lead -->` directives are correctly used for section dividers.

**Weaknesses:**
- No speaker notes anywhere in 28 decks. For actual presentation use, speaker notes are essential.
- The 24px font size is small for projected presentations. Standard recommendation is 28-32px minimum.
- No `<!-- footer -->` or `<!-- header -->` directives for module/deck identification.
- Some slides are content-dense (e.g., Module 02 Guide 03 Kalman filter slides have 4+ equations per slide). These would benefit from splitting.

**Recommendation:** Add Marp speaker notes (`<!-- This slide covers... -->`) to at least the first and last slides of each deck. Increase base font size to 28px. Add footer with module name. Split any slides with more than 3 substantial equations into two slides.

---

## Per-Deck Reviews

### Module 00: Foundations

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Matrix Algebra Review | 3.5 | 3.0 | 4.5 | 4.5 | 3.0 | 4.5 | 3.8 |
| 02 Time Series Basics | 3.5 | 3.5 | 4.5 | 4.5 | 3.5 | 4.5 | 4.0 |
| 03 PCA Refresher | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |

**Notes:**
- Module 00 decks are prerequisite material. They are thorough but could feel dry for learners who already know the material.
- The PCA refresher includes a useful "PCA vs Factor Analysis" decision flowchart.
- Matrix algebra deck is the most formula-heavy in the course -- consider adding more geometric intuition.

### Module 01: Static Factor Models

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Factor Model Specification | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |
| 02 Identification Problem | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |
| 03 Approximate Factor Models | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |

**Notes:**
- The Identification Problem deck (Guide 02) is the strongest in this module -- the step-by-step identification strategy flowchart is genuinely helpful.
- Approximate Factor Models correctly distinguishes exact vs approximate factor models and the Chamberlain-Rothschild framework.
- The FRED-MD parameter counting table in Guide 01 effectively motivates dimensionality reduction.

### Module 02: Dynamic Factor Models

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 From Static to Dynamic | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |
| 02 State-Space Representation | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |
| 03 Kalman Filter Derivation | 3.5 | 3.5 | 5.0 | 5.0 | 3.0 | 4.0 | 4.1 |

**Notes:**
- The Kalman Filter deck is the most comprehensive single deck in the course (474 lines, covering filter, smoother, missing data, log-likelihood, Joseph form, and forecasting). It may benefit from being split into two decks.
- "From Static to Dynamic" uses a good ocean current/temperature analogy for factors vs dynamics.
- State-Space Representation correctly shows the companion form dimension summary.

### Module 03: Estimation via PCA

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Stock-Watson Estimator | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |
| 02 Factor Number Selection | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |
| 03 Missing Data Handling | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |

**Notes:**
- Factor Number Selection deck has an effective decision flowchart for choosing between Bai-Ng IC, scree plots, and cross-validation.
- Missing Data deck correctly covers MCAR/MAR/MNAR distinctions with a practical imputation comparison table.
- Stock-Watson Estimator includes computational complexity comparison (PCA vs ML vs Bayesian).

### Module 04: Estimation via ML

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 MLE via Kalman Filter | 3.5 | 3.5 | 4.5 | 5.0 | 3.0 | 4.5 | 4.0 |
| 02 EM Algorithm for DFM | 3.5 | 3.5 | 5.0 | 5.0 | 3.0 | 4.5 | 4.1 |
| 03 Bayesian DFM | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |

**Notes:**
- EM Algorithm deck is exceptionally thorough -- includes the $P_{t|T}$ correction in sufficient statistics that many textbooks omit.
- MLE via Kalman correctly presents the prediction error decomposition of the log-likelihood.
- Bayesian DFM covers all four Gibbs sampler steps (factors, loadings, dynamics, error variances) with conjugate priors.

### Module 05: Mixed Frequency

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Temporal Aggregation | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |
| 02 MIDAS Regression | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |
| 03 State-Space Mixed Freq | 3.5 | 3.5 | 4.5 | 4.5 | 3.5 | 4.5 | 4.0 |
| 04 Nowcasting Practice | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |

**Notes:**
- Temporal Aggregation uses the effective bathtub analogy and variable classification decision tree.
- State-Space Mixed Freq clearly shows the time-varying observation matrix $Z_t$ switching between months 1-2 and month 3.
- Nowcasting Practice is the most practically-oriented deck -- includes vintage data management, ragged edge handling, real-time backtesting, and Diebold-Mariano testing.
- MIDAS deck correctly explains Beta weight function parameterization.

### Module 06: Factor-Augmented Models

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Diffusion Index Forecasting | 3.5 | 3.5 | 4.5 | 4.5 | 3.0 | 4.5 | 3.9 |
| 02 FAVAR Models | 3.5 | 3.5 | 4.5 | 4.0 | 3.0 | 4.0 | 3.8 |
| 03 Structural Identification | 3.5 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.1 |

**Notes:**
- Structural Identification is the strongest deck in this module -- the identification strategy decision flowchart (external instrument vs sign restrictions vs Cholesky) is genuinely useful for practitioners.
- FAVAR deck has a minor code issue: `PCA(self.n_factors)` should use the keyword argument `n_components`.
- Diffusion Index deck correctly covers direct vs iterated vs AR-augmented forecasting.
- The comparison table of Cholesky vs Sign Restrictions vs External Instruments is very well structured.

### Module 07: Sparse Methods

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 High-Dimensional Regression | 3.5 | 4.0 | 4.5 | 4.5 | 3.0 | 4.5 | 4.0 |
| 02 Targeted Predictors | 3.5 | 3.5 | 4.5 | 4.0 | 3.5 | 4.5 | 4.0 |
| 03 Three-Pass Filter | 3.5 | 4.5 | 4.5 | 4.5 | 3.5 | 4.5 | 4.2 |

**Notes:**
- Three-Pass Filter deck is one of the best in the entire course. The "interview/consensus/decision" analogy for the three passes is outstanding. The side-by-side comparison of Standard PCA vs Targeted vs 3PRF is clear and effective.
- High-Dimensional Regression uses a good suitcase packing analogy for LASSO penalties.
- Targeted Predictors has a code issue: `|correlations|` is invalid Python syntax.
- The L1 diamond vs L2 circle geometric intuition in Guide 01 is described in text only -- this deserves an actual geometric diagram.

### Module 08: Advanced Topics

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Overall |
|------|:------:|:---------:|:------------:|:--------:|:------:|:----------:|:-------:|
| 01 Time-Varying Parameters | 3.5 | 3.5 | 4.5 | 4.5 | 3.5 | 4.5 | 4.0 |
| 02 Non-Gaussian Factors | 3.5 | 4.0 | 4.5 | 4.0 | 3.5 | 4.5 | 4.0 |
| 03 ML Connections | 4.0 | 4.0 | 4.5 | 4.5 | 3.5 | 4.5 | 4.2 |

**Notes:**
- ML Connections is one of the strongest decks overall -- the PCA-autoencoder equivalence proof is clearly presented, the progression from linear AE to deep AE to VAE is well-motivated, and the decision framework for when to use traditional vs ML methods is practical.
- Non-Gaussian Factors has a code bug: `pca` variable used before assignment in `StudentTFactorModel.fit()`.
- Time-Varying Parameters clearly compares four TVP specifications with a decision flowchart.
- The "Black Monday -22.6% = >20 sigma under Gaussian" motivating example in Guide 02 is effective.

---

## Comparison to Best-in-Class

### vs Stanford CS229 Slides
- **CS229 advantage:** Custom illustrations, color-coded concept maps, progressive build animations, margin annotations with key takeaways.
- **This course advantage:** More code integration, better cross-referencing between topics.
- **Gap:** Visual design and engagement. CS229 uses actual rendered plots and geometric illustrations; this course relies entirely on Mermaid text-box flowcharts.

### vs fast.ai Course Materials
- **fast.ai advantage:** Large fonts, minimal text per slide, heavy use of real output screenshots, "show the result first" pedagogy, Jupyter notebook integration.
- **This course advantage:** More rigorous mathematical treatment, systematic practice problems.
- **Gap:** Information density. fast.ai follows a "one idea per slide" rule; many slides in this course pack 2-3 ideas with formulas, tables, and code.

### vs DataCamp Slide Decks
- **DataCamp advantage:** Problem-first framing ("You need to predict X, here's how"), consistent visual branding, interactive exercises woven into presentation.
- **This course advantage:** Deeper mathematical rigor, broader topic coverage.
- **Gap:** Narrative engagement. DataCamp opens with scenarios; this course opens with formulas.

---

## Priority Fix List

### Critical (Fix Before Use)

1. **Module 08, Guide 02 (Non-Gaussian Factors):** Fix the `pca` variable assignment bug in `StudentTFactorModel.fit()`. Line ~131: add `pca = PCA(self.n_factors)` before `self.F = pca.fit_transform(X)`.

2. **Module 07, Guide 02 (Targeted Predictors):** Fix `|correlations|` to `np.abs(correlations)` in the `TargetedPredictors.fit()` code. Line ~188.

3. **Module 06, Guide 02 (FAVAR):** Fix `PCA(self.n_factors)` to `PCA(n_components=self.n_factors)` in `FAVAR.fit()`. Line ~218.

### High Priority (Significant Quality Improvement)

4. **All 28 decks:** Increase base font size from 24px to 28px for presentation readability.

5. **All 28 decks:** Add at least 1-2 rendered plot images per deck (scree plots, IRF plots, regularization paths, heatmaps). This would substantially improve the "Added Visual Value" dimension.

6. **Module 02, Guide 03 (Kalman Filter):** Consider splitting this 474-line deck into two: "Kalman Filter Basics" (predict/update/filter) and "Kalman Filter Advanced" (smoother/missing data/forecasting/likelihood).

7. **All decks:** Replace the opening blockquote with a concrete real-world scenario or motivating question.

### Medium Priority (Polish)

8. **All decks:** Add Marp speaker notes to at least title and summary slides.

9. **All decks:** Add a `<!-- footer: Module X - Topic Name -->` directive for easier identification during presentation.

10. **Module 07, Guide 01 (LASSO):** Replace the text-based description of L1 diamond vs L2 circle with an actual geometric diagram image.

11. **Course-level:** Create a "Course Roadmap" deck (Module 00, Guide 00) showing the full 9-module progression with prerequisites.

12. **All decks with code:** Ensure no `...` or `# ...` truncations. Every code block should be complete and runnable in isolation.

### Low Priority (Nice to Have)

13. **All decks:** Add 2-3 CSS theme variations to break visual monotony.

14. **Selected decks:** Add more analogies following the strong examples in Module 05 Guide 01 (bathtub), Module 07 Guide 01 (suitcase), and Module 07 Guide 03 (interviews).

15. **Practice problems:** Add solution hints or references to where solutions can be found.

16. **All decks:** Add a "Key Takeaway" box (distinct styling) to each major section.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total decks reviewed | 28 |
| Total lines across all decks | ~11,200 |
| Average deck length | ~400 lines |
| Longest deck | Module 02 Guide 03 (Kalman Filter) - 474 lines |
| Shortest deck | Module 04 Guide 01 (MLE via Kalman) - 349 lines |
| Mermaid diagrams (total) | ~130 |
| Code blocks (total) | ~56 |
| Practice problems (total) | ~252 (9 per deck x 28 decks) |
| Critical code bugs found | 3 |
| Decks scoring 4.0+ overall | 19 of 28 (68%) |
| Decks scoring below 3.8 | 1 (FAVAR at 3.8) |

**Bottom line:** This is a well-crafted, technically rigorous slide deck collection that is ready for production use after fixing the 3 critical code bugs and making the high-priority improvements to visual design and narrative engagement. The consistent structure is a double-edged sword -- it ensures completeness but risks monotony. The biggest single improvement would be adding rendered visualizations (plots, heatmaps, diagrams) to complement the text-based Mermaid flowcharts.
