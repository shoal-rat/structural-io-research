---
name: structural-io-research
description: Methodological checklist and reasoning framework for structural IO demand estimation research, distilled from thesis advisor feedback (Prof. Louis Pape, Telecom Paris). Use this skill whenever working on structural demand models, nested logit estimation, GMM, instrumental variables, natural experiments, or platform economics research. Triggers on keywords like demand estimation, nested logit, Berry, BLP, GMM, IV, RDD, DiD, nesting structure, search costs, Monte Carlo, Hansen J.
user-invocable: true
allowed-tools: Read Grep Bash Agent Write Edit
argument-hint: [topic or question]
---

# Structural IO Research Methodology Guide

**Source:** Synthesized from all thesis supervision feedback by Prof. Louis Pape (Telecom Paris / CREST) during the "Visibility, Search Costs, and Demand Concentration in AI Model Marketplaces" project (March-April 2026). Each principle below is grounded in a specific methodological point raised during supervision.

---

## CORE PRINCIPLES

### 1. Monte Carlo Before Estimation

**Rule:** Before trusting ANY estimation result, run a Monte Carlo study to verify the estimation strategy recovers known parameters.

**Why (Prof. Pape, April 1, 2026):** "If you use non linear GMM, you need to deal with non convexity of the criterion function in some way. Try running a Monte Carlo study to check if your estimation strategy is plausible? I recommend you ignore the Hansen test."

**How to implement:**
1. Fix true parameters at values close to your real estimates
2. Simulate data from the exact DGP you assume (e.g., nested logit with search costs)
3. Include the same endogeneity structure (e.g., correlated trending scores and demand shocks)
4. Estimate the model using the exact same procedure (2SLS, GMM, etc.)
5. Repeat 100-200 times
6. Report: bias, RMSE, 95% CI coverage rate, Hansen J rejection rate

**What to check:**
- If bias is small and CI coverage is near 95%, the estimator works
- If Hansen J rejection rate is near 5% under the true model, the test has correct size
- If RMSE is large relative to the estimate, you have a precision problem (not a bias problem)
- If coverage is far below 95%, the standard errors are wrong

**Red flag:** If the estimator cannot recover known parameters in simulation, it cannot be trusted on real data. Do NOT proceed to empirical results without this check.

---

### 2. Do Not Let the Hansen Test Dictate Model Choice

**Rule:** The Hansen J test is informative but should NOT be the sole criterion for selecting among specifications. It can mislead, especially with nonlinear GMM.

**Why (Prof. Pape, April 1, 2026):** "Please keep in mind that the Hansen test does not dictate model choice."

**The problem with over-relying on Hansen J:**
- In nonlinear GMM, the criterion function may be non-convex. A "passing" J test may reflect a local minimum, not correct specification.
- With very large samples (millions of observations), the J test has enormous power and will reject even minor, economically irrelevant misspecification.
- A marginal p-value (e.g., 0.070) in a 3-million-observation sample is actually reassuring -- it means misspecification is tiny.

**What to do instead:**
- Use the Monte Carlo to verify the estimator works under the true model
- Compare specifications on economic grounds: does the nesting structure make economic sense?
- Check the placebo/lead tests (do future shifters predict current shocks? they shouldn't)
- Compare out-of-sample prediction
- Check if parameter estimates are stable across reasonable specification changes
- Report ALL specifications transparently, not just the one that "passes"

---

### 3. Every Empirical Exercise Must Have a Clear Economic Question

**Rule:** Before running any test, RDD, DiD, or regression, articulate in one sentence what economic question it answers and why a reader should care.

**Why (Prof. Pape, April 1, 2026):** "I don't understand what your natural experiment is supposed to teach us."

**Framework for motivating an empirical exercise:**
1. **What is the question?** (e.g., "Does platform-assigned visibility CAUSE downloads, or just correlate with quality?")
2. **Why can't existing results answer it?** (e.g., "The structural model shows correlation but cannot rule out omitted variable bias")
3. **What is the identifying variation?** (e.g., "The profile-listing reform changed visibility distribution without changing model quality")
4. **What would we learn from the result?** (e.g., "If downloads change after the reform, visibility has a causal effect beyond quality")
5. **What would we learn from a null result?** (e.g., "If no effect, the trending-download correlation may reflect quality, not visibility per se")

**Common mistake:** Running a test because you can, then trying to justify it post hoc. The economic motivation must come FIRST.

---

### 4. Always Show the Plots

**Rule:** Every empirical claim should be accompanied by a visual that a reader can inspect. Especially for RDD and event studies.

**Why (Prof. Pape, April 1-2, 2026):** "Can you send me some RDD plots? Perhaps run these tests at the pipeline level?" and "I look forward to the plots."

**RDD plots must show:**
- Binned scatter plot: average outcome by running variable (rank), with dots on both sides of the cutoff
- Fitted lines on each side of the cutoff (local linear or polynomial)
- Clear marking of the cutoff
- The visual discontinuity (or lack thereof) should be obvious without reading the coefficient

**Event study plots must show:**
- Coefficients by time bin relative to the event
- 95% confidence intervals
- A clear pre-period (should be flat/zero) and post-period
- The "parallel trends" assumption should be visually verifiable

**Rule of thumb:** If you cannot see the effect in the plot, the effect is either not there or not robust.

---

### 5. Test at the Right Level of Aggregation

**Rule:** When you have a test that works at one level (e.g., one product category), run it at broader levels to check generalizability.

**Why (Prof. Pape, April 1, 2026):** "Perhaps run these tests at the pipeline level?"

**How to implement:**
- If your main analysis is within text-to-image models, also run the same RDD/DiD for text-generation, image-classification, etc.
- Report results across all categories, not just the one where the result is strongest
- If the pattern is consistent across categories, the finding is more credible
- If it's inconsistent, discuss why (different user behavior, different market structure, etc.)

---

### 6. Nesting Structure Should Be Economically Motivated

**Rule:** The choice of nesting structure in a nested logit should reflect actual substitution patterns, not convenience.

**Why (Prof. Pape, March 13, 2026):** Sent the Almagro, Lai & Manresa (2025) "Data-Driven Nests" paper, asking the student to consider data-driven alternatives to ad-hoc license nesting.

**Checklist for nesting decisions:**
1. **Economic question:** When a user stops using model A, what model do they switch to? Models that are close substitutes should be in the same nest.
2. **Existing literature:** Does prior work suggest a natural grouping? (e.g., architecture family, quality tier, use case)
3. **Data-driven approach (Almagro et al. 2025):** Use k-means on demand curves to let the data reveal substitution groups. Two-step: (a) cluster products by demand response patterns, (b) estimate structural model with discovered nests.
4. **Sensitivity check:** Estimate under multiple nesting structures and report all results. If sigma is small (close to 0), the nesting choice doesn't matter much (close to logit).
5. **Validation:** Use an exogenous shock to test whether within-nest substitution is stronger than cross-nest substitution (like the Bud Light boycott in Almagro et al.).

**Key insight from our experience:** When sigma is small (e.g., 0.046), the model is close to logit regardless of nests. In this case, the nesting choice is not first-order important, but you should still report the comparison transparently.

---

### 7. Identification vs. Estimation

**Rule:** Clearly distinguish between what identifies the parameters (economic argument) and how you estimate them (statistical procedure). Referees care more about the former.

**Why (implicit in Prof. Pape's questioning of the natural experiment's contribution):** The structural model identifies beta_vis through instruments (count of models in/outside nest). The natural experiment identifies the causal effect of visibility through a reform shock. These answer DIFFERENT questions. Do not conflate them.

**Structure for presenting identification:**
1. **What parameter are you trying to estimate?** (beta_vis, sigma, phi, causal effect)
2. **What is the source of identifying variation?** (instruments, reform shock, page boundary)
3. **What assumption is needed?** (instrument exogeneity, parallel trends, continuity at cutoff)
4. **How do you test the assumption?** (first-stage F, placebo tests, pre-trend check)

---

### 8. Search Costs: Be Honest About Weak Identification

**Rule:** If some parameters are weakly identified (e.g., search-cost shifters hitting boundaries), say so clearly and show that the core results don't depend on them.

**Derived from the supervision process:** phi_2 (outside-nest growth shifter) hits the boundary in 64% of bootstrap draws. The correct response is:
- Report this honestly
- Show that beta_vis and sigma are robust to the search-cost specification
- Present the no-search specification as a robustness check
- Explain that the search block is supplementary, not the main contribution

---

### 9. Bootstrap for Nonlinear Models

**Rule:** Standard errors from nonlinear GMM may be unreliable. Use bootstrap (market-level resampling) for inference.

**How to implement:**
- Resample market-weeks (not individual observations) with replacement
- Re-estimate the full model for each draw
- Report bootstrap mean, SE, and percentile-based confidence intervals
- 200 draws is a good minimum; 500+ if computationally feasible
- Check if the point estimate falls within the bootstrap distribution (if not, there may be a convergence issue)

---

### 10. Foundation Models and Sample Composition

**Rule:** Be thoughtful about whether to include all models or restrict to subsets. The choice affects interpretation.

**Derived from supervision:** Foundation/root-lineage models capture 79% of downloads despite being 32% of models. Including all models means the structural estimates are driven by the tail. Restricting to downstream-only changes the market definition. Report both and discuss.

---

## WORKFLOW CHECKLIST

When starting or reviewing a structural IO demand estimation project:

- [ ] **Monte Carlo:** Has the estimation strategy been validated on simulated data?
- [ ] **Economic motivation:** Can you state in one sentence why each empirical exercise matters?
- [ ] **Nesting:** Is the nesting structure economically motivated? Have alternatives been tested?
- [ ] **Instruments:** Are they strong (F >> 10)? Are they plausibly exogenous?
- [ ] **Plots:** Does every RDD/event study have a visual? Can you see the effect?
- [ ] **Cross-level:** Has the analysis been run at multiple levels of aggregation?
- [ ] **Robustness:** Are weakly-identified parameters flagged? Are results robust to their exclusion?
- [ ] **Bootstrap:** Are standard errors from bootstrap, not just asymptotic formulas?
- [ ] **Transparency:** Are ALL specifications reported, not just the best-looking one?
- [ ] **Natural experiment:** Is the causal claim clearly separated from the structural estimation?

---

## REFERENCES

Key papers referenced in supervision:
- Berry (1994): Discrete choice demand estimation
- Berry, Levinsohn & Pakes (1995): Random coefficients logit
- Cardell (1997): Nested logit variance components
- Almagro, Lai & Manresa (2025): Data-driven nests
- Fosgerau, Monardo & De Palma (2024): Misspecification bias from wrong nests
- Moraga-González, Sandor & Wildenbeest (2023): Consumer search in automobiles
- Dinerstein, Einav, Levin & Sundaresan (2018): Platform design and consumer search
- Hansen (1982): GMM
- Stock & Yogo (2005): Weak instruments testing
