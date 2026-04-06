# Structural IO Research Methodology Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Code Skill](https://img.shields.io/badge/Claude_Code-Skill-blueviolet)](https://claude.ai/code)

A comprehensive, practitioner-oriented methodology guide for **structural Industrial Organization (IO) demand estimation research**. Built as a [Claude Code](https://claude.ai/code) skill that provides real-time methodological guidance during research.

Distilled from two sources:
1. **Thesis supervision feedback** by Prof. Louis Pape (Telecom Paris / CREST) during a structural demand estimation project on AI model marketplaces
2. **Extensive survey of 60+ academic papers** spanning demand estimation, causal inference, and empirical best practices

---

## What This Guide Covers

| Phase | Topics | Key References |
|-------|--------|----------------|
| **Estimation** | BLP random coefficients, Nested logit pitfalls, Nonlinear GMM, Instrument choice | Conlon & Gortmaker (2020), Gandhi & Houde (2020), Borusyak et al. (2025) |
| **Validation** | Monte Carlo studies, Weak instrument diagnostics, Bootstrap inference, Pre-testing bias | Stock & Yogo (2005), Andrews et al. (2019), Advani et al. (2019) |
| **Identification** | Berry-Haile theory, Market definition, Outside option, Nesting structure choice | Berry & Haile (2014), Bugni & Ura (2025), Almagro et al. (2025) |
| **Causal Inference** | Difference-in-Differences, Regression Discontinuity Design | Baker et al. (2025), Roth et al. (2023), Cattaneo et al. (2024) |
| **Policy Analysis** | Counterfactual simulation, Merger analysis, Consumer search models | Head & Mayer (2026), Ursu (2018), Moraga-Gonzalez et al. (2023) |
| **Research Craft** | Writing IO papers, Referee expectations, Transparency & replication | Bellemare (2020), Christensen & Miguel (2018) |

---

## 20 Sections

1. **BLP/Random Coefficients Demand Estimation Best Practices** — Tight contraction tolerance (1e-14), Halton draws, optimal instruments, multiple starting values
2. **Nested Logit Estimation Pitfalls** — Sigma bounds, within-group share endogeneity, RU1 normalization
3. **GMM Estimation for Nonlinear Models** — Two-step procedure, analytic gradients, non-convexity diagnosis
4. **Instrumental Variables in Demand Estimation** — From BLP instruments to Gandhi-Houde differentiation IVs to Borusyak recentered IVs
5. **Monte Carlo Validation** — 200-500 reps, realistic DGP, report bias/RMSE/coverage
6. **Weak Instruments in Demand Estimation** — Stock-Yogo, Anderson-Rubin, tF procedure, Olea-Pflueger effective F
7. **Identification in Differentiated Products Markets** — Berry-Haile nonparametric results, inversion theorem
8. **How to Write an Empirical IO Paper** — Structure, frontloading contribution, magnitudes over stars
9. **Consumer Search Models Estimation** — Sequential vs. simultaneous, position bias, exclusion restrictions
10. **Difference-in-Differences Best Practices** — TWFE problems, Callaway-Sant'Anna, staggered treatment, pre-trend testing
11. **Regression Discontinuity Design Best Practices** — MSE-optimal bandwidth, manipulation testing, no global polynomials
12. **Market Definition in Demand Estimation** — SSNIP test, sensitivity analysis, avoiding circular definitions
13. **Outside Option and Market Size** — Bugni & Ura (2025) critique, partial identification, sensitivity testing
14. **Counterfactual Analysis and Merger Simulation** — Equilibrium computation, pass-through, welfare bounds
15. **Nesting Structure Choice** — Data-driven nests (Almagro et al. 2025), misspecification bias (Fosgerau et al. 2024), validation
16. **Bootstrap Inference for GMM** — Cluster-level resampling, 200+ draws, percentile CIs
17. **Pre-Testing Bias and Specification Search** — Leamer critique, Armstrong et al. (2025), report all specifications
18. **Replication, Transparency, and Credibility** — Code sharing, pre-registration, Christensen & Miguel (2018)
19. **What Referees Look For in IO Papers** — Common fatal flaws, referee report structure
20. **Workflow Checklists** — Five-phase checklist from pre-estimation through submission

---

## Installation

### As a Claude Code Skill (recommended)

**Global installation** (available in all projects):
```bash
mkdir -p ~/.claude/skills/structural-io-research
cp SKILL.md ~/.claude/skills/structural-io-research/SKILL.md
```

**Project-level installation** (available in one project):
```bash
mkdir -p .claude/skills/structural-io-research
cp SKILL.md .claude/skills/structural-io-research/SKILL.md
```

### Usage in Claude Code

The skill activates automatically when you discuss IO research topics, or invoke it explicitly:

```
/structural-io-research
/structural-io-research "how to choose instruments for nested logit"
/structural-io-research "Monte Carlo design for BLP"
```

### As a standalone reference

The `SKILL.md` file is a self-contained Markdown document that can be read in any Markdown viewer, IDE, or browser. No dependencies required.

---

## Quick Start: The Workflow Checklist

Before submitting any structural IO paper, verify:

- [ ] **Monte Carlo:** Has the estimation strategy been validated on simulated data?
- [ ] **Economic motivation:** Can you state in one sentence why each empirical exercise matters?
- [ ] **Nesting:** Is the nesting structure economically motivated? Have alternatives been tested?
- [ ] **Instruments:** Are they strong (F >> 10)? Are they plausibly exogenous? Are differentiation IVs used?
- [ ] **Plots:** Does every RDD/event study have a visual? Can you see the effect?
- [ ] **Cross-level:** Has the analysis been run at multiple levels of aggregation?
- [ ] **Robustness:** Are weakly-identified parameters flagged? Are results robust to their exclusion?
- [ ] **Bootstrap:** Are standard errors from bootstrap, not just asymptotic formulas?
- [ ] **Transparency:** Are ALL specifications reported, not just the best-looking one?
- [ ] **Natural experiment:** Is the causal claim clearly separated from the structural estimation?
- [ ] **Outside option:** Is the market size assumption tested for sensitivity?
- [ ] **Counterfactuals:** Do welfare calculations account for the equilibrium response?

---

## Key Principles (from thesis supervision)

> **"If you use nonlinear GMM, you need to deal with non-convexity of the criterion function in some way. Try running a Monte Carlo study to check if your estimation strategy is plausible."** — Prof. Pape

> **"Please keep in mind that the Hansen test does not dictate model choice."** — Prof. Pape

> **"I don't understand what your natural experiment is supposed to teach us."** — Prof. Pape (on the importance of articulating the economic question before the statistical exercise)

> **"Can you send me some RDD plots? Perhaps run these tests at the pipeline level?"** — Prof. Pape (on always showing visual evidence and testing at multiple aggregation levels)

---

## References (selected, 60+ in full guide)

### Demand Estimation
- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*
- Berry, S., Levinsohn, J. & Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*
- Berry, S. & Haile, P. (2014). "Identification in Differentiated Products Markets." *Econometrica*
- Conlon, C. & Gortmaker, J. (2020). "Best Practices for BLP Demand Estimation with PyBLP." *RAND Journal of Economics*
- Nevo, A. (2000). "A Practitioner's Guide to RC Logit." *JEMS*

### Instruments
- Gandhi, A. & Houde, J.-F. (2020). "Measuring Substitution Patterns." *NBER WP 26375*
- Borusyak, K., Hull, P. & Jaravel, X. (2025). "Estimating Demand with Recentered Instruments." *arXiv*
- Stock, J. & Yogo, M. (2005). "Testing for Weak Instruments." Cambridge UP

### Nesting
- Almagro, M., Lai, K. & Manresa, E. (2025). "Data-Driven Nests." *Working Paper*
- Fosgerau, M., Monardo, J. & De Palma, A. (2024). "Misspecification of Nests." *Working Paper*
- Cardell, N.S. (1997). "Variance Components for Extreme-Value Distributions." *Econometric Theory*

### Causal Inference
- Baker, A., Callaway, B. et al. (2025). "A Practitioner's Guide to DiD." *arXiv*
- Roth, J. et al. (2023). "What's Trending in DiD." *Journal of Econometrics*
- Cattaneo, M., Idrobo, N. & Titiunik, R. (2024). *A Practical Introduction to RDD*. Cambridge UP

### Research Practice
- Bellemare, M. (2020). "How to Write Applied Papers in Economics"
- Christensen, G. & Miguel, E. (2018). "Transparency, Reproducibility, and Credibility." *JEL*

---

## Contributing

Contributions are welcome. If you have methodological advice from your own supervision, referee reports, or published best-practices papers, please open a PR or issue.

Areas where contributions would be especially valuable:
- Dynamic discrete choice estimation best practices
- Auction estimation methodology
- Entry/exit models
- Vertical relationships and bargaining models
- Machine learning methods in IO (demand estimation with forests, neural nets)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you find this guide useful in your research, please cite:

```bibtex
@misc{structural-io-research-skill,
  title={Structural IO Research Methodology Guide},
  author={Zhang, Weike},
  year={2026},
  url={https://github.com/shoal-rat/structural-io-research},
  note={Claude Code skill for structural demand estimation research methodology}
}
```
