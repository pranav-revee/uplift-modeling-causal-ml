# uplift-modeling-causal-ml

Senior-level Causal ML / Uplift Modeling portfolio project for marketing campaign decisioning under budget constraints.

## 1. Project motivation
Most marketing models answer: "Who is likely to convert?"  
This project answers a harder and more decision-relevant question: "Who is likely to convert **because of** the campaign?"

## 2. Business problem
**Core question:** Which customers should receive an email campaign to maximize **incremental conversions** and **incremental revenue**, not just conversion propensity?

Primary optimization target:
- **primary outcome:** `conversion`
- **secondary outcomes:** `visit`, `spend`

## 3. Dataset description
Primary dataset: Kevin Hillstrom's MineThatData Email Marketing dataset (about 64k customers).  
Canonical location in this repo: `data/hillstrom.csv`.

Original campaign setup is typically 3-arm:
- `No E-Mail`
- `Mens E-Mail`
- `Womens E-Mail`

This project uses a deliberate modeling simplification for core uplift workflows:
- **treated = any email** (`Mens E-Mail` or `Womens E-Mail`)
- **control = no email** (`No E-Mail`)

This simplification is explicit and intentional, not an oversight.

## 4. Why uplift modeling instead of response modeling
Response models rank likely buyers, including many "sure things" who would convert anyway.  
Uplift models rank expected **incremental impact**, which is closer to policy optimization.

Feature governance:
- all model training uses **pre-treatment covariates only**
- outcomes and treatment-derived labels are excluded from feature construction to avoid leakage

## 5. Methods overview
Layer 1 (classical uplift foundation):
- Naive treated-response model
- Two-model uplift baseline
- S-Learner, T-Learner, X-Learner

Layer 2 (modern causal ML + decisioning):
- DR-Learner / doubly robust CATE estimation
- Optional orthogonal/causal forest validation
- Policy simulation under budget constraints

Advanced extension:
- Optional TARNet deep uplift notebook

## 6. Evaluation framework
Core metrics and policy diagnostics:
- Uplift curve
- Qini coefficient
- AUUC
- Cumulative gain
- Uplift by decile
- Policy value at budget
- Incremental conversions / revenue at budget
- Budget sensitivity analysis

Important caveat:
- **Qini and AUUC are ranking metrics, not proof that every individual treatment effect is perfectly estimated.**

## 7. Notebook roadmap
1. `notebooks/01_eda_problem_framing.ipynb`
2. `notebooks/02_naive_and_two_model_baselines.ipynb`
3. `notebooks/03_meta_learners_s_t_x.ipynb`
4. `notebooks/04_modern_causal_ml_dr_learner.ipynb`
5. `notebooks/05_policy_evaluation_and_business_impact.ipynb`
6. `notebooks/06_optional_tarnet_deep_uplift.ipynb`

## 8. Project structure
```text
uplift-modeling-causal-ml/
├── data/
│   └── hillstrom.csv
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── .gitkeep
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. How to run
1. Use Python 3.10+.
2. On macOS, prefer native arm64 Python (Apple Silicon) instead of Rosetta.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Start Jupyter:
   - `jupyter lab` or `jupyter notebook`
5. Run notebooks in order.

Execution defaults for this repo:
- Global random seed: `42`
- Baseline tabular learner preference: `XGBoost` CPU with `tree_method="hist"`
- LightGBM available as alternative
- For deep notebook: use `torch` MPS when available, otherwise CPU fallback

## 10. Key results
Current offline run snapshot (shared split, seed 42):
- Notebook 02: `two_model_uplift` outperformed naive treated-response on Qini/AUUC.
- Notebook 03: `T-Learner` performed best among S/T/X on the current split.
- Notebook 05: uplift-aware targeting policies improved estimated incremental value versus naive and random policies.

These are offline holdout estimates and should be treated as directional until online validation.

## 11. Future work
- Doubly robust estimation improvements
- Double machine learning (DML)
- Orthogonal / causal forests
- Policy learning
- Multi-treatment uplift (restore full 3-arm treatment design)
- Continuous treatment / dosage optimization
- Calibration of treatment effects
- Fairness / responsible targeting
- Dynamic uplift over time

---

## Offline policy assumptions (used in Notebook 05)
- Model ranking is based on predicted uplift/CATE scores.
- Incremental conversions are estimated from observed treatment/control outcomes within ranked buckets.
- Incremental revenue is approximated from incremental conversions with average order value or observed spend proxies.
- These are **offline policy evaluation estimates**, not causal proof of production revenue lift.
