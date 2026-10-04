# OOP with Python: Assignment 1, Advent 2026 (Mini-Projects)

**Student:** _<your name>_ · **Reg. no.:** _<your registration number>_ · **Programme:** MSCS / MSDS

Five mini-projects set in Ugandan contexts. Each one is modelled with classes, backed by reusable code in `src/`, run from its own Jupyter notebook and covered by `pytest` tests.

| # | Mini-project | Notebook | Module |
|---|---|---|---|
| 1 | UBOS District Population Forecaster | `notebooks/project1_population.ipynb` | `src/population.py` |
| 2 | Solar Micro-Grid Dispatch Planner (Kasese) | `notebooks/project2_microgrid.ipynb` | `src/microgrid.py` |
| 3 | Lake Victoria Fish Stock & Export Risk Model | `notebooks/project3_fishery.ipynb` | `src/fishery.py` |
| 4 | Rainfall Pattern & Crop Suitability Analyser | `notebooks/project4_rainfall.ipynb` | `src/rainfall.py` |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | `notebooks/project5_taxi.ipynb` | `src/taxi.py` |

## Repository layout

```
├── README.md
├── requirements.txt
├── src/
│   ├── common/
│   │   ├── forecasting.py   # abstract Forecaster base class + MAE/RMSE/MAPE (shared by P1 & P5)
│   │   └── nbsetup.py       # plotting style and paths used by the notebooks
│   ├── population.py        # P1
│   ├── microgrid.py         # P2
│   ├── fishery.py           # P3
│   ├── rainfall.py          # P4
│   └── taxi.py              # P5
├── notebooks/               # one notebook per mini-project (committed with outputs)
├── tests/                   # pytest suites, one file per mini-project (79 tests)
├── data/                    # generated demand CSV (P2); optional NASA POWER files (P4)
├── figures/                 # PNGs saved by the notebooks
└── tools/
    ├── build_notebooks.py   # rebuilds + executes notebooks from tools/nb_src/
    └── nb_src/              # plain-text (percent-format) sources of the notebooks
```

### Design highlights (OOP)
* **One forecasting hierarchy, used by two projects.** `Forecaster` (an ABC) defines `fit()` / `predict(horizon)`, validates input and handles state. Subclasses only implement `_fit` and `_predict`. P1 adds the linear-trend, CAGR and Fibonacci-ratio models. P5 adds the moving-average, SES and seasonal-naïve models and **inherits** P1's linear trend instead of copying it.
* **Inheritance where it helps.** `HybridMicroGrid(MicroGrid)` reuses all of the solving, feasibility and conditioning logic for a 3×3 system.
* **Composition.** `FisheryScenario` combines `FishStock`, `PriceModel` and `RiskAssessor`. `SuitabilityAnalyser` combines `Region` and `CropRule` objects.
* **Dunder methods:** `__repr__` and `__len__` on the domain classes. Frozen `dataclass`es are used for value objects (`ClassroomPlanner`, `CostModel`, `CropRule`, `RiskAssessor`, `FleetPlanner`, `MarketModel`).
* **Validation:** every constructor rejects empty, negative, non-finite or mismatched input with a `ValueError`.
* **Type hints and docstrings** on all public classes and methods.

## Setup

```bash
git clone <this-repo-url>
cd advent2026-oop
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11, NumPy 2.4, SciPy 1.17, Matplotlib 3.10 and pandas 3.0. It should work with Python ≥ 3.10.

## Running

```bash
# Tests (from the repository root)
pytest -q

# Notebooks: open in Jupyter and use Kernel ▸ Restart & Run All
jupyter notebook notebooks/

# ...or execute all of them headlessly
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

The notebooks find `src/` whether Jupyter is started from the repository root or from `notebooks/`. `python tools/build_notebooks.py` regenerates the notebooks from their plain-text sources in `tools/nb_src/` and executes them. This keeps code reviews and diffs readable.

**Reproducibility:** all randomness goes through `np.random.default_rng(seed)`, with the seed fixed at the top of each notebook or passed into each class. The P2 demand CSV is regenerated from its seed every time the notebook runs.

## Summary of findings

**P1: Population.** Every district grows roughly exponentially. On a 2015–21 train / 2022–24 test split, the CAGR model beat the linear trend everywhere (MAPE 0.1–2%), and the Fibonacci-ratio model failed badly (MAPE ≈ 80%) because its ratios converge to φ ≈ 1.618, which implies about 62% annual growth. Wakiso grows fastest (6.5% a year). By 2029 it needs about **2,090 more classrooms**, against about 1,550 in Kampala, 710 in Mukono, 410 in Gulu and 310 in Mbarara. The upper 95% bootstrap bound adds 1–15%. `statistics.variance` divides by n−1 and `np.var` divides by n; `ddof=1` makes them match.

**P2: Micro-grid.** det(A) = −5 and κ ≈ 5.8, so the system is well-posed and well-conditioned. The vectorised solve is about 30× faster than the loop, mostly from per-call overhead. 2 of 30 synthetic days were infeasible (D2 > 4/3·D1) and were repaired with NNLS, with the shortfall reported. The battery is far more volatile relative to its size (CV 0.64 against 0.14 for solar) and drives cost: about UGX 9,300 a day, or roughly UGX 280,000 a month. The extension adds a diesel 3×3 system, shows what happens when a row is dependent, and runs a Monte Carlo sensitivity whose error amplification stays below κ.

**P3: Fishery.** The logistic model replaces the unbounded Fibonacci model. h = 0.20 = r/2 achieves MSY (1,000 t/week, stock at K/2). The current h = 0.10 is safe but lands 25% below MSY, and h = 0.30 catches less *and* leaves a thinner stock. The old "variance > 50,000" rule is meaningless because variance is in UGX², so risk is classified by CV instead. The 5% VaR of annual revenue is about 17% of expected revenue. A closed season helps only when the fishery is over-exploited (+7% at h = 0.3, +41% at h = 0.35) and costs revenue at h ≤ 0.2.

**P4: Rainfall.** Cosine similarity has been implemented correctly and verified against SciPy. It still rates every pair of regions as "similar" (0.75–0.89) because it ignores magnitude and all rainfall vectors are positive. Pearson correlation shows that the seasonal timing is essentially unrelated. The find_peaks method, with circular padding and a 30% relative-prominence rule, classifies Gulu as unimodal and Mbarara as bimodal, matching known climatology. Kampala comes out unimodal because the illustrative series lacks a short-rains dip, which is discussed in the notebook. The crop thresholds come from FAO and UCDA, and there is an advisory note for Mbarara farmers.

**P5: Taxis.** The Ntinda equilibrium is P* = UGX 2,200 and Q* = 76, so the current fare of UGX 2,000 creates a shortage of about 10 passengers per trip-hour. In walk-forward backtests, SES with a tuned α (= 1, i.e. naïve) beats the 3-day moving average on every route. The fleet size depends on what the counts measure: 1 vehicle per route if they are daily totals, or 6–8 if they are hourly stage counts. On 60 simulated days with a weekly pattern, a seasonal-naïve model cuts MAE by about 70% compared with the moving average.

## Data sources and assumptions
* The population, rainfall, route and demand data are **illustrative**, as stated in the brief. The Mbarara and Mukono population series in P1 and all of P2's demand data were generated by me.
* Crop water needs: Brouwer, C. & Heibloem, M. (1986) *Irrigation Water Management Training Manual No. 3: Irrigation Water Needs*, FAO, Tables 4–5. <https://www.fao.org/4/s2022e/s2022e02.htm>
* Robusta coffee rainfall: Uganda Coffee Development Authority (2019) *Robusta Coffee Handbook*.
* **Optional P4 extension (real data).** Download monthly precipitation from the [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/): Community *Agroclimatology*, Temporal *Monthly & Annual*, parameter *Precipitation Corrected (PRECTOTCORR)*, 2010–2023, format CSV. Suggested points: Kampala (0.35, 32.58), Gulu (2.78, 32.30), Mbarara (−0.61, 30.65). Save the files as `data/nasa_power_Kampala.csv` and so on, then re-run notebook 4. It will draw monthly box plots.

## AI-use declaration
_Edit this section so it describes accurately how **you** used AI._

This repository was developed with the help of an AI assistant (Anthropic's Claude). It was used to scaffold the repository structure, draft the class designs, implementations, tests and notebook text, and check the results. I have reviewed, run and tested all the code (`pytest` passes and every notebook runs with Restart & Run All). I can explain each method, formula and design decision. I checked the key results against hand calculations, which appear as `assert` statements in the notebooks.
