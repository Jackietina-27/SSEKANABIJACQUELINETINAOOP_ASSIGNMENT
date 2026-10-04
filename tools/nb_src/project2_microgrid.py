# %% [markdown]
# # Mini-Project 2: Solar Micro-Grid Dispatch Planner
#
# **Question from the Kasese health centre:** how much solar ($x$) and battery ($y$) energy should be drawn each day, and what will it cost?
#
# $$3x + 2y = D_1 \quad(\text{daytime load}),\qquad 4x + y = D_2 \quad(\text{critical-equipment load})$$
#
# **Code layout:** the classes live in `src/microgrid.py`: `MicroGrid`, `HybridMicroGrid`, `DemandInput` and `CostModel`.

# %%
import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import statistics
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

from src.common.nbsetup import setup_plotting, DATA_DIR, FIG_DIR
from src.microgrid import (MicroGrid, HybridMicroGrid, DemandInput, CostModel,
                           monte_carlo_sensitivity)

setup_plotting()
SEED = 42
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Task 1: The `MicroGrid` class and a well-posedness check

# %%
grid = MicroGrid()
print(grid)
print("A =\n", grid.A)
print(f"det(A)  = {grid.determinant():.4f}   (hand: 3*1 - 2*4 = -5)")
print(f"cond(A) = {grid.condition_number():.4f}")
print("singular values:", np.linalg.svd(grid.A, compute_uv=False).round(4))

# %% [markdown]
# **What the numbers mean.**
# * **Determinant = −5, which is non-zero.** The two constraint lines are not parallel, so every pair $(D_1, D_2)$ has **exactly one** solution. A zero determinant would mean either no solution or infinitely many. The sign only shows that the rows are ordered in a "left-handed" way and has no physical meaning.
# * **Condition number ≈ 5.8.** This is the ratio of the largest to the smallest singular value. It bounds how much a relative error in the demands can be amplified in the solution: $\frac{\|\delta s\|}{\|s\|}\le\kappa\frac{\|\delta d\|}{\|d\|}$. So a 1% measurement error in demand can cause at most about a 6% error in the dispatch. A value near 1 is ideal and above about $10^8$ is numerically dangerous, so this system is **well-conditioned**.
#
# **Hand solution, used as a check:** $x = (2D_2 - D_1)/5$ and $y = (4D_1 - 3D_2)/5$. Both are non-negative only when $D_1/2 \le D_2 \le 4D_1/3$. This feasibility band matters in Task 4.

# %%
x, y = grid.solve_day(130, 160)
print(f"solve_day(130, 160) -> solar = {x:.2f} kWh, battery = {y:.2f} kWh")
assert np.isclose(x, (2 * 160 - 130) / 5) and np.isclose(y, (4 * 130 - 3 * 160) / 5)

# %% [markdown]
# ## Task 2: Two input modes
# ### (a) Interactive `input()` with validation
# `DemandInput.prompt_demand` loops until it gets a valid value. It rejects empty strings, non-numbers, negatives, `nan` and `inf`, and prompts again each time. The input function can be swapped in, so the cell below replays a scripted sequence of bad and good answers and **Restart & Run All** doesn't wait for a keyboard. Set `INTERACTIVE = True` to type the values yourself.

# %%
INTERACTIVE = False

if INTERACTIVE:
    d1, d2 = DemandInput.interactive()
else:
    scripted = iter(["", "abc", "-20", "nan", "128.5",     # D1: four bad attempts then OK
                     "1e400", "150"])                      # D2: inf rejected, then OK
    def fake_input(prompt: str) -> str:
        answer = next(scripted)
        print(f"{prompt}{answer}")
        return answer
    d1, d2 = DemandInput.interactive(input_fn=fake_input)

print(f"\nAccepted D1 = {d1}, D2 = {d2} -> dispatch {grid.solve_day(d1, d2).round(2)} kWh")

# %% [markdown]
# ### (b) 30 days of demand from a CSV
# I generate the CSV with a seeded random generator, using these assumptions:
# * **D1 (daytime load)** is about 135 kWh on weekdays (outpatient clinics and lab work) and about 110 kWh at weekends, with noise (σ = 6).
# * **D2 (critical equipment: vaccine fridges, oxygen and theatre)** is about 1.2 × D1, plus 10 kWh on Tuesday and Thursday theatre days, plus its own noise (σ = 10).

# %%
csv_path = DATA_DIR / "demand_30days.csv"
DemandInput.generate_csv(csv_path, days=30, seed=SEED)
demand = DemandInput.load_csv(csv_path)
display(demand.head(8))
D = demand[["D1", "D2"]].to_numpy().T      # shape (2, 30)
print("Right-hand side shape:", D.shape)

# %% [markdown]
# ## Task 3: Solve all 30 days, first in a loop and then vectorised

# %%
S_loop = grid.solve_loop(D)
S_vec = grid.solve_many(D)
assert np.allclose(S_loop, S_vec)
assert np.allclose(grid.A @ S_vec, D)          # second-method check: A s reproduces d

n_rep = 200
t_loop = min(timeit.repeat(lambda: [linalg.solve(grid.A, D[:, j]) for j in range(D.shape[1])],
                           number=n_rep, repeat=5)) / n_rep
t_vec = min(timeit.repeat(lambda: linalg.solve(grid.A, D), number=n_rep, repeat=5)) / n_rep
print(f"Python loop : {t_loop*1e6:8.1f} µs per 30-day solve")
print(f"Vectorised  : {t_vec*1e6:8.1f} µs per 30-day solve")
print(f"Speed-up    : {t_loop / t_vec:.1f}x")

# %% [markdown]
# **Comment.** The vectorised call is roughly as many times faster as there are days. The arithmetic is trivial for a 2×2 system, so almost all the time goes on **per-call overhead**: SciPy input checks, array conversion and the LAPACK dispatch. The loop pays that cost 30 times and the vectorised call pays it once. A single call also factorises $A$ **once** (LU decomposition) and reuses the factors for all 30 right-hand sides. For 30 days either approach is fast enough. For a year of hourly data (8,760 columns) the difference would matter.

# %% [markdown]
# ## Task 4: Detect and handle infeasible days

# %%
raw = S_vec
bad = grid.infeasible_mask(raw)
print(f"Infeasible days: {bad.sum()} of {bad.size} -> {demand.loc[bad, 'date'].tolist()}")

nnls_plan = grid.dispatch(D, strategy="nnls")
clip_plan = grid.dispatch(D, strategy="clip")
compare = pd.concat({"NNLS": nnls_plan.loc[bad, ["solar", "battery", "gap_D1", "gap_D2"]],
                     "Clip": clip_plan.loc[bad, ["solar", "battery", "gap_D1", "gap_D2"]]}, axis=1)
compare.insert(0, ("raw", "battery"), raw[1, bad])
compare.insert(0, ("raw", "solar"), raw[0, bad])
compare.insert(0, ("demand", "D2"), D[1, bad])
compare.insert(0, ("demand", "D1"), D[0, bad])
compare.index = demand.loc[bad, "date"]
compare.round(2)

# %% [markdown]
# **Why these days are infeasible.** On these days $D_2 > \tfrac43 D_1$: the critical load is too large relative to the daytime load. The only exact solution would make the battery *charge* (negative $y$) while solar covers everything. Nothing in the model allows this.
#
# **Handling strategy.** I use **non-negative least squares** (`scipy.optimize.nnls`). It finds the non-negative dispatch that comes closest to both constraints in the least-squares sense. Plain clipping sets $y = 0$ but keeps the raw $x$, which over-supplies $D_1$ and $D_2$ by more than NNLS does. NNLS spreads the error more evenly: it slightly over-supplies $D_1$ and falls about 4.8 kWh short on $D_2$ on the worst day. Every repaired day appears in the report above with its gap (+ means over-supply, − means unmet demand), so the facility manager can see where diesel or load-shedding is needed. **Operational note:** at a health centre the critical load *must* be met, so in practice I would give $D_2$ priority by weighting its row more heavily in NNLS. Diesel backup is modelled in the extension.

# %%
plan = nnls_plan          # the feasible dispatch used from here on
plan.insert(0, "date", demand["date"])

# %% [markdown]
# ## Task 5: Statistics of solar and battery usage (`statistics` module)

# %%
rows = {}
for src in ["solar", "battery"]:
    v = plan[src].tolist()
    mean = statistics.mean(v)
    rows[src] = {"mean kWh": mean, "variance kWh²": statistics.variance(v),
                 "std kWh": statistics.stdev(v), "CV": statistics.stdev(v) / mean}
usage_stats = pd.DataFrame(rows).T
usage_stats.round(3)

# %% [markdown]
# **Which source is more volatile?** The two standard deviations are almost the same (about 5.2 kWh). The **battery is far more volatile relative to its size**: its coefficient of variation is about 0.64, against 0.14 for solar. The battery works as the *balancing* source. It absorbs whatever solar can't cover, so small swings in demand become large relative swings in battery draw, sometimes all the way down to 0. That matters for battery life, because deep, irregular cycling wears cells faster.

# %% [markdown]
# ## Task 6: Cost model (solar UGX 150/kWh, battery UGX 450/kWh)

# %%
tariff = CostModel({"solar": 150, "battery": 450})
plan["cost_ugx"] = tariff.daily_cost(plan)
monthly = tariff.total_cost(plan)
print(f"Mean daily cost : UGX {plan['cost_ugx'].mean():,.0f}")
print(f"Min / max daily : UGX {plan['cost_ugx'].min():,.0f} / {plan['cost_ugx'].max():,.0f}")
print(f"Monthly (30 d)  : UGX {monthly:,.0f}")
# hand check for day 1
r0 = plan.iloc[1]
assert np.isclose(r0.cost_ugx, 150 * r0.solar + 450 * r0.battery)
plan[["date", "solar", "battery", "cost_ugx"]].head().round(1)

# %% [markdown]
# ## Task 7: Stacked bar chart with daily cost on a second axis

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
days = np.arange(1, len(plan) + 1)
ax.bar(days, plan["solar"], color="#f2b705", label="Solar (x)")
ax.bar(days, plan["battery"], bottom=plan["solar"], color="#2f6db3", label="Battery (y)")
for d in days[bad]:
    ax.annotate("NNLS\nrepair", (d, plan["solar"].iloc[d - 1] + plan["battery"].iloc[d - 1]),
                ha="center", va="bottom", fontsize=7, color="crimson")
ax.set_xlabel("Day of month (September 2026, synthetic)")
ax.set_ylabel("Energy drawn (kWh)")
ax.set_xticks(days)
ax.set_ylim(0, (plan["solar"] + plan["battery"]).max() * 1.3)
ax2 = ax.twinx()
ax2.plot(days, plan["cost_ugx"] / 1000, "o-", color="crimson", lw=1.5, ms=4, label="Daily cost")
ax2.set_ylabel("Daily cost ('000 UGX)")
ax2.spines["right"].set_visible(True)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3)
ax.set_title("Daily solar vs battery dispatch and cost: cost spikes follow battery use")
fig.tight_layout()
fig.savefig(FIG_DIR / "p2_dispatch.png", bbox_inches="tight")
plt.show()
print("Correlation(battery kWh, cost):", round(np.corrcoef(plan["battery"], plan["cost_ugx"])[0, 1], 3))

# %% [markdown]
# The chart shows that **cost is driven by the battery, not by total energy.** Battery energy costs three times as much per kWh, so days with tall blue segments are the expensive days, even when the total bar height is similar.

# %% [markdown]
# ## Extension 1: `HybridMicroGrid` with a diesel generator
# $$3x+2y+z=D_1,\qquad 4x+y=D_2,\qquad x+y+z=D_3\ (\text{total daily energy drawn})$$

# %%
hybrid = HybridMicroGrid()
print(hybrid)
print("det by hand: 3(1·1-0·1) - 2(4·1-0·1) + 1(4·1-1·1) = 3 - 8 + 3 =", 3 - 8 + 3)
# Pick D3 = total energy of a sensible dispatch (x=30, y=10, z=5) to test
true = np.array([30, 10, 5.0])
d = hybrid.A @ true
print("Demands", d, "-> recovered", hybrid.solve_day(*d))

# %% [markdown]
# **What if the new equation depends linearly on the others?** Suppose the third constraint were $7x + 3y + z = D_3$, which is row 1 + row 2. Then:

# %%
dependent = HybridMicroGrid([[3, 2, 1], [4, 1, 0], [7, 3, 1]])
print(f"det = {dependent.determinant():.2e}, rank = {dependent.rank()}, cond = {dependent.condition_number():.2e}")
try:
    dependent.solve_day(100, 130, 230)
except np.linalg.LinAlgError as err:
    print("LinAlgError:", err)

# Consistent (D3 = D1 + D2) -> infinitely many solutions: the null space direction
null = linalg.null_space(dependent.A).ravel()
print("Null-space direction (x, y, z):", (null / null[0]).round(3))
sol, res, rk, sv = np.linalg.lstsq(dependent.A, [100, 130, 230], rcond=None)
print("One (minimum-norm) solution:", sol.round(3), "| residual:", np.round(dependent.A @ sol - [100, 130, 230], 6))
sol2, *_ = np.linalg.lstsq(dependent.A, [100, 130, 250], rcond=None)
print("Inconsistent D3=250 -> best fit misses by", np.round(dependent.A @ sol2 - [100, 130, 250], 3))

# %% [markdown]
# The 3×3 system is still well-posed (det = −2), but its condition number rises to about 24, so it is more sensitive to demand errors than the 2×2 system.
#
# With a dependent row, the determinant is 0 and the rank drops to 2. The third equation adds **no new information**, so the system either:
# * has **infinitely many solutions** when the demands are consistent ($D_3 = D_1 + D_2$). Any point on the line $s_0 + t\,(1, -4, 5)$ works, and the operator would need another criterion, such as lowest cost, to choose one; or
# * has **no solution** when they are inconsistent ($D_3 \neq D_1+D_2$). Least squares then gives only a compromise.
#
# `solve` refuses both cases. This is why `MicroGrid._check()` tests the rank and condition number before solving.

# %% [markdown]
# ## Extension 2: Monte Carlo sensitivity (±5% demand, 1,000 draws)

# %%
base_d = np.array([demand["D1"].mean(), demand["D2"].mean()])
base_s = grid.solve_day(*base_d)
sims = monte_carlo_sensitivity(grid, base_d, rel=0.05, draws=1000, rng=rng)
rel_change = (sims - base_s) / base_s
summary = pd.DataFrame({
    "base kWh": base_s,
    "sd kWh": sims.std(axis=0, ddof=1),
    "max |rel change| %": np.abs(rel_change).max(axis=0) * 100,
    "95% range kWh": [f"{np.percentile(sims[:, i], 2.5):.1f} – {np.percentile(sims[:, i], 97.5):.1f}" for i in range(2)],
}, index=grid.sources)
display(summary.round(2))
# Worst-case amplification observed vs condition-number bound
d_pert = base_d[:, None] * (1 + rng.uniform(-0.05, 0.05, (2, 1000)))
amp = (np.linalg.norm(grid.solve_many(d_pert) - base_s[:, None], axis=0) / np.linalg.norm(base_s)) / \
      (np.linalg.norm(d_pert - base_d[:, None], axis=0) / np.linalg.norm(base_d))
print(f"Observed max amplification ‖δs‖/‖s‖ ÷ ‖δd‖/‖d‖ = {amp.max():.2f}  (bound: cond = {grid.condition_number():.2f})")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, i, c in zip(axes, range(2), ["#f2b705", "#2f6db3"]):
    ax.hist(rel_change[:, i] * 100, bins=40, color=c, edgecolor="white")
    ax.set_title(f"{grid.sources[i].title()}: % change from ±5% demand noise")
    ax.set_xlabel("Change in dispatch (%)")
    ax.set_ylabel("Count of draws")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Relating this to the condition number.** A ±5% change in demand moves solar by at most about 11% (sd ≈ 2 kWh). The **battery moves by up to about 120%**, and its 95% range even dips just below zero, meaning the day becomes infeasible. The battery's base value is small (about 8 kWh), so the same absolute change is a large *relative* one: $\partial y/\partial D_1 = 4/5$ and $\partial y/\partial D_2 = -3/5$ pull in opposite directions. The largest observed amplification of the overall relative error stays **below κ ≈ 5.8**, as the theory predicts. The condition number bounds the error of the vector as a whole. Individual small components, like the battery here, can still be very sensitive.

# %% [markdown]
# ## Findings & Limitations
#
# **Findings.** The dispatch system is well-posed (det = −5) and well-conditioned (κ ≈ 5.8), so demand errors grow by at most about 6× in the solution. Over the synthetic month the health centre draws on average about **37.5 kWh of solar and 8 kWh of battery a day**. That costs about **UGX 9,300 a day**, or roughly **UGX 280,000 a month**. Solar supplies about 82% of the energy but only about 60% of the cost. Every kWh moved from battery to solar saves UGX 300, so the most effective way to cut costs is to shift flexible daytime loads (sterilisation, lab work) into sunny hours. Two days (7% of the month) were physically infeasible because the critical load was too high relative to the daytime load. NNLS repaired them and reported a small shortfall on $D_2$. In practice that shortfall would be covered by diesel, which is why the hybrid extension is useful. The battery is the most volatile source relative to its size (CV ≈ 0.64 against 0.14 for solar) and the most sensitive to demand forecast errors.
#
# **Limitations.** (1) The two equations are given, not derived. It is unclear why the "critical load" depends on $4x + y$, so the physical meaning of the coefficients should be checked with the engineer. (2) With 2 equations and 2 unknowns there is **no optimisation**: the demands fully determine the dispatch. A real planner would use linear programming to minimise cost subject to $\ge$ constraints, with battery state-of-charge, solar irradiance limits and inverter capacity. (3) The demand is synthetic, and the tariffs are illustrative *levelised* costs rather than cash costs. (4) NNLS weights both constraints equally, but clinically the critical load should come first. (5) Timing results depend on the machine and the SciPy version.
