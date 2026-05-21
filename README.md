# Bayesian Optimisation For Mountain Bike Frame

Mountain bike frame tube thickness optimisation using constrained Bayesian
Optimisation (BO) with a Gaussian-Process surrogate
---

## Contents

```
Phase_2/
├── test2.py        # main BO module + command line interface (CLI)
├── Phase_2.csv     # 165-point LHS sweep (FEA team output)
├── README.md       # this file
└── outputs/        # figures + results written here
```

---

## Setup

Python 3.9+.

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

For the live-Abaqus mode, `abaqus` must be on the system `PATH` (or edit
`ABAQUS_CMD` at the top of `test2.py`).

---

## Quick start

```bash
# Dev mode (GP stand-in for Abaqus)
python test2.py --iterations 370 --output-dir outputs

# Custom iteration count for smoke testing
python test2.py --iterations 20 --output-dir outputs_smoke

# Production mode (live Abaqus on WinIon)
python test2.py --abaqus --template frame_template.inp
```

Calling from a GUI:

```python
from test2 import run_optimisation

results = run_optimisation(
    csv_path="Phase_2.csv",
    output_dir="outputs",
    n_iterations=370,
    use_real_abaqus=False,           # True for live Abaqus
)
print(results["recommended_geometry_mm"])
print(results["fatigue_life_gain_factor"])
```

---

## Method

### Objectives (internal, BO-driven)

Aggregated as a weighted sum, minimised after sign-flipping:

| term | direction | weightage |
|---|---|---|
| Worst-joint design stress (`LOAD_FACTOR × max(σ_ST, σ_HT, σ_BB)`) | minimise | 0.20 |
| Vertical stiffness `Kv` | minimise | 0.50 |
| Lateral stiffness `Kl` | maximise | 0.30 |

The objective uses the **worst-joint** stress so that the BO drives down
exactly the quantity that controls fatigue life (the SN curve is read off
the highest-stressed joint). This keeps the objective and the fatigue
study in sync, so the post-BO design has a meaningfully lower peak
stress than the best CSV design.

`Kv` is weighted heavily (0.50) because mountain-bike vertical compliance
matters more than mid-range lateral-stiffness gains in this design space.

### Stress constraint

```
LOAD_FACTOR × σ_j  ≤  σ_yield / SF      for every joint j ∈ {ST, HT, BB}
```

with `σ_yield = 262 MPa` (group-confirmed 6061-T6), `SF = 1.5`, giving a
limit of **174.7 MPa**. The acquisition function is
constrained Expected Improvement:

```
acquisition(x) = EI_objective(x) × Π_j P(g_j(x) ≤ 0)
```

EI is estimated by Monte Carlo because the objective is a non-linear
function of multiple GP outputs; the feasibility probabilities are
closed-form Gaussian CDFs.

### Dynamic load factor

The CSV stresses are nominal static-FEA values. They are multiplied by
`LOAD_FACTOR = 2.0` to represent dynamic peak loading (impacts, rough
terrain). The factor is applied uniformly to the constraint, the objective
stress term, and the fatigue calculation — one coherent "design load" used
everywhere stress matters.

Per ISO 4210-6 / EN 14764, dynamic peak loads on mountain-bike frames are
typically 2–6× nominal pedalling loads. A factor of 2 is on the conservative
end and is a defensible engineering choice.

### Fatigue (diagnostic only, not in objective)

6061-T6 SN curve under fully-reversed loading (R = −1) using Basquin's law:

```
σ_a = σ_f' × (2N)^b
σ_f' = 660 MPa,  b = −0.11
Endurance limit  = 96.5 MPa (ASM/MatWeb 6061-T6)
```

Below the endurance limit, life is treated as infinite (capped at 10⁸
cycles for numerical stability). Reported life uses the **worst-joint
design stress** (LOAD_FACTOR × max σ_j).

### GP surrogate

* One GP per output (5 total): ARD-RBF kernel + WhiteKernel, `normalize_y=True`.
* **90/10 train/test split** for the parity plot (`TEST_FRAC = 0.10`).
* Hyperparameters re-optimised every `RETRAIN_EVERY = 50` BO iterations.
* `N_RESTARTS_EI = 3` L-BFGS-B restarts per EI maximisation.
* `N_MC = 256` Monte Carlo samples for the EI estimate.

---

## Output figures

Five PNGs are written to `output_dir/`:

| file | shows |
|---|---|
| `fig1_parity.png` | GP accuracy on the held-out 10% test set, with 2σ error bars and R² per output. |
| `fig2_convergence.png` | Running-best design plus all BO evaluations vs iteration, one panel per output (5 panels). |
| `fig3_pareto.png` | Three pairwise Pareto fronts: Kv vs Kl, worst-joint design stress vs Kv, worst-joint design stress vs Kl. Recommended design starred. |
| `fig4_sensitivity.png` | Variable importances from ARD inverse length-scales of the final working GPs. |
| `fig5_fatigue.png` | The 6061-T6 SN curve with **two points**: the worst CSV row (an un-optimised worst-case baseline) and the recommended design (after BO). Title shows the fatigue-life gain multiplier. |

---

## Result dictionary

`run_optimisation()` returns a dict with these keys (selected):

| key | meaning |
|---|---|
| `recommended_geometry_mm` | List of 9 tube thicknesses (mm). |
| `recommended_stresses_MPa` | `[ST, HT, BB]` nominal joint stresses at the recommendation. |
| `recommended_ST_design_stress_MPa` | `LF × σ_ST` at the recommendation. |
| `recommended_vertical_stiffness` / `recommended_lateral_stiffness` | `Kv`, `Kl` at the recommendation. |
| `predicted_fatigue_life_cycles` | Worst-joint Basquin life at the recommendation. |
| `baseline_fatigue_life_cycles` | Same metric for the worst CSV row (un-optimised baseline). |
| `fatigue_life_gain_factor` | `recommended_life / baseline_life`. |
| `stress_constraint_satisfied` | `True` if `LF × max σ_j ≤ limit`. |
| `test_set_R2` | GP R² on the held-out test set, one per output. |
| `figures` | Paths to the five PNGs. |
| `history.y_running_best` | Running-best 5-vector at every iteration. |

Geometry order: Seat tube, Down tube, Seat stays, Head tube, Bottom
bracket, Chain stay [BF], Chain stay [ST], Top tube [ST], Top tube [HT].

---

## Configuration

Defaults at the top of `test2.py`:

```python
N_ITER         = 370    # BO iterations
RETRAIN_EVERY  = 50     # GP refit interval
N_MC           = 256    # MC samples for EI
N_RESTARTS_EI  = 3      # L-BFGS-B restarts per EI maximisation
TEST_FRAC      = 0.10   # 90/10 train/test split
SEED           = 42

SIGMA_YIELD_MPA  = 262.0
SAFETY_FACTOR    = 1.5
LOAD_FACTOR      = 2.0
ENDURANCE_MPA    = 96.5
SIGMA_F_PRIME    = 660.0
BASQUIN_B        = -0.11

OBJ_WEIGHTS = [0.20, 0.50, 0.30]     # [worst-joint design stress, Kv, Kl]
```

---

## Abaqus integration (production mode)

When `use_real_abaqus=True`, the script needs:

1. **A `.inp` template** containing the placeholder strings:
   ```
   <SEAT_TUBE>     <DOWN_TUBE>     <SEAT_STAYS>
   <HEAD_TUBE>     <BOTTOM_BRACKET> <CHAIN_STAY_BF>
   <CHAIN_STAY_ST> <TOP_TUBE_ST>    <TOP_TUBE_HT>
   ```
   Each will be substituted with the candidate's tube thickness (in metres,
   6 decimal places).

2. **A stress-output script** that writes `<jobname>_stress.csv` with columns:
   ```
   seat_tube_joint_MPa, head_tube_joint_MPa, bottom_bracket_MPa,
   vertical_stiffness_N_per_mm, lateral_stiffness_N_per_mm
   ```

Abaqus job files land in `outputs/abaqus_runs/`.

---

## Progress callback (for the GUI)

```python
def update_bar(iteration, total, f_best):
    progress_bar.set_value(100 * iteration / total)

results = run_optimisation(csv_path="Phase_2.csv", progress_callback=update_bar)
```
