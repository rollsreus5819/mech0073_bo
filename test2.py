# =============================================================================
# 1. INTRODUCTION & IMPORTS

"""
Bike Frame Bayesian Optimisation

Constrained BO with a Gaussian-Process surrogate.  See README.md for the
full write-up of objectives, constraints, load-factor convention, and the
6061-T6 Basquin fatigue parameters.
"""

import os
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import norm

# =============================================================================
# 2. CONFIGURATION OF VARIABLES

SEED = 42
ABAQUS_CMD = "abaqus"

INPUT_COLS = [
    "Seat tube (m)",
    "Down tube (m)",
    "Seat stays (m)",
    "Head tube (m)",
    "Bottom bracket (m)",
    "Chain stay start (m) [Back fork side]",
    "Chain stay end (m) [seat tube side]",
    "Top tube start (m) [Seat tube side]",
    "Top tube end (m) [Head tube side]",
]

STRESS_COLS = [
    "Seat tube Joint (MPa)",
    "Head tube Joint (MPa)",
    "Bottom Bracket Joint (MPa)",
]

STIFFNESS_COLS = [
    "Vertical Stiffness (N/mm)",
    "Lateral Stiffness (N/mm)",
]

OUTPUT_COLS = STRESS_COLS + STIFFNESS_COLS

VAR_NAMES = [
    "Seat tube", "Down tube", "Seat stays", "Head tube", "Bottom bracket",
    "Chain stay [BF]", "Chain stay [ST]", "Top tube [ST]", "Top tube [HT]",
]

OUT_NAMES = [
    "Seat Tube Joint", "Head Tube Joint", "Bottom Bracket",
    "Vertical Stiffness", "Lateral Stiffness",
]

INP_PLACEHOLDERS = [
    "<SEAT_TUBE>", "<DOWN_TUBE>", "<SEAT_STAYS>", "<HEAD_TUBE>",
    "<BOTTOM_BRACKET>", "<CHAIN_STAY_BF>", "<CHAIN_STAY_ST>",
    "<TOP_TUBE_ST>", "<TOP_TUBE_HT>",
]

# Output index map: y = [ST, HT, BB, Kv, Kl]
IDX_STRESSES       = [0, 1, 2]
IDX_PRIMARY_STRESS = 0
IDX_VERT_STIFF     = 3
IDX_LAT_STIFF      = 4

# Internal BO weights: [worst-joint design stress, Kv, Kl], lower-is-better
OBJ_WEIGHTS = np.array([0.20, 0.50, 0.30])
OBJ_SIGNS   = np.array([+1,   +1,   -1])

# BO settings
N_ITER         = 370
RETRAIN_EVERY  = 50
N_MC           = 256
N_RESTARTS_EI  = 3
TEST_FRAC      = 0.10   # 90/10 train/test split

# Material (6061-T6), see README
SIGMA_YIELD_MPA  = 262.0
SAFETY_FACTOR    = 1.5
STRESS_LIMIT_MPA = SIGMA_YIELD_MPA / SAFETY_FACTOR
LOAD_FACTOR      = 2.0

# Basquin fatigue parameters (R = -1)
SIGMA_F_PRIME        = 660.0
BASQUIN_B            = -0.11
ENDURANCE_MPA        = 96.5
ENDURANCE_N          = 1e8
LOAD_AMPLITUDE_RATIO = 1.0

# =============================================================================
# 3. FATIGUE VARIABLES

def fatigue_life_6061(stress_amplitude_MPa):
    """Basquin life for 6061-T6; capped at ENDURANCE_N below the endurance limit."""
    s = np.atleast_1d(stress_amplitude_MPa).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        N = 0.5 * (s / SIGMA_F_PRIME) ** (1.0 / BASQUIN_B)
    N = np.where(s > ENDURANCE_MPA, N, ENDURANCE_N)
    return np.minimum(N, ENDURANCE_N)


def log10_fatigue_from_outputs(Y):
    """log10(life) from the worst-joint peak, after LOAD_FACTOR scaling."""
    Y = np.atleast_2d(Y)
    peak  = LOAD_FACTOR * np.max(Y[..., IDX_STRESSES], axis=-1)
    amp   = LOAD_AMPLITUDE_RATIO * peak
    return np.log10(np.maximum(fatigue_life_6061(amp), 1.0))

# =============================================================================
# 4. ABAQUS EVALUATOR (use only when use_real_abaqus=True)

def call_abaqus(x_phys, template_inp, work_dir, job_id):
    """Submit one Abaqus job; return [ST, HT, BB, Kv, Kl]."""
    os.makedirs(work_dir, exist_ok=True)
    job_name = f"frame_job_{job_id}"
    inp_path = os.path.join(work_dir, f"{job_name}.inp")

    with open(template_inp, "r") as f:
        inp_text = f.read()
    for placeholder, value in zip(INP_PLACEHOLDERS, x_phys):
        inp_text = inp_text.replace(placeholder, f"{value:.6f}")
    with open(inp_path, "w") as f:
        f.write(inp_text)

    cmd = [ABAQUS_CMD, f"job={job_name}", f"input={inp_path}", "interactive"]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Abaqus job {job_name} failed.\nSTDERR:\n{result.stderr}")

    out_csv = os.path.join(work_dir, f"{job_name}_stress.csv")
    df_out = pd.read_csv(out_csv)
    return np.array([
        df_out["seat_tube_joint_MPa"].iloc[0],
        df_out["head_tube_joint_MPa"].iloc[0],
        df_out["bottom_bracket_MPa"].iloc[0],
        df_out["vertical_stiffness_N_per_mm"].iloc[0],
        df_out["lateral_stiffness_N_per_mm"].iloc[0],
    ])

# =============================================================================
# 5. GP + PARETO HELPERS

def make_gp(n_vars, seed, n_restarts=3):
    """ARD-RBF + white-noise GP (matches Phase 1 setup)."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_vars), length_scale_bounds=(1e-2, 1e4))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )

def pareto_mask(Y, directions=None):
    """Boolean mask of Pareto-optimal points for a 2D problem."""
    if directions is None:
        directions = np.array([+1, +1])
    Y_min = Y * np.asarray(directions)
    mask = np.ones(Y_min.shape[0], dtype=bool)
    min_y2 = np.inf
    for idx in np.argsort(Y_min[:, 0]):
        if Y_min[idx, 1] < min_y2:
            min_y2 = Y_min[idx, 1]
        else:
            mask[idx] = False
    return mask

# =============================================================================
# 6. MAIN ENTRY POINT FOR OPTIMISATION

def run_optimisation(
    csv_path,
    template_inp=None,
    output_dir="outputs",
    n_iterations=N_ITER,
    use_real_abaqus=False,
    seed=SEED,
    progress_callback=None,
    stress_limit_MPa=STRESS_LIMIT_MPA,
    obj_weights=None,
):
    """Run the constrained Phase-2 BO pipeline.  See README.md for details."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if obj_weights is None:
        obj_weights = OBJ_WEIGHTS
    obj_weights = np.asarray(obj_weights, dtype=float)
    assert obj_weights.size == 3, "Need 3 weights: [worst_design_stress, Kv, Kl]"

    if use_real_abaqus and template_inp is None:
        raise ValueError("template_inp is required when use_real_abaqus=True")

    # =============================================================================
    # 6.1 LOAD CSV DATA

    df = pd.read_csv(csv_path)
    for c in ("Source", "SampleID"):
        if c in df.columns:
            df = df.drop(columns=c)

    X_all = df[INPUT_COLS].values
    Y_all = df[OUTPUT_COLS].values
    n_vars, n_out = X_all.shape[1], Y_all.shape[1]
    assert n_out == 5, f"Expected 5 outputs, got {n_out}"

    # =============================================================================
    # 6.2 90/10 TRAIN/TEST SPLIT

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=TEST_FRAC, random_state=seed, shuffle=True)

    mu_X, sig_X = X_train.mean(axis=0), X_train.std(axis=0)
    sig_X[sig_X == 0] = 1.0

    def normalise(X):
        return (X - mu_X) / sig_X

    X_train_n, X_test_n = normalise(X_train), normalise(X_test)
    lb_n, ub_n = normalise(X_all.min(axis=0)), normalise(X_all.max(axis=0))

    # =============================================================================
    # 6.3 INITIAL GP

    print(f"Fitting {n_out} initial GPs on {X_train_n.shape[0]} points...", flush=True)
    t_gp = time.time()
    gps_train = []
    for k in range(n_out):
        gps_train.append(make_gp(n_vars, seed).fit(X_train_n, Y_train[:, k]))
        print(f"  GP {k+1}/{n_out} ({OUT_NAMES[k]}) [{time.time()-t_gp:.1f}s]", flush=True)

    r2_test = []
    for k in range(n_out):
        mu_pred = gps_train[k].predict(X_test_n)
        ss_res = np.sum((Y_test[:, k] - mu_pred) ** 2)
        ss_tot = np.sum((Y_test[:, k] - Y_test[:, k].mean()) ** 2)
        r2_test.append(1 - ss_res / ss_tot)

    # =============================================================================
    # 6.4 ORACLE - ABAQUS STAND-IN

    if use_real_abaqus:
        work_dir = os.path.join(output_dir, "abaqus_runs")
        job_counter = {"id": 0}

        def oracle_query(x_phys):
            job_counter["id"] += 1
            return call_abaqus(x_phys, template_inp, work_dir, job_counter["id"])
    else:
        print(f"Fitting {n_out} oracle GPs on all {X_all.shape[0]} points...", flush=True)
        t_oracle = time.time()
        oracle_gps = [make_gp(n_vars, seed).fit(normalise(X_all), Y_all[:, k])
                      for k in range(n_out)]
        print(f"  oracle GPs fitted [{time.time()-t_oracle:.1f}s]", flush=True)

        def oracle_query(x_phys):
            x_n = normalise(x_phys.reshape(1, -1))
            return np.array([gp.predict(x_n)[0] for gp in oracle_gps])

    # =============================================================================
    # 6.5 INTERNAL SCALARISED OBJECTIVE 

    # Stress term = LOAD_FACTOR x max(ST, HT, BB) so the BO drives down the SAME quantity the fatigue diagnostic uses
    
    worst_train = LOAD_FACTOR * np.max(Y_train[:, IDX_STRESSES], axis=1)
    obj_scale = np.array([
        np.abs(worst_train).mean(),
        np.abs(Y_train[:, IDX_VERT_STIFF]).mean(),
        np.abs(Y_train[:, IDX_LAT_STIFF]).mean(),
    ])
    obj_scale[obj_scale == 0] = 1.0

    def objective(Y):
        Y = np.atleast_2d(Y)
        terms = np.column_stack([
            LOAD_FACTOR * np.max(Y[:, IDX_STRESSES], axis=1),
            Y[:, IDX_VERT_STIFF],
            Y[:, IDX_LAT_STIFF],
        ])
        return np.sum(obj_weights * OBJ_SIGNS * terms / obj_scale, axis=-1)

    def is_feasible(Y):
        Y = np.atleast_2d(Y)
        return np.all(LOAD_FACTOR * Y[:, IDX_STRESSES] <= stress_limit_MPa, axis=-1)

    # =============================================================================
    # 6.6 CONSTRAINED EXPECTED IMPROVEMENT

    def predict_all(x_n, gps_current):
        x_n = np.atleast_2d(x_n)
        mus, sds = [], []
        for gp in gps_current:
            m, s = gp.predict(x_n, return_std=True)
            mus.append(m)
            sds.append(s)
        return np.array(mus).T, np.array(sds).T

    def feasibility_probability(mu, sd):
        prob = np.ones(mu.shape[0])
        effective_limit = stress_limit_MPa / LOAD_FACTOR
        for k in IDX_STRESSES:
            sd_safe = np.maximum(sd[:, k], 1e-9)
            z = (effective_limit - mu[:, k]) / sd_safe
            prob *= norm.cdf(z)
        return prob

    def expected_improvement(mu, sd, f_best):
        n_points = mu.shape[0]
        samples = mu[:, None, :] + sd[:, None, :] * rng.standard_normal(
            (n_points, N_MC, n_out))
        f_samples = objective(samples.reshape(-1, n_out)).reshape(n_points, N_MC)
        return np.maximum(f_best - f_samples, 0.0).mean(axis=1)

    def acquisition(x_n, gps_current, f_best):
        mu, sd = predict_all(x_n, gps_current)
        return expected_improvement(mu, sd, f_best) * feasibility_probability(mu, sd)

    def propose_next(gps_current, f_best):
        best_x, best_a = None, -np.inf
        bounds = list(zip(lb_n, ub_n))
        starts = rng.uniform(lb_n, ub_n, size=(N_RESTARTS_EI, n_vars))
        for x0 in starts:
            res = minimize(
                lambda x: -acquisition(x, gps_current, f_best)[0],
                x0=x0, bounds=bounds, method="L-BFGS-B")
            if -res.fun > best_a:
                best_a, best_x = -res.fun, res.x
        return best_x

    # =============================================================================
    # 6.7 BO LOOP

    X_work_n = X_train_n.copy()
    Y_work = Y_train.copy()
    print(f"Fitting {n_out} BO-working GPs...", flush=True)
    t_work = time.time()
    gps_work = [make_gp(n_vars, seed).fit(X_work_n, Y_work[:, k])
                for k in range(n_out)]
    print(f"  working GPs fitted [{time.time()-t_work:.1f}s]", flush=True)

    feas_init = is_feasible(Y_work)
    if feas_init.any():
        f_best = float(objective(Y_work[feas_init]).min())
        best_design = Y_work[feas_init][int(np.argmin(objective(Y_work[feas_init])))]
    else:
        f_best = np.inf
        best_design = Y_work[0]

    y_history = [best_design.copy()]

    print(f"Running Phase-2 BO for {n_iterations} iterations "
          f"(use_real_abaqus={use_real_abaqus})")
    print(f"Stress constraint: LF ({LOAD_FACTOR}) x joint stress <= "
          f"{stress_limit_MPa:.1f} MPa (yield {SIGMA_YIELD_MPA:.0f} / SF {SAFETY_FACTOR})")
    print(f"Endurance {ENDURANCE_MPA:.1f} MPa, Basquin (sf'={SIGMA_F_PRIME}, b={BASQUIN_B})")

    t0 = time.time()
    for it in range(1, n_iterations + 1):
        x_next_n = propose_next(gps_work, f_best)
        x_next   = x_next_n * sig_X + mu_X
        y_next   = oracle_query(x_next)

        X_work_n = np.vstack([X_work_n, x_next_n])
        Y_work   = np.vstack([Y_work, y_next])

        if is_feasible(y_next)[0]:
            new_f = float(objective(y_next)[0])
            if new_f < f_best:
                f_best = new_f
                best_design = y_next.copy()
        y_history.append(best_design.copy())

        if it % 10 == 0 or it == 1:
            print(f"  iter {it:4d}/{n_iterations}  "
                  f"n_feasible = {is_feasible(Y_work).sum()}", flush=True)
        if it % RETRAIN_EVERY == 0:
            for k in range(n_out):
                gps_work[k] = make_gp(n_vars, seed).fit(X_work_n, Y_work[:, k])
            print(f"    [GP refit at iter {it}]", flush=True)
        if progress_callback is not None:
            progress_callback(it, n_iterations, f_best)

    print(f"BO finished in {time.time()-t0:.1f}s")

    # =============================================================================
    # 6.8 COMBINE CSV + BO POINTS 

    X_csv_all_n  = np.vstack([X_train_n, X_test_n])
    Y_csv_all    = np.vstack([Y_train, Y_test])
    X_all_eval_n = np.vstack([X_csv_all_n, X_work_n[X_train.shape[0]:]])
    Y_all_eval   = np.vstack([Y_csv_all, Y_work[X_train.shape[0]:]])

    # =============================================================================
    # 6.9 RECOMMENDED GEOMETRY

    feas_all = is_feasible(Y_all_eval)
    if feas_all.any():
        feas_idx = np.where(feas_all)[0]
        best_idx = int(feas_idx[int(np.argmin(objective(Y_all_eval[feas_idx])))])
    else:
        print("WARNING: no feasible point - falling back to min stress")
        best_idx = int(np.argmin(np.max(Y_all_eval[:, IDX_STRESSES], axis=1)))

    x_best = X_all_eval_n[best_idx] * sig_X + mu_X
    y_best = Y_all_eval[best_idx]

    peak_stress_best = float(np.max(y_best[IDX_STRESSES]))
    design_peak_best = LOAD_FACTOR * peak_stress_best
    life_best        = float(fatigue_life_6061(LOAD_AMPLITUDE_RATIO * design_peak_best)[0])

    # =============================================================================
    # 6.10 WORST CSV ROW 

    # The "pre-BO" reference is the CSV row with the HIGHEST worst-joint design stress

    csv_worst_design  = LOAD_FACTOR * np.max(Y_csv_all[:, IDX_STRESSES], axis=1)
    csv_baseline_idx  = int(np.argmax(csv_worst_design))
    y_csv_baseline    = Y_csv_all[csv_baseline_idx]
    peak_stress_base  = float(np.max(y_csv_baseline[IDX_STRESSES]))
    design_peak_base  = LOAD_FACTOR * peak_stress_base
    life_baseline     = float(fatigue_life_6061(LOAD_AMPLITUDE_RATIO * design_peak_base)[0])
    life_gain_factor  = life_best / life_baseline if life_baseline > 0 else np.inf

    # =============================================================================
    # 6.11 FIGURES

    figures = _save_figures(
        output_dir=output_dir, gps_train=gps_train, X_test_n=X_test_n, Y_test=Y_test,
        r2_test=r2_test, y_history=y_history, n_init_train=X_train.shape[0],
        Y_work=Y_work, Y_all_eval=Y_all_eval, y_best=y_best,
        y_csv_baseline=y_csv_baseline,
        gps_work=gps_work, n_vars=n_vars, n_out=n_out,
        n_csv=X_csv_all_n.shape[0], n_iter=n_iterations,
        stress_limit=stress_limit_MPa, load_factor=LOAD_FACTOR,
    )

    return {
        "recommended_geometry_m":              x_best.tolist(),
        "recommended_geometry_mm":             (x_best * 1000).tolist(),
        "recommended_stresses_MPa":            y_best[IDX_STRESSES].tolist(),
        "recommended_ST_stress_MPa":           float(y_best[IDX_PRIMARY_STRESS]),
        "recommended_ST_design_stress_MPa":    LOAD_FACTOR * float(y_best[IDX_PRIMARY_STRESS]),
        "recommended_worst_stress_MPa":        peak_stress_best,
        "recommended_design_worst_stress_MPa": design_peak_best,
        "recommended_vertical_stiffness":      float(y_best[IDX_VERT_STIFF]),
        "recommended_lateral_stiffness":       float(y_best[IDX_LAT_STIFF]),
        "predicted_fatigue_life_cycles":       life_best,
        "predicted_log10_fatigue_life":        float(np.log10(max(life_best, 1.0))),
        "baseline_stresses_MPa":               y_csv_baseline[IDX_STRESSES].tolist(),
        "baseline_worst_stress_MPa":           peak_stress_base,
        "baseline_design_worst_stress_MPa":    design_peak_base,
        "baseline_fatigue_life_cycles":        life_baseline,
        "fatigue_life_gain_factor":            float(life_gain_factor),
        "load_factor":                         float(LOAD_FACTOR),
        "endurance_MPa":                       float(ENDURANCE_MPA),
        "stress_constraint_satisfied":         bool(design_peak_best <= stress_limit_MPa),
        "stress_limit_MPa":                    float(stress_limit_MPa),
        "test_set_R2":                         [float(r) for r in r2_test],
        "figures":                             figures,
        "history": {
            "y_running_best": np.array(y_history).tolist(),
            "n_evaluations":  int(Y_work.shape[0]),
            "n_feasible":     int(is_feasible(Y_work).sum()),
        },
    }

# =============================================================================
# 7. FIGURES

def _save_figures(output_dir, gps_train, X_test_n, Y_test, r2_test,
                  y_history, n_init_train, Y_work, Y_all_eval,
                  y_best, y_csv_baseline, gps_work, n_vars, n_out, n_csv, n_iter,
                  stress_limit, load_factor):
    figures = {}

    # =============================================================================
    # 7.1 FIG 1 - PARITY

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for k in range(n_out):
        ax = axes_flat[k]
        mu_pred, sd_pred = gps_train[k].predict(X_test_n, return_std=True)
        y_true = Y_test[:, k]
        lo = min(y_true.min(), mu_pred.min())
        hi = max(y_true.max(), mu_pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
        ax.errorbar(y_true, mu_pred, yerr=2*sd_pred, fmt='o',
                    ms=5, capsize=3, alpha=0.8, color='steelblue')
        units = "MPa" if k < 3 else "N/mm"
        ax.set_xlabel(f"True ({units})")
        ax.set_ylabel(f"GP predicted ({units})")
        ax.set_title(f"{OUT_NAMES[k]}\n$R^2$ = {r2_test[k]:.4f}")
        ax.grid(alpha=0.3)
    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig1_parity.png")
    plt.savefig(fp, dpi=160, bbox_inches='tight'); plt.close()
    figures["parity"] = fp

    # =============================================================================
    # 7.2 FIG 2 - CONVERGENCE 

    all_evals_Y = Y_work[n_init_train:]
    y_hist = np.array(y_history)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for k, pos in enumerate(positions):
        ax = axes[pos]
        ax.plot(np.arange(len(y_hist)), y_hist[:, k], '-', lw=2, color='C3',
                label="Running-best design")
        ax.scatter(np.arange(1, len(all_evals_Y) + 1), all_evals_Y[:, k],
                   s=8, c='C0', alpha=0.4, label="BO evaluations")
        units = "MPa" if k < 3 else "N/mm"
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{OUT_NAMES[k]} ({units})")
        ax.set_title(OUT_NAMES[k])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 2].set_visible(False)
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig2_convergence.png")
    plt.savefig(fp, dpi=160, bbox_inches='tight'); plt.close()
    figures["convergence"] = fp

    # =============================================================================
    # 7.3 FIG 3 - PARETO FRONT

    worst_design = load_factor * np.max(Y_all_eval[:, IDX_STRESSES], axis=1)
    Kv_all       = Y_all_eval[:, IDX_VERT_STIFF]
    Kl_all       = Y_all_eval[:, IDX_LAT_STIFF]
    ws_best = float(load_factor * np.max(y_best[IDX_STRESSES]))
    Kv_best = float(y_best[IDX_VERT_STIFF])
    Kl_best = float(y_best[IDX_LAT_STIFF])

    pairs = [
        (Kv_all, Kl_all, +1, -1,
         "Vertical stiffness Kv (N/mm)  - minimise",
         "Lateral stiffness Kl (N/mm)  - maximise",
         "Kv vs Kl", Kv_best, Kl_best, False),
        (worst_design, Kv_all, +1, +1,
         f"Worst-joint design stress (MPa, LF={load_factor})  - minimise",
         "Vertical stiffness Kv (N/mm)  - minimise",
         "Worst stress vs Kv", ws_best, Kv_best, True),
        (worst_design, Kl_all, +1, -1,
         f"Worst-joint design stress (MPa, LF={load_factor})  - minimise",
         "Lateral stiffness Kl (N/mm)  - maximise",
         "Worst stress vs Kl", ws_best, Kl_best, True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (xv, yv, xd, yd, xlab, ylab, title,
             x_rec, y_rec, add_limit) in zip(axes, pairs):
        Y_pair = np.column_stack([xv, yv])
        mask = pareto_mask(Y_pair, np.array([xd, yd]))
        ax.scatter(xv[:n_csv], yv[:n_csv], s=14, c='lightgrey', alpha=0.7,
                   label=f"CSV sweep ({n_csv} pts)")
        ax.scatter(xv[n_csv:], yv[n_csv:], s=14, c='lightblue', alpha=0.7,
                   label=f"BO points ({n_iter} pts)")
        pf = Y_pair[mask]
        order = np.argsort(pf[:, 0])
        ax.scatter(pf[order, 0], pf[order, 1], s=28, c='blue',
                   label=f"Pareto front ({mask.sum()} pts)", zorder=3,
                   edgecolors='navy', linewidths=0.5)
        ax.scatter([x_rec], [y_rec], s=240, marker='*', c='gold',
                   edgecolors='k', linewidths=1.2, zorder=5, label="Recommended")
        if add_limit:
            ax.axvline(stress_limit, color='red', ls='--', lw=1.2,
                       label=f"Stress limit ({stress_limit:.0f} MPa)")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig3_pareto.png")
    plt.savefig(fp, dpi=160, bbox_inches='tight'); plt.close()
    figures["pareto"] = fp

    # =============================================================================
    # 7.4 FIG 4 - ARD SENSITIVITY

    importance = np.zeros((n_vars, n_out))
    for k in range(n_out):
        rbf = None
        for hp_val in gps_work[k].kernel_.get_params().values():
            if isinstance(hp_val, RBF):
                rbf = hp_val
                break
        ls = np.atleast_1d(rbf.length_scale)
        if ls.size == 1:
            ls = np.full(n_vars, ls.item())
        importance[:, k] = 1.0 / ls
        importance[:, k] /= importance[:, k].sum()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x_pos = np.arange(n_out)
    width = 0.085
    colours = plt.cm.tab10(np.linspace(0, 1, n_vars))
    for d in range(n_vars):
        ax.bar(x_pos + (d - n_vars/2)*width, importance[d], width,
               label=VAR_NAMES[d], color=colours[d])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(OUT_NAMES, rotation=15, ha='right')
    ax.set_ylabel("Relative importance (normalised 1/length-scale)")
    ax.set_title("Variable sensitivity - ARD length scales of final GP")
    ax.legend(ncol=3, fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig4_sensitivity.png")
    plt.savefig(fp, dpi=160, bbox_inches='tight'); plt.close()
    figures["sensitivity"] = fp

    # =============================================================================
    # 7.5 FIG 5 - SN CURVE WITH PRE AND POST BO

    peak_base = float(load_factor * np.max(y_csv_baseline[IDX_STRESSES]))
    peak_rec  = float(load_factor * np.max(y_best[IDX_STRESSES]))
    life_base = float(fatigue_life_6061(LOAD_AMPLITUDE_RATIO * peak_base)[0])
    life_rec  = float(fatigue_life_6061(LOAD_AMPLITUDE_RATIO * peak_rec)[0])
    gain      = life_rec / life_base if life_base > 0 else np.inf

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(1e3, 1e10)
    ax.set_ylim(min(50, ENDURANCE_MPA - 10), 320)

    s_basquin = np.linspace(ENDURANCE_MPA, 320, 400)
    N_basquin = 0.5 * (s_basquin / SIGMA_F_PRIME) ** (1.0 / BASQUIN_B)
    ax.semilogx(N_basquin, s_basquin, '-', color='C3', lw=2.2,
                label="6061-T6 SN curve (Basquin, R = -1)")
    ax.axhline(ENDURANCE_MPA, color='C3', ls='--', lw=1.6, alpha=0.6,
               label=f"Endurance plateau ({ENDURANCE_MPA:.1f} MPa)")

    ax.scatter([life_base], [peak_base], s=200, marker='o', c='lightgrey',
               edgecolors='k', linewidths=1.2, zorder=5,
               label=f"Before BO (worst CSV row)  {peak_base:.1f} MPa  ->  N = {life_base:.2e}")
    ax.scatter([life_rec], [peak_rec], s=260, marker='*', c='gold',
               edgecolors='k', linewidths=1.2, zorder=6,
               label=f"After BO (recommended)  {peak_rec:.1f} MPa  ->  N = {life_rec:.2e}")

    ax.set_xlabel("Cycles to failure $N$")
    ax.set_ylabel(f"Worst-joint design stress (MPa, LF={load_factor})")
    ax.set_title(f"Designs on the 6061-T6 SN curve  -  fatigue life gain: {gain:.2f} x")
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig5_fatigue.png")
    plt.savefig(fp, dpi=160, bbox_inches='tight'); plt.close()
    figures["fatigue"] = fp

    return figures

# =============================================================================
# 8. COMMAND LINE INTERFACE

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="Phase_2.csv")
    p.add_argument("--template", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--iterations", type=int, default=N_ITER)
    p.add_argument("--abaqus", action="store_true")
    p.add_argument("--stress-limit", type=float, default=STRESS_LIMIT_MPA)
    args = p.parse_args()

    res = run_optimisation(
        csv_path=args.csv,
        template_inp=args.template,
        output_dir=args.output_dir,
        n_iterations=args.iterations,
        use_real_abaqus=args.abaqus,
        stress_limit_MPa=args.stress_limit,
    )

    LF = res["load_factor"]
    print("\n" + "=" * 60)
    print("RECOMMENDED GEOMETRY")
    print("=" * 60)
    for v, val in zip(VAR_NAMES, res["recommended_geometry_mm"]):
        print(f"  {v:<18}  {val:7.4f} mm")

    print("-" * 60)
    print(f"STRESSES  (constraint: LF={LF} x nominal <= {res['stress_limit_MPa']:.1f} MPa)")
    for n, val in zip(OUT_NAMES[:3], res["recommended_stresses_MPa"]):
        design = LF * val
        ok = "OK" if design <= res["stress_limit_MPa"] else "VIOLATED"
        print(f"  {n:<18} nominal {val:6.2f}  ->  design {design:7.2f} MPa   [{ok}]")

    print("-" * 60)
    print("STIFFNESS")
    print(f"  Vertical stiffness   {res['recommended_vertical_stiffness']:7.1f} N/mm")
    print(f"  Lateral stiffness    {res['recommended_lateral_stiffness']:7.1f} N/mm")

    print("-" * 60)
    print(f"FATIGUE (LF={LF}, endurance {res['endurance_MPa']:.1f} MPa)")
    print(f"  Before BO (worst CSV) N = {res['baseline_fatigue_life_cycles']:.2e} cycles  "
          f"(worst joint {res['baseline_design_worst_stress_MPa']:.1f} MPa design)")
    print(f"  After  BO (recommend) N = {res['predicted_fatigue_life_cycles']:.2e} cycles  "
          f"(worst joint {res['recommended_design_worst_stress_MPa']:.1f} MPa design)")
    print(f"  Fatigue life gain     {res['fatigue_life_gain_factor']:.2f} x")

    print("-" * 60)
    print(f"  Test R\u00b2             {[f'{r:.3f}' for r in res['test_set_R2']]}")
    print(f"  Feasible designs    {res['history']['n_feasible']}"
          f" / {res['history']['n_evaluations']}")
    print("=" * 60)
    print(f"\nFigures: {args.output_dir}/")

# =============================================================================