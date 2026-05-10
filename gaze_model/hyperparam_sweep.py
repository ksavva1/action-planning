"""
Hyperparameter Sweep

Sweeps over specified parameter ranges, runs N trials per combination and outputs:
  1.  sweep_results.csv  - one row per (param_combo, task, trial)
  2.  sweep_summary.csv  - aggregated stats per (param_combo, task)
  3.  sweep_plots.png    - grid of metric plots

Use:
    python hyperparam_sweep.py                      # default sweep
    python hyperparam_sweep.py --trials 30          # more trials
    python hyperparam_sweep.py --param habit_strength --values 0.1 0.3 0.5 0.7 0.9
    python hyperparam_sweep.py --param gaze_switch_rate --values 0.1 0.2 0.3 0.4 0.5 0.6 0.7

The default sweep varies 4 parameters simultaneously across a grid that covers:
  - gaze_switch_rate   (information-gathering frequency)
  - habit_strength     (habitual vs goal-directed balance)
  - planning_horizon   (lookahead depth)
  - wm_capacity        (working memory slots)
"""

import numpy as np
import itertools, argparse, os, sys, csv, json
from dataclasses import dataclass, asdict, fields
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planning_cascade_model import (
    DevelopmentalParams, PlanningCascadeNetwork, TaskConfig, TASKS, TrialResult)


# Sweep config
# -------------------------------------------------------------------------

# Default grid: 4 params × several values each
DEFAULT_GRID = {
    "gaze_switch_rate":   [0.10, 0.25, 0.40, 0.55, 0.70],
    "habit_strength":     [0.15, 0.40, 0.65, 0.85],
    "planning_horizon":   [1, 3, 5],
    "wm_capacity":        [1, 3, 5],
}

# Base config (mid-range params)
BASE = DevelopmentalParams(
    name="sweep",
    gaze_switch_rate=0.35, fixation_duration_mean=3.0,
    target_bias=0.45, simultaneous_rate=0.20,
    sampling_rate=0.50, perceptual_noise=0.25,
    location_acuity=0.85, orientation_acuity=0.45, relation_acuity=0.20,
    wm_capacity=3, wm_decay=0.12, wm_unfixated_decay=0.25,
    affordance_coupling=0.45, affordance_noise=0.22,
    planning_horizon=3, motor_noise=0.22,
    habit_strength=0.45, goal_directed_strength=0.55,
    correction_rate=0.15, correction_delay=1,
    initiation_threshold=0.35,
)


def make_params(overrides: dict) -> DevelopmentalParams:
    """
    Create a DevelopmentalParams from BASE with overrides applied.
    
    Args:
        overrides: dict mapping parameter names (str) to override values

    Returns:
        DevelopmentalParams with BASE values except where overridden
    """
    d = {f.name: getattr(BASE, f.name) for f in fields(DevelopmentalParams)}
    d.update(overrides)

    # Keep habit + goal-directed balanced
    if "habit_strength" in overrides:
        d["goal_directed_strength"] = 1.0 - d["habit_strength"]

    d["name"] = "_".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    
    return DevelopmentalParams(**d)


# Sweeps
# -------------------------------------------------------------------------

def run_sweep(grid: dict, task_names: List[str], n_trials: int = 15,
              seed: int = 42, verbose: bool = True) -> List[dict]:
    """
    Run all combinations in the grid.
    
    Args:
        grid: dict mapping parameter names to lists of values to try.
        task_names: list of str task keys from TASKS.
        n_trials: int number of trials per (combo, task) pair.
        seed: int base random seed; each trial uses seed + trial_index.
        verbose: bool, print progress updates if True.

    Returns:
        list of dicts, one per (combo, task, trial). Each contains
            the swept parameter values plus: task, trial, success, timesteps,
            pos_error, angle_error, efficiency, gaze_switches, obj_fix_pct,
            tgt_fix_pct, movement_onset.
    """
    param_names = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in param_names)))
    total = len(combos) * len(task_names) * n_trials

    if verbose:
        print(f"Sweep: {len(param_names)} params × {len(combos)} combos "
              f"× {len(task_names)} tasks × {n_trials} trials = {total} runs")

    rows = []
    done = 0
    for combo in combos:
        overrides = dict(zip(param_names, combo))
        params = make_params(overrides)

        for tname in task_names:
            task = TASKS[tname]
            for trial_i in range(n_trials):
                net = PlanningCascadeNetwork(params, seed=seed + trial_i)
                res = net.run_trial(task, trial_id=trial_i)

                row = {**overrides}
                row["task"] = tname
                row["trial"] = trial_i
                row["success"] = int(res.success)
                row["timesteps"] = res.timesteps_used
                row["pos_error"] = round(res.final_pos_error, 4)
                row["angle_error"] = round(res.final_angle_error, 4)
                row["efficiency"] = round(res.efficiency, 4)
                row["gaze_switches"] = res.total_gaze_switches
                row["obj_fix_pct"] = round(res.object_fixation_pct, 4)
                row["tgt_fix_pct"] = round(res.target_fixation_pct, 4)
                row["movement_onset"] = res.movement_onset
                rows.append(row)

                done += 1
                if verbose and done % 200 == 0:
                    print(f"  {done}/{total} ({100*done/total:.0f}%)")

    if verbose:
        print(f"  {done}/{total} complete.")
    return rows


def summarise(rows: List[dict]) -> List[dict]:
    """
    Aggregate rows into summary stats.
    
    Args:
        rows: list of dicts from run_sweep(), one per trial.

    Returns:
        list of dicts, one per (combo, task). Each contains the swept param
            values plus: task, n_trials, success_rate, mean_timesteps,
            mean_pos_error, mean_angle_error, mean_efficiency,
            mean_gaze_switches, mean_movement_onset.
    """
    groups: Dict[str, list] = {}
    # Find which columns are sweep params
    metric_cols = {"task","trial","success","timesteps","pos_error","angle_error",
                   "efficiency","gaze_switches","obj_fix_pct","tgt_fix_pct","movement_onset"}
    param_cols = [k for k in rows[0] if k not in metric_cols]

    for r in rows:
        key = tuple(r[k] for k in param_cols) + (r["task"],)
        groups.setdefault(key, []).append(r)

    summary = []
    for key, trials in groups.items():
        s = {k: trials[0][k] for k in param_cols}
        s["task"] = trials[0]["task"]
        s["n_trials"] = len(trials)
        s["success_rate"] = np.mean([t["success"] for t in trials])
        s["mean_timesteps"] = np.mean([t["timesteps"] for t in trials])
        s["mean_pos_error"] = np.mean([t["pos_error"] for t in trials])
        s["mean_angle_error"] = np.mean([t["angle_error"] for t in trials])
        s["mean_efficiency"] = np.mean([t["efficiency"] for t in trials])
        s["mean_gaze_switches"] = np.mean([t["gaze_switches"] for t in trials])
        s["mean_movement_onset"] = np.mean([t["movement_onset"] for t in trials])
        summary.append(s)
    return summary


# Output
# -------------------------------------------------------------------------

def save_csv(rows, path):
    """
    Write list of dicts to csv
    """
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


def plot_sweep(summary, grid, out_path):
    """
    Plot success_rate, mean_timesteps, and efficiency vs each swept parameter.

    Args:
        summary: list of dicts from summarise().
        grid: dict of {param_name: [values]} defining the sweep.
        out_path: str file path for the output PNG.
    """
    param_names = sorted(grid.keys())
    n = len(param_names)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.8*n), facecolor="#0c0e12")
    fig.subplots_adjust(hspace=.45, wspace=.3, left=.07, right=.96,
                        top=.94, bottom=.06)
    fig.text(.5, .98, "Hyperparameter Sweep Results",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="#6ee7b7", fontfamily="serif",
             path_effects=[patheffects.withStroke(linewidth=2, foreground="#0c0e12")])

    metrics = [("success_rate", "Success Rate", "#6ee7b7"),
               ("mean_timesteps", "Mean Timesteps", "#60a5fa"),
               ("mean_efficiency", "Efficiency", "#fbbf24")]

    task_colors = {"rotate_insert":"#fb7185", "translate_only":"#fbbf24",
                   "rotate_only":"#60a5fa", "complex_manipulation":"#a78bfa"}

    for pi, pname in enumerate(param_names):
        vals = sorted(set(grid[pname]))
        for mi, (mkey, mtitle, mcol) in enumerate(metrics):
            ax = axes[pi, mi] if n > 1 else axes[mi]
            ax.set_facecolor("#1a1e28")
            for sp in ax.spines.values(): sp.set_color("#2a3148")
            ax.tick_params(colors="#636b83", labelsize=7)
            ax.set_xlabel(pname.replace("_"," "), fontsize=8, color="#9aa1b9")
            ax.set_ylabel(mtitle, fontsize=8, color="#9aa1b9")

            for tname, tcol in task_colors.items():
                pts = []
                for v in vals:
                    matching = [s for s in summary
                                if s.get(pname) == v and s["task"] == tname]
                    if matching:
                        pts.append((v, np.mean([s[mkey] for s in matching])))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=tcol, markersize=4, lw=1.5,
                            alpha=.8, label=tname.replace("_"," "))

            if pi == 0 and mi == 0:
                ax.legend(fontsize=6, facecolor="#13161d", edgecolor="#2a3148",
                          labelcolor="#e4e8f1", loc="best")

    plt.savefig(out_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_heatmaps(summary, grid, out_path):
    """
    Plot success rate as a 2D heatmap (only for 2 param sweeps).

    Args:
        summary: list of dicts from summarise().
        grid: dict with 2 keys mapping to value lists.
        out_path: str file path for the output PNG.
    """
    param_names = sorted(grid.keys())
    if len(param_names) != 2:
        return  # heatmaps only make sense for 2D sweeps

    p1, p2 = param_names
    v1, v2 = sorted(set(grid[p1])), sorted(set(grid[p2]))
    tasks = sorted(set(s["task"] for s in summary))

    fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 4.5),
                             facecolor="#0c0e12")
    if len(tasks) == 1: axes = [axes]
    fig.subplots_adjust(wspace=.35, top=.88, bottom=.15, left=.08, right=.95)
    fig.text(.5, .96, f"Success Rate Heatmap: {p1} vs {p2}",
             ha="center", fontsize=14, fontweight="bold", color="#6ee7b7")

    for ti, tname in enumerate(tasks):
        ax = axes[ti]; ax.set_facecolor("#1a1e28")
        mat = np.zeros((len(v2), len(v1)))

        for s in summary:
            if s["task"] == tname:
                i1 = v1.index(s[p1]); i2 = v2.index(s[p2])
                mat[i2, i1] = s["success_rate"]

        im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(v1))); ax.set_xticklabels(v1, fontsize=7, color="#9aa1b9")
        ax.set_yticks(range(len(v2))); ax.set_yticklabels(v2, fontsize=7, color="#9aa1b9")
        ax.set_xlabel(p1.replace("_"," "), fontsize=8, color="#9aa1b9")
        ax.set_ylabel(p2.replace("_"," "), fontsize=8, color="#9aa1b9")
        ax.set_title(tname.replace("_"," ").title(), fontsize=10, color="#e4e8f1")
        
        # Value labels
        for i2_ in range(len(v2)):
            for i1_ in range(len(v1)):
                ax.text(i1_, i2_, f"{mat[i2_,i1_]:.2f}", ha="center",
                        va="center", fontsize=6.5, color="black" if mat[i2_,i1_]>.5 else "white")
        
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    plt.savefig(out_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# CLI
# -----------------------------------------------------------------

def parse_value(v):
    """
    Try to parse as int, then float, then string.
    
    Args:
        v: str from argparse.

    Returns:
        int, float, or str depending on what parses successfully.
    """
    try: return int(v)
    except ValueError:
        try: return float(v)
        except ValueError: return v


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Hyperparameter sweep")
    pa.add_argument("--trials", type=int, default=15,
                    help="Trials per combination (default: 15)")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--task", nargs="+", default=None,
                    help="Tasks to include (default: all)")
    pa.add_argument("--param", type=str, default=None,
                    help="Single param to sweep (use with --values)")
    pa.add_argument("--values", nargs="+", default=None,
                    help="Values for --param")
    pa.add_argument("--param2", type=str, default=None,
                    help="Optional second param for 2D sweep")
    pa.add_argument("--values2", nargs="+", default=None,
                    help="Values for --param2")
    args = pa.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    task_names = args.task or list(TASKS.keys())

    # Build grid
    if args.param and args.values:
        grid = {args.param: [parse_value(v) for v in args.values]}
        if args.param2 and args.values2:
            grid[args.param2] = [parse_value(v) for v in args.values2]
    else:
        grid = DEFAULT_GRID

    print(f"\n{'='*60}")
    print("Hyperparam Sweep")
    print(f"{'='*60}")
    print(f"  Grid: { {k: v for k, v in grid.items()} }")
    print(f"  Tasks: {task_names}")
    print(f"  Trials per combo: {args.trials}")
    n_combos = 1
    for v in grid.values(): n_combos *= len(v)
    print(f"  Total combos: {n_combos}")
    print(f"  Total runs: {n_combos * len(task_names) * args.trials}")
    print(f"{'='*60}\n")

    # Run
    rows = run_sweep(grid, task_names, args.trials, args.seed)
    summary = summarise(rows)

    # Save CSVs
    save_csv(rows, os.path.join(out_dir, "sweep_results.csv"))
    save_csv(summary, os.path.join(out_dir, "sweep_summary.csv"))

    # Plots
    plot_sweep(summary, grid, os.path.join(out_dir, "sweep_plots.png"))
    if len(grid) == 2:
        plot_heatmaps(summary, grid, os.path.join(out_dir, "sweep_heatmap.png"))

    print("\nDone")