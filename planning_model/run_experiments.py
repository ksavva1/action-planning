"""Run all three experiments and every robustness check, writing results to JSON.

This script is the single entry point for reproducing the numbers reported in
the write up.

Usage:
    python run_experiments.py [--out results] [--quick]

``--quick`` reduces every trial count for a smoke test; it does not reproduce
the reported figures.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import analysis
from affordance_matrices import AFFORDANCE_MATRIX_VARIANTS
from experiment_config import (
    BATTERY_TASKS,
    DEVELOPMENTAL_PROFILES,
    EXTENDED_SWEEP_TRIALS,
    EXTENDED_SWEEP_VALUES,
    MATRIX_TASKS,
    MATRIX_TRIALS_PER_SEED_SET,
    PRIMARY_BASELINE,
    PRIMARY_SWEEP_TASK,
    SEED_SETS,
    STEP_LIMIT_VARIANTS,
    SWEEP_BASELINES,
    SWEEP_CONFIG,
    SWEEP_TASKS,
    SWEEP_TRIALS_PER_SEED_SET,
    TASK_META,
    TOLERANCE_VARIANTS,
    TRIALS_PER_SEED_SET,
    build_battery,
    sweep_config,
)
from model_utils import NumpyEncoder, make_rng, summarise_group
from planning_cascade_model import run_trial

SEED_POLICY = "independent"


def run_batches(params, task, label, seed_sets, trials_per_set):
    """Run one condition as independent seed batches."""

    return {
        base_seed: [
            run_trial(
                params,
                task,
                trial_id=trial,
                rng=make_rng(base_seed, label, trial, SEED_POLICY),
            )
            for trial in range(trials_per_set)
        ]
        for base_seed in seed_sets
    }


def flatten_batches(batches):
    """Flatten a ``{seed: trials}`` mapping in insertion order."""

    return [trial for batch in batches.values() for trial in batch]


def add_batch_stats(summary, per_batch, seed_sets):
    """Attach between-batch success-rate diagnostics to ``summary``."""

    rates = [per_batch[str(seed)]["success_rate"] for seed in seed_sets]
    summary["batch_success_rates"] = rates
    summary["between_batch_sd"] = float(np.std(rates, ddof=1))
    summary["between_batch_range"] = float(max(rates) - min(rates))
    return summary


def run_condition(params, task, label, seed_sets, trials_per_set, keep_trials=False):
    """Run one condition as several independent batches and summarise it.

    Args:
        params: developmental parameter set for this condition.
        task: task geometry, tolerances and step limit.
        label: condition label used to key the random streams.
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        keep_trials: if True, return the pooled trial list; otherwise None.

    Returns:
        Tuple of (pooled summary, per-batch summaries keyed by seed string,
        pooled trials or None).
    """

    batches = run_batches(params, task, label, seed_sets, trials_per_set)
    pooled = flatten_batches(batches)
    summary = summarise_group(label, task.name, pooled)
    per_batch = {
        str(seed): summarise_group(label, task.name, batch) for seed, batch in batches.items()
    }
    add_batch_stats(summary, per_batch, seed_sets)
    return summary, per_batch, pooled if keep_trials else None


def collect_failure_records(trials):
    """Per-failure (initial, peak, final) position error, so trajectories can be dropped.

    Args:
        trials: list of TrialResult objects to scan for failures.

    Returns:
        List of (initial, peak, final) position-error tuples for failed
        trials that moved at all.
    """

    return [
        (
            trial.trajectory[0].pos_error,
            max(step.pos_error for step in trial.trajectory),
            trial.final_pos_error,
        )
        for trial in trials
        if not trial.success and len(trial.trajectory) > 1
    ]


def summarise_failure_records(records):
    """Characterise what failure looks like, rather than only how large the error is.

    Args:
        records: list of (initial, peak, final) position-error tuples, as
            returned by :func:`collect_failure_records`.

    Returns:
        Dict of failure-count and descriptive statistics; ``{"n_failed": 0}``
        if ``records`` is empty.
    """

    if not records:
        return {"n_failed": 0}

    initial, peak, final = (np.array(column) for column in zip(*records))
    return {
        "n_failed": len(records),
        "mean_initial_pos_error": float(np.mean(initial)),
        "mean_peak_pos_error": float(np.mean(peak)),
        "mean_final_pos_error": float(np.mean(final)),
        "prop_ending_further_than_start": float(np.mean(final > initial)),
        "prop_ending_beyond_twice_start": float(np.mean(final > 2 * initial)),
        "prop_final_within_5pct_of_peak": float(np.mean(final >= 0.95 * peak)),
    }


def strip_trajectories(trials):
    """Drop the per-timestep records, which dominate memory, keeping the scalars.

    Args:
        trials: list of TrialResult objects, mutated in place.

    Returns:
        The same list, with each trial's trajectory and gaze history cleared.
    """

    for trial in trials:
        trial.trajectory = []
        trial.gaze_history = []
    return trials


# Experiment 1
def experiment_1(seed_sets, trials_per_set, log):
    """Developmental profiles across the 27-task battery.

    Args:
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        log: callback for progress messages.

    Returns:
        Dict of cell summaries, per-batch results, profile summaries,
        convergence curves and reports, failure diagnostics, marginal
        effects, demand contrasts, elongation effect, and informative-cell
        flags.
    """

    log("Experiment 1: profiles x battery")
    summaries, per_batch_all, curves = [], {}, {}
    cell_flags = {name: {} for name in DEVELOPMENTAL_PROFILES}
    profile_trials = {name: [] for name in DEVELOPMENTAL_PROFILES}
    profile_batches = {name: {seed: [] for seed in seed_sets} for name in DEVELOPMENTAL_PROFILES}
    failure_records = {name: [] for name in DEVELOPMENTAL_PROFILES}

    for profile_name, params in DEVELOPMENTAL_PROFILES.items():
        for task_name, task in BATTERY_TASKS.items():
            label = f"{profile_name}|{task_name}"
            batches = run_batches(params, task, label, seed_sets, trials_per_set)
            pooled = flatten_batches(batches)

            summary = summarise_group(profile_name, task_name, pooled)
            per_batch = {
                str(seed): summarise_group(profile_name, task_name, batch)
                for seed, batch in batches.items()
            }
            add_batch_stats(summary, per_batch, seed_sets)
            summaries.append(summary)
            per_batch_all[label] = per_batch

            cell_flags[profile_name][task_name] = [bool(trial.success) for trial in pooled]
            failure_records[profile_name].extend(collect_failure_records(pooled))
            for base_seed in seed_sets:
                profile_batches[profile_name][base_seed].extend(
                    trial.success for trial in batches[base_seed]
                )
            profile_trials[profile_name].extend(strip_trajectories(pooled))
        log(f"  profile {profile_name} done")

    # At each k the profile estimate is recomputed from the first k trials of
    # every cell, so the curve shows what would have been concluded had the
    # battery been run at that resolution.
    total_per_cell = len(seed_sets) * trials_per_set
    grid = sorted({k for k in (5, 10, 20, 25, 30, 40, 50, 75, 100, 125, 150) if k <= total_per_cell}
                  | {total_per_cell})
    for profile_name, tasks in cell_flags.items():
        curve = []
        for k in grid:
            flags = [flag for cell in tasks.values() for flag in cell[:k]]
            successes = int(sum(flags))
            low, high = analysis.wilson_interval(successes, len(flags))
            curve.append({
                "n": k,
                "n_total": len(flags),
                "estimate": successes / len(flags),
                "ci_low": low, "ci_high": high,
                "half_width": (high - low) / 2,
            })
        curves[f"profile_{profile_name}"] = curve

    interior = sorted(
        (row for row in summaries if 0.2 <= row["success_rate"] <= 0.8),
        key=lambda row: abs(row["success_rate"] - 0.5),
    )[:6]
    for row in interior:
        curves[f"cell_{row['stage']}_{row['task']}"] = analysis.convergence_curve(
            cell_flags[row["stage"]][row["task"]],
        )
    diagnostics = {
        profile_name: summarise_failure_records(records)
        for profile_name, records in failure_records.items()
    }

    profile_summary = {}
    for profile_name, trials in profile_trials.items():
        row = summarise_group(profile_name, "battery", trials)
        rates = [
            float(np.mean(profile_batches[profile_name][seed])) for seed in seed_sets
        ]
        row["batch_success_rates"] = rates
        row["between_batch_sd"] = float(np.std(rates, ddof=1))
        row["between_batch_range"] = float(max(rates) - min(rates))
        profile_summary[profile_name] = row

    marginals = {
        factor: analysis.marginal_by_factor(summaries, TASK_META, factor)
        for factor in ("dist", "rot", "aspect")
    }
    contrasts = {factor: analysis.demand_contrasts(values) for factor, values in marginals.items()}

    return {
        "cell_summaries": summaries,
        "per_batch": per_batch_all,
        "profile_summary": profile_summary,
        "convergence": curves,
        "convergence_report": analysis.convergence_report(curves),
        "failed_trajectories": diagnostics,
        "marginals": marginals,
        "demand_contrasts": contrasts,
        "elongation": analysis.elongation_effect(summaries, TASK_META),
        "informative_cells": analysis.informative_cells(summaries),
    }


def experiment_1_design_robustness(seed_sets, trials_per_set, log):
    """Success tolerance, step limit and object symmetry robustness checks.

    Args:
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        log: callback for progress messages.

    Returns:
        Dict with per-design profile summaries and design comparisons for
        tolerance, step limit and symmetry.
    """

    log("Experiment 1 robustness: tolerance / step limit / symmetry")
    designs = {}

    for name, (position_tolerance, angle_tolerance) in TOLERANCE_VARIANTS.items():
        tasks, _ = build_battery(position_tolerance=position_tolerance, angle_tolerance=angle_tolerance)
        designs[f"tolerance_{name}"] = tasks
    for name, (intercept, slope) in STEP_LIMIT_VARIANTS.items():
        if name == "default":
            continue
        tasks, _ = build_battery(intercept=intercept, slope=slope)
        designs[f"steplimit_{name}"] = tasks
    tasks, _ = build_battery(angular_symmetry=2)
    designs["symmetry_rectangle"] = tasks

    output = {}
    for design_name, tasks in designs.items():
        per_profile = {}
        for profile_name, params in DEVELOPMENTAL_PROFILES.items():
            pooled = []
            for task_name, task in tasks.items():
                label = f"{design_name}|{profile_name}|{task_name}"
                batches = run_batches(params, task, label, seed_sets, trials_per_set)
                pooled.extend(flatten_batches(batches))
            per_profile[profile_name] = summarise_group(profile_name, design_name, pooled)
        output[design_name] = per_profile
        log(f"  {design_name} done")

    return {
        "per_design": output,
        "tolerance_comparison": analysis.compare_designs(
            {k: v for k, v in output.items() if k.startswith("tolerance_")}
        ),
        "steplimit_comparison": analysis.compare_designs(
            {k: v for k, v in output.items()
             if k.startswith("steplimit_") or k == "tolerance_default"}
        ),
        "symmetry_comparison": analysis.compare_designs(
            {k: v for k, v in output.items()
             if k in ("tolerance_default", "symmetry_rectangle")}
        ),
    }


# Experiment 2
def run_sweep(baseline, task, config, seed_sets, trials_per_set, label_prefix):
    """Sweep every parameter in ``config`` around ``baseline`` on ``task``.

    Args:
        baseline: DevelopmentalParams to vary one parameter from at a time.
        task: TaskConfig to run every condition on.
        config: mapping from parameter name to its list of sweep values.
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        label_prefix: prefix for the random-stream labels.

    Returns:
        Mapping from parameter name to {value: summary dict}.
    """

    summaries = {}
    for parameter, values in config.items():
        summaries[parameter] = {}
        for value in values:
            overrides = {parameter: value}
            if parameter == "habit_strength":
                overrides["goal_directed_strength"] = round(1.0 - value, 4)
            params = replace(baseline, name=f"{baseline.name}_{parameter}_{value}", **overrides)
            label = f"{label_prefix}|{parameter}|{value}"
            summary, _, _ = run_condition(params, task, label, seed_sets, trials_per_set)
            summaries[parameter][value] = summary
    return summaries


def experiment_2(seed_sets, trials_per_set, extended_trials, log):
    """Local one-factor-at-a-time sensitivity, and its stability across designs.

    Args:
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch for the primary sweep.
        extended_trials: trials run within each batch for the extended grid.
        log: callback for progress messages.

    Returns:
        Dict with the primary sweep summaries and sensitivity index, the
        extended baseline x task grid, and rank stability across it.
    """

    log("Experiment 2: primary sweep")
    primary = run_sweep(
        SWEEP_BASELINES[PRIMARY_BASELINE], SWEEP_TASKS[PRIMARY_SWEEP_TASK], SWEEP_CONFIG,
        seed_sets, trials_per_set, f"{PRIMARY_BASELINE}|{PRIMARY_SWEEP_TASK}",
    )
    primary_index = analysis.relative_sensitivity_index(primary)

    log("Experiment 2: extended baseline x task grid")
    coarse = sweep_config(EXTENDED_SWEEP_VALUES)
    grid, rankings = {}, {}
    for baseline_name, baseline in SWEEP_BASELINES.items():
        for task_name, task in SWEEP_TASKS.items():
            key = f"{baseline_name}|{task_name}"
            summaries = run_sweep(
                baseline, task, coarse, seed_sets[:2], extended_trials, key,
            )
            index = analysis.relative_sensitivity_index(summaries)
            grid[key] = {
                "mean_index": index["mean_index"],
                "order": index["order"],
                "baseline_success": {
                    parameter: {
                        str(value): summaries[parameter][value]["success_rate"]
                        for value in summaries[parameter]
                    }
                    for parameter in ("planning_horizon",)
                },
            }
            rankings[key] = index["mean_index"]
            log(f"  {key} done")

    return {
        "primary": {
            parameter: {str(value): summary for value, summary in by_value.items()}
            for parameter, by_value in primary.items()
        },
        "primary_index": primary_index,
        "grid": grid,
        "rank_stability": analysis.rank_stability(rankings),
    }


# Experiment 3
def experiment_3(seed_sets, trials_per_set, log):
    """Affordance-matrix robustness across a difficulty-graded task set.

    Args:
        seed_sets: base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        log: callback for progress messages.

    Returns:
        Dict with per-cell summaries, informative-cell flags, matrix
        contrasts, ordering-preservation flags, and jitter-vs-separation
        diagnostics.
    """

    log("Experiment 3: matrices x profiles x tasks")
    cells, summaries_flat = {}, []
    for profile_name, base_params in DEVELOPMENTAL_PROFILES.items():
        for task_name, task in MATRIX_TASKS.items():
            for variant in AFFORDANCE_MATRIX_VARIANTS:
                params = replace(
                    base_params,
                    name=f"{profile_name}_{variant}",
                    affordance_matrix_variant=variant,
                )
                label = f"{profile_name}|{task_name}|{variant}"
                summary, _, _ = run_condition(params, task, label, seed_sets, trials_per_set)
                summary["stage"] = profile_name
                summary["task"] = task_name
                summary["variant"] = variant
                cells.setdefault(profile_name, {}).setdefault(task_name, {})[variant] = summary
                summaries_flat.append(summary)
        log(f"  profile {profile_name} done")

    informative = analysis.informative_cells(
        [row for row in summaries_flat if row["variant"] == "baseline"]
    )
    contrasts = {
        profile: {
            task: analysis.matrix_contrast(variants)
            for task, variants in tasks.items()
        }
        for profile, tasks in cells.items()
    }
    ordering_preserved = {}
    for task_name in MATRIX_TASKS:
        for variant in AFFORDANCE_MATRIX_VARIANTS:
            rates = [cells[profile][task_name][variant]["success_rate"]
                     for profile in DEVELOPMENTAL_PROFILES]
            ordering_preserved[f"{task_name}|{variant}"] = bool(
                all(earlier <= later for earlier, later in zip(rates, rates[1:]))
            )

    jitter = {
        profile: analysis.matrix_separation_vs_jitter(
            AFFORDANCE_MATRIX_VARIANTS,
            DEVELOPMENTAL_PROFILES[profile].affordance_coupling,
            DEVELOPMENTAL_PROFILES[profile].affordance_jitter_sd,
        )
        for profile in DEVELOPMENTAL_PROFILES
    }

    return {
        "cells": cells,
        "informative_cells": informative,
        "matrix_contrasts": contrasts,
        "ordering_preserved": ordering_preserved,
        "jitter_vs_separation": jitter,
    }


# Entry point
def main():
    """Parse CLI arguments and run the requested experiments, writing JSON output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default="all")
    arguments = parser.parse_args()

    started = time.time()

    def log(message):
        """Print ``message`` prefixed with the elapsed time since start."""

        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    seed_sets = SEED_SETS[:2] if arguments.quick else SEED_SETS
    battery_trials = 5 if arguments.quick else TRIALS_PER_SEED_SET
    sweep_trials = 3 if arguments.quick else SWEEP_TRIALS_PER_SEED_SET
    extended_trials = 3 if arguments.quick else EXTENDED_SWEEP_TRIALS
    matrix_trials = 3 if arguments.quick else MATRIX_TRIALS_PER_SEED_SET
    robustness_trials = 3 if arguments.quick else 20

    out = Path(arguments.out)
    out.mkdir(exist_ok=True)

    def dump(name, payload):
        """Write ``payload`` to ``<out>/<name>.json`` using NumpyEncoder."""

        with open(out / f"{name}.json", "w") as handle:
            json.dump(payload, handle, cls=NumpyEncoder, indent=1)
        log(f"wrote {name}.json")

    wanted = arguments.only.split(",")
    everything = "all" in wanted

    if everything or "exp1" in wanted:
        dump("experiment_1", experiment_1(seed_sets, battery_trials, log))
    if everything or "robust" in wanted:
        dump("experiment_1_robustness",
             experiment_1_design_robustness(seed_sets[:2], robustness_trials, log))
    if everything or "exp2" in wanted:
        dump("experiment_2", experiment_2(seed_sets, sweep_trials, extended_trials, log))
    if everything or "exp3" in wanted:
        dump("experiment_3", experiment_3(seed_sets, matrix_trials, log))

    log("done")


if __name__ == "__main__":
    main()
