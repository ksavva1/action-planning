"""
Simulation runner and result-aggregation utilities for the planning cascade model.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import DEVELOPMENTAL_STAGES, TASKS


# JSON serialisation
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy bool_, integer, floating, and ndarray types."""
    def default(self, o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


# Simulation runner
def run_simulation(stages=None, tasks=None, n_trials=20, seed=42):
    """Run all stage × task × trial combinations and return the raw results list.

    Args:
        stages: list of stage name strings (default: all in DEVELOPMENTAL_STAGES)
        tasks: list of task name strings (default: all in TASKS)
        n_trials: int number of trials per (stage, task) pair
        seed: int base random seed; each trial uses seed + trial_index

    Returns:
        list of TrialResult, one per trial.
    """
    from planning_cascade_model import PlanningCascadeNetwork

    if stages is None:
        stages = list(DEVELOPMENTAL_STAGES.keys())
    if tasks is None:
        tasks = list(TASKS.keys())

    results = []
    for sn in stages:
        p = DEVELOPMENTAL_STAGES[sn]
        for tn in tasks:
            task = TASKS[tn]
            for i in range(n_trials):
                net = PlanningCascadeNetwork(p, seed=seed + i)
                results.append(net.run_trial(task, trial_id=i))
    return results


# Result aggregation
def group_results(results):
    """Group a flat list of TrialResult objects by (params_name, task_name).

    Args:
        results: list of TrialResult

    Returns:
        dict mapping (params_name, task_name) → list of TrialResult
    """
    groups = {}
    for r in results:
        groups.setdefault((r.params_name, r.task_name), []).append(r)
    return groups


def summarise_group(stage, task, trials):
    """Compute aggregated statistics for one (stage, task) group of trials.

    Args:
        stage: str stage name
        task: str task name
        trials: list of TrialResult sharing this stage and task

    Returns:
        dict of summary statistics
    """
    n = len(trials)
    return {
        "stage": stage, "task": task, "n_trials": n,
        "success_rate": sum(t.success for t in trials) / n,
        "mean_timesteps": float(np.mean([t.timesteps_used for t in trials])),
        "mean_pos_error": float(np.mean([t.final_pos_error for t in trials])),
        "mean_angle_error": float(np.mean([t.final_angle_error for t in trials])),
        "mean_efficiency": float(np.mean([t.efficiency for t in trials])),
        "mean_movement_onset": float(np.mean([t.movement_onset for t in trials])),
        "mean_gaze_switches": float(np.mean([t.total_gaze_switches for t in trials])),
        "mean_object_fixation_pct": float(np.mean([t.object_fixation_pct for t in trials])),
        "mean_target_fixation_pct": float(np.mean([t.target_fixation_pct for t in trials])),
        "translate_before_rotate_rate": sum(
            1 for t in trials if t.translation_onset < t.rotation_onset) / n,
    }


def best_trial_trajectory(trials):
    """Return serialisable trajectory data for the trial closest to the goal.

    Args:
        trials: list of TrialResult

    Returns:
        dict with keys "steps", "success", "gaze_history"
    """
    best = min(trials, key=lambda t: t.final_pos_error + t.final_angle_error)
    steps = [
        {"t": s.t, "x": round(s.obj_x, 4), "y": round(s.obj_y, 4),
         "a": round(s.obj_angle, 4), "gz": s.gaze_target, "mv": s.movement_started,
         "pe": round(s.pos_error, 4), "ae": round(s.angle_error, 4)}
        for s in best.trajectory
    ]
    return {"steps": steps, "success": best.success, "gaze_history": best.gaze_history}


def dev_params_as_dict(stages_dict):
    """Serialise a stages dict to plain param-value dicts.

    Args:
        stages_dict: dict of {stage_name: DevelopmentalParams}

    Returns:
        dict of {stage_name: {param_name: value}}
    """
    keys = [
        "gaze_switch_rate", "fixation_duration_mean", "target_bias", "simultaneous_rate",
        "sampling_rate", "perceptual_noise", "location_acuity", "orientation_acuity",
        "relation_acuity", "wm_capacity", "wm_decay", "wm_unfixated_decay",
        "affordance_coupling", "planning_horizon", "habit_strength", "goal_directed_strength",
        "correction_rate", "initiation_threshold",
    ]
    return {name: {k: getattr(p, k) for k in keys} for name, p in stages_dict.items()}


def compile_results(results):
    """Aggregate trial results into summary stats and best-trial trajectories.

    Args:
        results: list of TrialResult from run_simulation()

    Returns:
        dict with keys "summary", "trajectories", "developmental_params",
            "stages", "tasks" — ready for JSON serialisation.
    """
    groups = group_results(results)
    summary, trajectories = {}, {}

    for (stage, task), trials in groups.items():
        k = f"{stage}_{task}"
        summary[k] = summarise_group(stage, task, trials)
        trajectories[k] = best_trial_trajectory(trials)

    return {
        "summary": summary,
        "trajectories": trajectories,
        "developmental_params": dev_params_as_dict(DEVELOPMENTAL_STAGES),
        "stages": list(DEVELOPMENTAL_STAGES.keys()),
        "tasks": list(TASKS.keys()),
    }
