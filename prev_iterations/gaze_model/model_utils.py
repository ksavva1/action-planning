"""Configuration, result containers, and simulation helpers."""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class DevelopmentalParams:
    """
    Parameters for one developmental profile in the planning cascade model.

    The stages are not ages directly; they are ordered profiles of increasing
    gaze flexibility, perceptual acuity, working-memory stability, planning
    horizon, and goal-directed control.

    Attributes:
        name: identifier for this configuration.
        gaze_switch_rate: base probability of switching fixation per timestep.
        fixation_duration_mean: mean dwell timesteps before switching is likely.
        target_bias: probability of remaining target-oriented after looking away
            from the target; higher values model stronger goal-directed attention.
        simultaneous_rate: probability of extracting relational features when
            object and target have both been fixated recently.
        sampling_rate: probability that each visible feature is sampled.
        perceptual_noise: standard deviation of Gaussian noise added to percepts.
        location_acuity: noise reduction factor for x/y features.
        orientation_acuity: noise reduction factor for angle and size features.
        relation_acuity: noise reduction factor for relational gap features.
        wm_capacity: maximum number of strong traces maintained at once.
        wm_decay: per-timestep decay for currently attended features.
        wm_unfixated_decay: faster decay for unattended and relational features.
        affordance_coupling: strength of perception-to-action mapping.
        affordance_noise: standard deviation of affordance activation noise.
        planning_horizon: relative lookahead depth, capped at 6 in motor planning.
        motor_noise: standard deviation of motor command noise.
        habit_strength: weight of the translate-first habit.
        goal_directed_strength: weight of goal-directed motor control.
        correction_rate: gain on the delayed online error-correction signal.
        correction_delay: timesteps before feedback can affect the command.
        initiation_threshold: mean trace strength required to begin moving.
        affordance_matrix_variant: named affordance weight matrix to use.
    """

    name: str = "default"

    # Gaze control parameters.
    gaze_switch_rate: float = 0.30
    fixation_duration_mean: float = 3.0
    target_bias: float = 0.40
    simultaneous_rate: float = 0.15

    # Perceptual parameters.
    sampling_rate: float = 0.40
    perceptual_noise: float = 0.30
    location_acuity: float = 0.85
    orientation_acuity: float = 0.40
    relation_acuity: float = 0.15

    # Working memory parameters.
    wm_capacity: int = 3
    wm_decay: float = 0.12
    wm_unfixated_decay: float = 0.28

    # Affordance parameters.
    affordance_coupling: float = 0.40
    affordance_noise: float = 0.25

    # Motor planning parameters.
    planning_horizon: int = 2
    motor_noise: float = 0.25

    # Habit vs. goal-directed control balance.
    habit_strength: float = 0.55
    goal_directed_strength: float = 0.45

    # Online error correction parameters.
    correction_rate: float = 0.12
    correction_delay: int = 2

    # Movement initiation threshold.
    initiation_threshold: float = 0.35

    # Affordance matrix experiment parameter.
    affordance_matrix_variant: str = "baseline"


@dataclass
class TaskConfig:
    """
    Defines a 2D object manipulation task with start and goal poses.

    The task geometry is 2D: x and y describe position in the workspace plane,
    and angle describes orientation. The agent must move the object from its
    starting pose to the goal pose within the specified tolerances.

    Attributes:
        name: identifier string for this task
        start_x, start_y, start_angle: initial position and orientation of object
        goal_x, goal_y, goal_angle: target position and required orientation
        obj_width, obj_height: lateral dimensions of the manipulated object
        position_tolerance: maximum Euclidean distance from goal centre for success
        angle_tolerance: maximum absolute angular difference from goal angle for success
            (radians)
        max_timesteps: trial terminates after this many steps regardless of success
    """

    name: str = "task"
    start_x: float = 0.0
    start_y: float = 0.0
    start_angle: float = 0.0
    goal_x: float = 0.5
    goal_y: float = 0.0
    goal_angle: float = 0.0
    obj_width: float = 0.3
    obj_height: float = 0.4
    position_tolerance: float = 0.05
    angle_tolerance: float = 0.10
    max_timesteps: int = 120


@dataclass
class TimestepRecord:
    """
    State snapshot at one simulation timestep.

    Attributes:
        t: timestep index
        obj_x, obj_y, obj_angle: current pose after movement
        gaze_target: "object" or "target" — what the eye is fixating
        movement_started: whether the information threshold has been reached
        obj_info: mean WM trace strength for object features [0:5]
        tgt_info: mean WM trace strength for target features [5:8]
        rel_info: mean WM trace strength for relational features [8:11]
        pos_error: Euclidean distance from object centre to goal centre
        angle_error: absolute angular distance from object angle to goal angle
        gaze_switches: cumulative number of fixation switches so far
    """

    t: int
    obj_x: float
    obj_y: float
    obj_angle: float
    gaze_target: str
    movement_started: bool
    obj_info: float
    tgt_info: float
    rel_info: float
    pos_error: float
    angle_error: float
    gaze_switches: int


@dataclass
class TrialResult:
    """
    Complete results from one simulated trial.

    Attributes:
        params_name: name of the DevelopmentalParams config used
        task_name: name of the TaskConfig used
        trial_id: index of this trial within a batch
        success: True if the object reached the goal within position and angle tolerance
        timesteps_used: number of timesteps before the trial ended
        final_pos_error: Euclidean position error at trial end
        final_angle_error: absolute angle error at trial end
        trajectory: list of TimestepRecord, one per timestep
        movement_onset: timestep when motor output first occurred
        rotation_onset: timestep of first significant rotation command
        translation_onset: timestep of first significant translation command
        efficiency: ratio of optimal straight-line distance to actual path length (0-1)
        total_gaze_switches: total object-to-target fixation switches during trial
        object_fixation_pct: fraction of timesteps spent fixating the object
        target_fixation_pct: fraction of timesteps spent fixating the target
        gaze_history: full sequence of "object"/"target" strings, one per timestep
    """

    params_name: str
    task_name: str
    trial_id: int
    success: bool
    timesteps_used: int
    final_pos_error: float
    final_angle_error: float
    trajectory: list[TimestepRecord]
    movement_onset: int
    rotation_onset: int
    translation_onset: int
    efficiency: float
    total_gaze_switches: int
    object_fixation_pct: float
    target_fixation_pct: float
    gaze_history: list[str]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalar and array values to Python types."""

    def default(self, obj):
        """Convert NumPy objects before falling back to the parent encoder."""

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_simulation(
    stages: dict[str, DevelopmentalParams],
    task_configs: dict[str, TaskConfig],
    n_trials: int = 20,
    seed: int = 42,
) -> list:
    """
    Run all requested stage x task x trial combinations.

    Args:
        stages: dict mapping name → DevelopmentalParams.
        task_configs: dict mapping name → TaskConfig.
        n_trials: number of trials per stage/task pair.
        seed: base seed; trial i uses seed + i for reproducibility.

    Returns:
        Flat list of TrialResult objects.
    """

    from prev_iterations.gaze_model.planning_cascade_model import run_trial

    return [
        run_trial(params, task, seed=seed + trial, trial_id=trial)
        for params in stages.values()
        for task in task_configs.values()
        for trial in range(n_trials)
    ]


def group_results(results):
    """Group a flat list of TrialResult objects by (params_name, task_name)."""

    groups = {}
    for result in results:
        groups.setdefault((result.params_name, result.task_name), []).append(result)
    return groups


def summarise_group(stage, task, trials):
    """Compute aggregated statistics for one (stage, task) group of trials."""

    return {
        "stage": stage,
        "task": task,
        "n_trials": len(trials),
        "success_rate": np.mean([trial.success for trial in trials]),
        "mean_timesteps": float(np.mean([trial.timesteps_used for trial in trials])),
        "mean_pos_error": float(np.mean([trial.final_pos_error for trial in trials])),
        "mean_angle_error": float(np.mean([trial.final_angle_error for trial in trials])),
        "mean_efficiency": float(np.mean([trial.efficiency for trial in trials])),
        "mean_movement_onset": float(np.mean([trial.movement_onset for trial in trials])),
        "mean_gaze_switches": float(np.mean([trial.total_gaze_switches for trial in trials])),
        "mean_object_fixation_pct": float(np.mean([trial.object_fixation_pct for trial in trials])),
        "mean_target_fixation_pct": float(np.mean([trial.target_fixation_pct for trial in trials])),
        "translate_before_rotate_rate": np.mean([
            trial.translation_onset < trial.rotation_onset for trial in trials
        ]),
    }


def best_trial_trajectory(trials):
    """
    Return compact trajectory data for the trial closest to the goal.

    The best trial is chosen by final position plus angle error so animations and
    dashboards show a representative successful-looking attempt when available.
    """

    trial = min(trials, key=lambda item: item.final_pos_error + item.final_angle_error)
    return {
        "steps": [
            {
                "t": step.t,
                "x": round(step.obj_x, 4),
                "y": round(step.obj_y, 4),
                "a": round(step.obj_angle, 4),
                "gz": step.gaze_target,
                "mv": step.movement_started,
                "pe": round(step.pos_error, 4),
                "ae": round(step.angle_error, 4),
            }
            for step in trial.trajectory
        ],
        "success": trial.success,
        "gaze_history": trial.gaze_history,
    }


def dev_params_as_dict(stages: dict[str, DevelopmentalParams]) -> dict:
    """Return serialisable developmental parameters for the given stages."""

    keys = [
        "gaze_switch_rate", "fixation_duration_mean", "target_bias", "simultaneous_rate",
        "sampling_rate", "perceptual_noise", "location_acuity", "orientation_acuity",
        "relation_acuity", "wm_capacity", "wm_decay", "wm_unfixated_decay",
        "affordance_matrix_variant", "affordance_coupling", "planning_horizon",
        "habit_strength", "goal_directed_strength", "correction_rate",
        "initiation_threshold",
    ]
    return {name: {key: getattr(params, key) for key in keys} for name, params in stages.items()}


def compile_results(results, stages: dict[str, DevelopmentalParams]) -> dict:
    """Aggregate trial results into summary stats and best-trial trajectories.

    Args:
        results: list of TrialResult from run_simulation()
        stages: dict of {name: DevelopmentalParams} used in the simulation

    Returns:
        dict with keys "summary", "trajectories", "developmental_params",
        "stages", "tasks" — ready for JSON serialisation.
    """

    groups = group_results(results)
    task_names = sorted({task for _, task in groups})
    return {
        "summary": {
            f"{stage}_{task}": summarise_group(stage, task, trials)
            for (stage, task), trials in groups.items()
        },
        "trajectories": {
            f"{stage}_{task}": best_trial_trajectory(trials)
            for (stage, task), trials in groups.items()
        },
        "developmental_params": dev_params_as_dict(stages),
        "stages": list(stages),
        "tasks": task_names,
    }
