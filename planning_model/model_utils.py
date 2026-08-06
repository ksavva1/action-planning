"""Configuration containers, seeding policy, and simulation summaries."""

import json
import zlib
from dataclasses import dataclass, field
import numpy as np

SEED_POLICIES = ("common", "independent")


def condition_key(name: str | int) -> int:
    """Map a condition label onto a stable non-negative integer.

    Args:
        name: condition label, or an integer already.
    Returns:
        A stable non-negative integer derived from ``name``.
    """

    if isinstance(name, (int, np.integer)):
        return int(name)
    return int(zlib.crc32(str(name).encode("utf-8")))


def make_rng(
    base_seed: int,
    condition: str | int,
    trial_index: int,
    policy: str = "common",
) -> np.random.Generator:
    """
    Build the generator for one trial under the requested seeding policy.

    Args:
        base_seed: the experiment's base seed.
        condition: label identifying the condition (profile, parameter value,
            matrix variant, or a composite of these with the task).
        trial_index: index of the trial within its condition.
        policy: "common" or "independent"; see the module notes above.

    Returns:
        A NumPy Generator.
    """

    if policy == "common":
        return np.random.default_rng(base_seed + trial_index)
    if policy == "independent":
        sequence = np.random.SeedSequence(
            [int(base_seed), condition_key(condition), int(trial_index)]
        )
        return np.random.default_rng(sequence)
    raise ValueError(f"unknown seed policy {policy!r}; expected one of {SEED_POLICIES}")


def resolve_rng(rng: np.random.Generator | None, seed: int) -> np.random.Generator:
    """Return ``rng`` if supplied, otherwise a generator seeded with ``seed``.

    Args:
        rng: an existing generator, or None.
        seed: seed to use when ``rng`` is None.
    Returns:
        A NumPy Generator.
    """

    return rng if rng is not None else np.random.default_rng(seed)


@dataclass
class DevelopmentalParams:
    """
    Parameters for one developmental profile in the planning cascade model.

    The profiles are not ages directly; they are ordered profiles of increasing
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
        affordance_jitter_sd: per-trial Gaussian jitter on the scaled affordance
            matrix. Exposed as a parameter so that its size relative to the
            differences between matrix variants can be varied and reported.
        motor_jitter_sd: per-trial Gaussian jitter on the base motor matrix.
    """

    name: str = "default"

    # Gaze control parameters
    gaze_switch_rate: float = 0.30
    fixation_duration_mean: float = 3.0
    target_bias: float = 0.40
    simultaneous_rate: float = 0.15

    # Perceptual parameters
    sampling_rate: float = 0.40
    perceptual_noise: float = 0.30
    location_acuity: float = 0.85
    orientation_acuity: float = 0.40
    relation_acuity: float = 0.15

    # Working memory parameters
    wm_capacity: int = 3
    wm_decay: float = 0.12
    wm_unfixated_decay: float = 0.28

    # Affordance parameters
    affordance_coupling: float = 0.40
    affordance_noise: float = 0.25

    # Motor planning parameters
    planning_horizon: int = 2
    motor_noise: float = 0.25

    # Habit vs. goal-directed control balance
    habit_strength: float = 0.55
    goal_directed_strength: float = 0.45

    # Online error correction parameters
    correction_rate: float = 0.12
    correction_delay: int = 2

    # Movement initiation threshold
    initiation_threshold: float = 0.35

    # Affordance matrix experiment parameters
    affordance_matrix_variant: str = "baseline"
    affordance_jitter_sd: float = 0.03
    motor_jitter_sd: float = 0.02


@dataclass
class TaskConfig:
    """
    Defines a 2D object manipulation task with start and goal poses.

    The task geometry is 2D. The agent must move the object from its
    starting pose to the goal pose within the specified tolerances.

    Attributes:
        name: identifier string for this task
        start_x, start_y, start_angle: initial position and orientation of object
        goal_x, goal_y, goal_angle: target position and required orientation
        obj_width, obj_height: lateral dimensions of the manipulated object
        position_tolerance: maximum Euclidean distance from goal centre for success
        angle_tolerance: maximum absolute angular difference from goal angle for
            success (radians)
        angular_symmetry: order of the object's rotational symmetry group, used
            when computing angular error. ``None`` treats the object as
            directional with an unbounded angle, which is the behaviour of the
            original implementation and the default for the main results. ``2``
            treats it as an unmarked rectangle, for which 0 and π are equivalent
            end states, and ``4`` as a square. Used in the symmetry robustness
            check.
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
    angular_symmetry: int | None = None
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
        success: True if the object reached the goal within both tolerances
        timesteps_used: number of timesteps before the trial ended
        timed_out: True if the trial ran to the step limit without succeeding.
            This is the exact complement of ``success``, and is stored so that
            the time-out rate can be reported without recomputing it.
        final_pos_error: Euclidean position error at trial end
        final_angle_error: absolute angle error at trial end
        trajectory: list of TimestepRecord, one per timestep
        movement_initiated: whether the initiation threshold was ever crossed
        movement_onset: timestep of first motor output, or None if never
        rotation_onset: timestep of first rotation command, or None if never
        translation_onset: timestep of first translation command, or None
        efficiency: optimal over actual path length in the combined
            position-orientation space, capped at 1, or None if the object
            never moved
        translational_efficiency: the same ratio for the x-y path only
        rotational_efficiency: the same ratio for angular travel only
        total_gaze_switches: total object-to-target fixation switches
        object_fixation_pct: fraction of timesteps fixating the object. This is
            the exact complement of ``target_fixation_pct``; only one of the two
            should be reported.
        target_fixation_pct: fraction of timesteps fixating the target
        pre_movement_target_fixation_pct: fraction of pre-onset timesteps
            fixating the target, or None if movement began immediately. This is
            the measure that maps onto anticipatory looking; whole-trial target
            fixation does not.
        time_to_first_target_fixation: timestep of the first target fixation,
            or None if the target was never fixated
        gaze_history: full sequence of "object"/"target" strings, one per timestep
    """

    params_name: str
    task_name: str
    trial_id: int
    success: bool
    timesteps_used: int
    timed_out: bool
    final_pos_error: float
    final_angle_error: float
    trajectory: list[TimestepRecord]
    movement_initiated: bool
    movement_onset: int | None
    rotation_onset: int | None
    translation_onset: int | None
    efficiency: float | None
    translational_efficiency: float | None
    rotational_efficiency: float | None
    total_gaze_switches: int
    object_fixation_pct: float
    target_fixation_pct: float
    pre_movement_target_fixation_pct: float | None
    time_to_first_target_fixation: int | None
    gaze_history: list[str] = field(default_factory=list)

    @property
    def translate_before_rotate(self) -> bool | None:
        """Whether the first translation command preceded the first rotation command.

        ``None`` where neither command ever occurred, so that trials in which
        the agent never moved are excluded from the rate rather than counted as
        rotation-first.

        Returns:
            True if translation began first, False if rotation began first, or
            None if neither occurred.
        """

        if self.translation_onset is None and self.rotation_onset is None:
            return None
        if self.rotation_onset is None:
            return True
        if self.translation_onset is None:
            return False
        return self.translation_onset < self.rotation_onset

    @property
    def time_to_success(self) -> int | None:
        """Timesteps taken on successful trials, and ``None`` on failures.

        Returns:
            Timesteps used if the trial succeeded, else None.
        """

        return self.timesteps_used if self.success else None


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalar and array values to Python types."""

    def default(self, obj):
        """Convert NumPy objects before falling back to the parent encoder.

        Args:
            obj: object being serialised.
        Returns:
            A JSON-serialisable Python value.
        """

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
    stages: dict,
    task_configs: dict,
    n_trials: int = 200,
    seed: int = 42,
    seed_policy: str = "common",
) -> list[TrialResult]:
    """
    Run all requested profile x task x trial combinations.

    Args:
        stages: dict mapping name → DevelopmentalParams.
        task_configs: dict mapping name → TaskConfig.
        n_trials: number of trials per profile/task pair.
        seed: base seed.
        seed_policy: "common" reproduces the original scheme, in which trial i
            uses ``seed + i`` in every condition. "independent" gives each
            (condition, trial) pair its own spawned stream.

    Returns:
        Flat list of TrialResult objects.
    """

    from planning_cascade_model import run_trial

    results: list[TrialResult] = []
    for profile_name, params in stages.items():
        for task_name, task in task_configs.items():
            label = f"{profile_name}|{task_name}"
            for trial in range(n_trials):
                rng = make_rng(seed, label, trial, seed_policy)
                results.append(run_trial(params, task, trial_id=trial, rng=rng))
    return results


def group_results(
    results: list[TrialResult],
) -> dict[tuple[str, str], list[TrialResult]]:
    """Group a flat list of TrialResult objects by (params_name, task_name).

    Args:
        results: flat list of TrialResult objects.
    Returns:
        Mapping from (params_name, task_name) to its list of trials.
    """

    groups = {}
    for result in results:
        groups.setdefault((result.params_name, result.task_name), []).append(result)
    return groups


def _defined(values) -> np.ndarray:
    """Drop None entries, returning a float array of the rest.

    Args:
        values: iterable possibly containing None entries.
    Returns:
        NumPy float array of the non-None values.
    """

    return np.array([value for value in values if value is not None], dtype=float)


def _mean_sd(values, n_total: int) -> tuple[float, float, float, float]:
    """Mean, SD, standard error and censoring rate for a possibly censored measure.

    Args:
        values: measurements, with None marking censored trials.
        n_total: total number of trials, including censored ones.
    Returns:
        Tuple of (mean, sd, se, censored_rate).
    """

    defined_values = _defined(values)
    n_defined = len(defined_values)
    censored = 1.0 - n_defined / n_total if n_total else float("nan")
    if n_defined == 0:
        return float("nan"), float("nan"), float("nan"), censored
    sd = float(np.std(defined_values, ddof=1)) if n_defined > 1 else 0.0
    return float(np.mean(defined_values)), sd, sd / np.sqrt(n_defined), censored


def summarise_group(stage: str, task: str, trials: list[TrialResult]) -> dict:
    """Compute aggregated statistics for one (profile, task) group of trials.

    Every continuous measure is returned with its standard deviation and
    standard error, and every censored measure with the proportion of trials on
    which it was undefined.

    Args:
        stage: profile label to attach to the summary.
        task: task name to attach to the summary.
        trials: list of TrialResult objects for this group.

    Returns:
        Dict of aggregated statistics keyed by measure name.
    """

    from analysis import wilson_interval

    n = len(trials)
    successes = int(sum(trial.success for trial in trials))
    low, high = wilson_interval(successes, n)

    summary = {
        "stage": stage,
        "task": task,
        "n_trials": n,
        "successes": successes,
        "success_rate": successes / n if n else float("nan"),
        "success_ci_low": low,
        "success_ci_high": high,
        "timeout_rate": float(np.mean([trial.timed_out for trial in trials])),
        "no_onset_rate": float(
            np.mean([not trial.movement_initiated for trial in trials])
        ),
    }

    censored_metrics = {
        "efficiency": [trial.efficiency for trial in trials],
        "translational_efficiency": [
            trial.translational_efficiency for trial in trials
        ],
        "rotational_efficiency": [trial.rotational_efficiency for trial in trials],
        "movement_onset": [trial.movement_onset for trial in trials],
        "time_to_success": [trial.time_to_success for trial in trials],
        "pre_movement_target_fixation": [
            trial.pre_movement_target_fixation_pct for trial in trials
        ],
        "time_to_first_target_fixation": [
            trial.time_to_first_target_fixation for trial in trials
        ],
    }
    complete_metrics = {
        "timesteps": [trial.timesteps_used for trial in trials],
        "pos_error": [trial.final_pos_error for trial in trials],
        "angle_error": [trial.final_angle_error for trial in trials],
        "gaze_switches": [trial.total_gaze_switches for trial in trials],
        "target_fixation": [trial.target_fixation_pct for trial in trials],
    }

    all_metrics = {**complete_metrics, **censored_metrics}
    for key, values in all_metrics.items():
        mean, sd, se, censored = _mean_sd(values, n)
        summary[f"mean_{key}"] = mean
        summary[f"sd_{key}"] = sd
        summary[f"se_{key}"] = se
        if key in censored_metrics:
            summary[f"censored_{key}"] = censored

    tbr_defined = [
        value
        for value in (trial.translate_before_rotate for trial in trials)
        if value is not None
    ]
    summary["translate_before_rotate_rate"] = (
        float(np.mean(tbr_defined)) if tbr_defined else float("nan")
    )
    summary["translate_before_rotate_n"] = len(tbr_defined)

    return summary


def best_trial_trajectory(trials):
    """Return compact trajectory data for the trial closest to the goal.

    The best trial is chosen by final position plus angle error so animations and
    dashboards show a representative successful attempt when available.

    Args:
        trials: list of TrialResult objects to choose from.
    Returns:
        Dict with rounded per-timestep ``steps``, ``success``, and
        ``gaze_history``.
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


def dev_params_as_dict(stages: dict) -> dict:
    """Return serialisable developmental parameters for the given profiles.

    Args:
        stages: mapping from profile name to DevelopmentalParams.
    Returns:
        Mapping from profile name to a dict of its key parameter values.
    """

    keys = [
        "gaze_switch_rate",
        "fixation_duration_mean",
        "target_bias",
        "simultaneous_rate",
        "sampling_rate",
        "perceptual_noise",
        "location_acuity",
        "orientation_acuity",
        "relation_acuity",
        "wm_capacity",
        "wm_decay",
        "wm_unfixated_decay",
        "affordance_matrix_variant",
        "affordance_coupling",
        "planning_horizon",
        "habit_strength",
        "goal_directed_strength",
        "correction_rate",
        "initiation_threshold",
    ]
    return {
        name: {key: getattr(params, key) for key in keys}
        for name, params in stages.items()
    }


def compile_results(results, stages: dict) -> dict:
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
