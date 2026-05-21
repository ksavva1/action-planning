"""Core gaze planning model.

The model is intentionally small, but each layer maps to a psychological claim:
looking controls what can be sampled, working memory determines what survives
across fixations, affordances translate perception into action options, and the
motor layer blends goal-directed control with a developmentally weaker or
stronger translate-first habit.
"""

from dataclasses import dataclass, field

import numpy as np

try:
    from .affordance_matrices import AFFORDANCE_MATRIX_VARIANTS, get_affordance_matrix
    from .model_config import (
        DEVELOPMENTAL_STAGES,
        TASKS,
        DevelopmentalParams,
        TaskConfig,
        TimestepRecord,
        TrialResult,
    )
except ImportError:  # Allows `python gaze_model/planning_cascade_model.py`.
    from affordance_matrices import AFFORDANCE_MATRIX_VARIANTS, get_affordance_matrix
    from model_config import (
        DEVELOPMENTAL_STAGES,
        TASKS,
        DevelopmentalParams,
        TaskConfig,
        TimestepRecord,
        TrialResult,
    )

__all__ = [
    "DevelopmentalParams", "DEVELOPMENTAL_STAGES", "TaskConfig", "TASKS",
    "TimestepRecord", "TrialResult", "GazeState", "WorkingMemoryState",
    "AffordanceWeights", "MotorWeights", "HabitState", "CorrectionState",
    "AFFORDANCE_MATRIX_VARIANTS", "run_trial",
]

ACTION_BIAS = np.array([.15, .1, .05, .2])
# Affordances mostly map to matching motor dimensions, with translate feeding
# x/y movement. This keeps the motor layer interpretable rather than learned.
MOTOR_BASE = np.array([[.9, 0, 0, 0], [0, .9, 0, 0], [0, 0, .9, 0], [.5, .5, 0, 0]])


@dataclass
class GazeState:
    """Mutable fixation state for one trial."""

    current_fixation: str = "object"
    dwell_time: int = 0
    switch_count: int = 0
    object_fixation_count: int = 0
    target_fixation_count: int = 0
    fixation_history: list[str] = field(default_factory=list)


@dataclass
class WorkingMemoryState:
    """Feature values and trace strengths for object, target, and relation."""

    memory_buffer: np.ndarray = field(default_factory=lambda: np.zeros(11))
    trace_strength: np.ndarray = field(default_factory=lambda: np.zeros(11))


@dataclass
class AffordanceWeights:
    """Per-trial percept-to-affordance weights with small random jitter."""

    weight_matrix: np.ndarray
    action_bias: np.ndarray


@dataclass
class MotorWeights:
    """Per-trial affordance-to-motor weights with small random jitter."""

    weight_matrix: np.ndarray


@dataclass
class HabitState:
    """Tracks progression through the habitual movement sequence."""

    step_count: int = 0


@dataclass
class CorrectionState:
    """Stores recent motor errors so feedback can be delayed."""

    error_history: list[np.ndarray] = field(default_factory=list)


def build_affordance_weights(params: DevelopmentalParams, rng: np.random.Generator) -> AffordanceWeights:
    """Scale the selected affordance matrix by developmental coupling and add noise."""

    # The variant is selected before coupling/noise so experiments isolate the
    # structure of the mapping while keeping developmental scaling unchanged.
    base_matrix = get_affordance_matrix(params.affordance_matrix_variant)
    matrix = base_matrix * params.affordance_coupling
    matrix = np.clip(matrix + rng.normal(0, .03, base_matrix.shape), 0, 1)
    return AffordanceWeights(matrix, ACTION_BIAS)


def build_motor_weights(params: DevelopmentalParams, rng: np.random.Generator) -> MotorWeights:
    """Return the base motor mapping with trial-level execution variability."""

    return MotorWeights(MOTOR_BASE + rng.normal(0, .02, MOTOR_BASE.shape))


def step_gaze(gaze: GazeState, params: DevelopmentalParams, rng: np.random.Generator) -> str:
    """
    Advance fixation by one timestep and return "object" or "target".

    The dwell-time hazard makes switches unlikely immediately after a fixation
    but increasingly likely as processing time accumulates. Developmental stages
    differ in how often they leave the currently fixated entity and how strongly
    they bias attention toward the target slot.
    """

    gaze.dwell_time += 1
    hazard = 1.0 - np.exp(-gaze.dwell_time / params.fixation_duration_mean)

    if rng.random() < params.gaze_switch_rate * hazard:
        if gaze.current_fixation == "object":
            # Looking away from the manipulated object means checking the goal.
            gaze.current_fixation = "target"
        else:
            # Older profiles are more likely to keep gaze target-oriented.
            gaze.current_fixation = "object" if rng.random() < (1 - params.target_bias) else "target"
        gaze.dwell_time = 0
        gaze.switch_count += 1

    gaze.fixation_history.append(gaze.current_fixation)
    if gaze.current_fixation == "object":
        gaze.object_fixation_count += 1
    else:
        gaze.target_fixation_count += 1
    return gaze.current_fixation


def sample_percept(
    params: DevelopmentalParams,
    object_state: np.ndarray,
    target_state: np.ndarray,
    gaze_target: str,
    both_recently_fixated: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample noisy feature values from the fixated entity.

    Features are laid out as object [0:5], target [5:8], and relation [8:11].
    Relation features are only sampled when both entities are still recently
    active, modelling the extra difficulty of comparing spatially separated
    object and target representations.
    """

    percept = np.zeros(11)
    sampled = np.zeros(11)

    if gaze_target == "object":
        mask = rng.random(5) < params.sampling_rate
        acuity = np.array([params.location_acuity] * 2 + [params.orientation_acuity] * 3)
        percept[:5] = (object_state[:5] + rng.normal(0, params.perceptual_noise, 5) * (1 - acuity)) * mask
        sampled[:5] = mask.astype(float)
    else:
        mask = rng.random(3) < params.sampling_rate
        acuity = np.array([params.location_acuity] * 2 + [params.orientation_acuity])
        percept[5:8] = (target_state[:3] + rng.normal(0, params.perceptual_noise, 3) * (1 - acuity)) * mask
        sampled[5:8] = mask.astype(float)

    if both_recently_fixated and rng.random() < params.simultaneous_rate:
        # Relational perception is the model's bridge from looking-at-things to
        # comparing-things, which is why it has its own later-developing rate.
        mask = rng.random(3) < params.sampling_rate
        acuity = np.array([params.relation_acuity] * 3)
        relation = target_state[:3] - object_state[:3]
        percept[8:11] = (relation + rng.normal(0, params.perceptual_noise, 3) * (1 - acuity)) * mask
        sampled[8:11] = mask.astype(float)

    return percept, sampled


def update_wm(
    wm: WorkingMemoryState,
    params: DevelopmentalParams,
    percept: np.ndarray,
    sampled: np.ndarray,
    gaze_target: str,
) -> np.ndarray:
    """
    Integrate percepts into working memory and return trace-weighted values.

    Attended features decay slowly, unattended features decay faster, and
    relational features decay at the unattended rate because they depend on
    maintaining two separately sampled entities. Capacity pressure weakens the
    least active traces when too many features are being held at once.
    """

    decay = np.full(11, params.wm_decay)
    decay[5:8] = params.wm_unfixated_decay if gaze_target == "object" else params.wm_decay
    decay[:5] = params.wm_decay if gaze_target == "object" else params.wm_unfixated_decay
    decay[8:11] = params.wm_unfixated_decay
    wm.trace_strength *= 1 - decay

    for index in np.flatnonzero(sampled > .5):
        # Existing traces anchor new noisy samples, approximating a simple
        # evidence-integration process rather than full replacement.
        if wm.trace_strength[index] > .1:
            wm.memory_buffer[index] = .4 * wm.memory_buffer[index] + .6 * percept[index]
        else:
            wm.memory_buffer[index] = percept[index]
        wm.trace_strength[index] = min(1.0, wm.trace_strength[index] + .5)

    if np.sum(wm.trace_strength > .1) > params.wm_capacity:
        # Capacity is implemented as competition between traces, matching the
        # idea that younger agents cannot keep all task dimensions equally active.
        threshold = np.sort(wm.trace_strength)[::-1][params.wm_capacity]
        wm.trace_strength[wm.trace_strength < threshold] *= .3

    return wm.memory_buffer * wm.trace_strength


def estimate_affordances(
    weights: AffordanceWeights,
    params: DevelopmentalParams,
    working_memory: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Map working memory onto reach, grasp, rotate, and translate affordances.

    The sigmoid turns weak, uncertain memory into low action activation while
    allowing strong memory traces to produce clear action candidates.
    """

    activations = np.dot(working_memory, weights.weight_matrix) + weights.action_bias * .1
    activations += rng.normal(0, params.affordance_noise, 4)
    return 1 / (1 + np.exp(-5 * (activations - .3)))


def plan_motor_command(
    weights: MotorWeights,
    params: DevelopmentalParams,
    affordances: np.ndarray,
    goal_motor_state: np.ndarray,
    current_motor_state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blend affordance-driven action with direct goal-error control.

    A short planning horizon keeps movement mostly reactive. Longer horizons
    increase the direct goal-error term, modelling more anticipatory control.
    Commands are reduced near the goal to avoid repeated overshoot.
    """

    affordance_command = np.dot(affordances, weights.weight_matrix)
    goal_error = goal_motor_state[:4] - current_motor_state[:4]
    planning_weight = min(1.0, params.planning_horizon / 6.0)
    command = (
        (1 - planning_weight) * affordance_command
        + planning_weight * goal_error * .3
        + rng.normal(0, params.motor_noise, 4)
    )
    return np.clip(command * min(1.0, np.sqrt(np.sum(goal_error[:3] ** 2)) / .5), -1, 1)


def blend_habit(habit: HabitState, params: DevelopmentalParams, command: np.ndarray) -> np.ndarray:
    """
    Blend a translate-first habit with the goal-directed command.

    Early in a trial the habit pushes the object laterally before rotation; later
    it shifts toward rotation. The habit fades over time so mature, goal-directed
    control can take over even in high-habit profiles.
    """

    habit.step_count += 1
    phase_length = int(4 + 12 * params.habit_strength)
    habitual = np.array([0.7, 0.7, 0.0, 0.2]) if habit.step_count < phase_length else np.array([0.0, 0.0, 0.8, 0.5])
    fade = max(0.0, 1.0 - habit.step_count / (phase_length * 3))
    habit_weight = params.habit_strength * fade
    goal_weight = params.goal_directed_strength + params.habit_strength * (1 - fade)
    total = habit_weight + goal_weight
    return (habit_weight * habitual + goal_weight * command) / total


def apply_correction(
    correction: CorrectionState,
    params: DevelopmentalParams,
    current_motor_state: np.ndarray,
    goal_motor_state: np.ndarray,
    current_time: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply delayed online correction from recent motor error.

    The delay represents immature sensorimotor feedback: less mature profiles
    keep acting on older error estimates, while mature profiles can correct more
    immediately during the movement.
    """

    current_error = goal_motor_state[:4] - current_motor_state[:4]
    correction.error_history.append(current_error.copy())
    if current_time < params.correction_delay:
        return np.zeros(4)

    delayed_error = correction.error_history[len(correction.error_history) - 1 - params.correction_delay]
    command = params.correction_rate * delayed_error + rng.normal(0, params.motor_noise * .5, 4)
    return command * min(1.0, np.sqrt(np.sum(current_error[:3] ** 2)) / .4)


def both_entities_recently_fixated(current_time: int, object_time: int, target_time: int) -> bool:
    """Return True when object and target traces are both warm enough to compare."""

    return current_time - object_time <= 3 and current_time - target_time <= 3


def extract_wm_info_strengths(strengths: np.ndarray) -> tuple[float, float, float]:
    """Return mean trace strengths for object, target, and relation features."""

    return float(np.mean(strengths[:5])), float(np.mean(strengths[5:8])), float(np.mean(strengths[8:11]))


def compute_near_goal_command(command: np.ndarray, object_x: float, object_y: float, object_angle: float, task: TaskConfig) -> np.ndarray:
    """
    Add fine-grained proportional control as the object nears the goal.

    This models a shift from coarse, ballistic movement to local error correction
    during insertion, where small position and angle differences matter most.
    """

    dx, dy, da = task.goal_x - object_x, task.goal_y - object_y, task.goal_angle - object_angle
    distance = np.sqrt(dx ** 2 + dy ** 2 + da ** 2)
    if distance < .6:
        closeness = 1.0 - min(distance / .6, 1.0)
        command = command * (1 - closeness) + np.array([dx, dy, da, 0]) * 2.0 * closeness
    return np.clip(command * min(1.0, distance * 2.5), -1, 1)


def apply_motor_command(object_x: float, object_y: float, object_angle: float, command: np.ndarray) -> tuple[float, float, float]:
    """Advance object pose by one fixed-size motor step."""

    return object_x + command[0] * .10, object_y + command[1] * .10, object_angle + command[2] * .10


def update_onset_times(command, movement_started, current_time, rotation_onset, translation_onset, max_timesteps):
    """Record first meaningful rotation and translation timesteps."""

    if movement_started and abs(command[2]) > .1 and rotation_onset == max_timesteps:
        rotation_onset = current_time
    if movement_started and (abs(command[0]) > .1 or abs(command[1]) > .1) and translation_onset == max_timesteps:
        translation_onset = current_time
    return rotation_onset, translation_onset


def compute_path_efficiency(trajectory: list[TimestepRecord], task: TaskConfig) -> float:
    """Return optimal straight-line path length divided by actual path length."""

    optimal = np.sqrt(
        (task.goal_x - task.start_x) ** 2
        + (task.goal_y - task.start_y) ** 2
        + (task.goal_angle - task.start_angle) ** 2
    )
    actual = sum(
        np.sqrt(
            (trajectory[i].obj_x - trajectory[i - 1].obj_x) ** 2
            + (trajectory[i].obj_y - trajectory[i - 1].obj_y) ** 2
            + (trajectory[i].obj_angle - trajectory[i - 1].obj_angle) ** 2
        )
        for i in range(1, len(trajectory))
    )
    return min(optimal / max(actual, .01), 1.0)


def run_trial(params: DevelopmentalParams, task: TaskConfig, seed: int = 42, trial_id: int = 0) -> TrialResult:
    """
    Simulate one object-manipulation trial.

    Each timestep runs the cascade: gaze, perceptual sampling, working-memory
    update, movement initiation, affordance estimation, motor planning, habit
    blending, delayed correction, execution, and success checking.
    """

    rng = np.random.default_rng(seed)
    affordance_weights = build_affordance_weights(params, rng)
    motor_weights = build_motor_weights(params, rng)
    gaze = GazeState()
    wm = WorkingMemoryState()
    habit = HabitState()
    correction = CorrectionState()

    object_x, object_y, object_angle = task.start_x, task.start_y, task.start_angle
    target_state = np.array([task.goal_x, task.goal_y, task.goal_angle])
    goal_motor_state = np.array([task.goal_x, task.goal_y, task.goal_angle, 0.5])

    trajectory = []
    movement_started = False
    movement_onset = rotation_onset = translation_onset = task.max_timesteps
    last_object_fixation = last_target_fixation = -10

    for current_time in range(task.max_timesteps):
        position_error = np.sqrt((object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2)
        angle_error = abs(object_angle - task.goal_angle)

        # Gaze gates perception: the model can only sample detailed information
        # from the object or the target currently being fixated.
        gaze_target = step_gaze(gaze, params, rng)
        if gaze_target == "object":
            last_object_fixation = current_time
        else:
            last_target_fixation = current_time

        object_state = np.array([object_x, object_y, object_angle, task.obj_width, task.obj_height])
        percept, sampled = sample_percept(
            params,
            object_state,
            target_state,
            gaze_target,
            both_entities_recently_fixated(current_time, last_object_fixation, last_target_fixation),
            rng,
        )
        working_memory = update_wm(wm, params, percept, sampled, gaze_target)
        strengths = wm.trace_strength.copy()
        object_info, target_info, relational_info = extract_wm_info_strengths(strengths)

        # Movement starts only once enough information has accumulated. This is
        # the simulated pause before action commitment.
        if not movement_started and np.mean(strengths) >= params.initiation_threshold:
            movement_started = True
            movement_onset = current_time

        affordances = estimate_affordances(affordance_weights, params, working_memory, rng)
        current_motor_state = np.array([object_x, object_y, object_angle, 0.3])
        # Motor output combines the current perceptual action options with a
        # habitual sequence and delayed correction, so failures can arise from
        # either poor information or immature control.
        command = blend_habit(
            habit,
            params,
            plan_motor_command(motor_weights, params, affordances, goal_motor_state, current_motor_state, rng),
        )
        command += apply_correction(correction, params, current_motor_state, goal_motor_state, current_time, rng)
        final_command = compute_near_goal_command(command, object_x, object_y, object_angle, task) if movement_started else np.zeros(4)

        object_x, object_y, object_angle = apply_motor_command(object_x, object_y, object_angle, final_command)
        rotation_onset, translation_onset = update_onset_times(
            final_command, movement_started, current_time, rotation_onset, translation_onset, task.max_timesteps
        )

        trajectory.append(TimestepRecord(
            current_time, object_x, object_y, object_angle, gaze_target, movement_started,
            object_info, target_info, relational_info, position_error, angle_error, gaze.switch_count,
        ))
        if position_error < task.position_tolerance and angle_error < task.angle_tolerance:
            break

    final_position_error = np.sqrt((object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2)
    final_angle_error = abs(object_angle - task.goal_angle)
    total_fixations = len(gaze.fixation_history)

    return TrialResult(
        params.name,
        task.name,
        trial_id,
        final_position_error < task.position_tolerance and final_angle_error < task.angle_tolerance,
        len(trajectory),
        final_position_error,
        final_angle_error,
        trajectory,
        movement_onset,
        rotation_onset,
        translation_onset,
        compute_path_efficiency(trajectory, task),
        gaze.switch_count,
        gaze.object_fixation_count / max(total_fixations, 1),
        gaze.target_fixation_count / max(total_fixations, 1),
        gaze.fixation_history,
    )
