"""Core gaze-planning model."""

import math
from dataclasses import dataclass, field

import numpy as np

from affordance_matrices import AFFORDANCE_MATRIX_VARIANTS, get_affordance_matrix
from model_utils import (
    DevelopmentalParams,
    TaskConfig,
    TimestepRecord,
    TrialResult,
    resolve_rng,
)

__all__ = [
    "DevelopmentalParams",
    "TaskConfig",
    "TimestepRecord",
    "TrialResult",
    "GazeState",
    "WorkingMemoryState",
    "AffordanceWeights",
    "MotorWeights",
    "HabitState",
    "CorrectionState",
    "AFFORDANCE_MATRIX_VARIANTS",
    "run_trial",
    "angular_error",
]

OBJECT_FEATURES = slice(0, 5)
TARGET_FEATURES = slice(5, 8)
RELATION_FEATURES = slice(8, 11)
FEATURE_COUNT = 11

ACTION_BIAS = np.array([0.15, 0.1, 0.05, 0.2])
# 4 affordances (reach, grasp, rotate, translate) to 3 motor dimensions (x, y, angle).
MOTOR_BASE = np.array(
    [
        [0.9, 0, 0],  # reach -> x
        [0, 0.9, 0],  # grasp -> y
        [0, 0, 0.9],  # rotate -> angle
        [0.5, 0.5, 0],  # translate -> x and y
    ]
)

# Total path length below which a trial counts as having produced no movement.
MOVEMENT_EPSILON = 1e-9

# Minimum command magnitude counted as translation or rotation onset.
ONSET_COMMAND_THRESHOLD = 0.1


def angular_error(
    angle: float, goal_angle: float, symmetry: int | None = None
) -> float:
    """
    Absolute angular distance from ``angle`` to ``goal_angle``.

    Args:
        angle: current object orientation in radians.
        goal_angle: required orientation in radians.
        symmetry: order of the object's rotational symmetry group.
    Returns:
        Absolute angular error in radians.
    """

    if symmetry is None:
        return abs(angle - goal_angle)

    period = 2 * math.pi / symmetry
    difference = (angle - goal_angle) % period
    return float(min(difference, period - difference))


@dataclass
class GazeState:
    """Mutable fixation state for one trial."""

    current_fixation: str = "object"
    dwell_time: int = 0
    switch_count: int = 0
    object_fixation_count: int = 0
    target_fixation_count: int = 0
    pre_movement_steps: int = 0
    pre_movement_target_count: int = 0
    first_target_fixation: int | None = None
    fixation_history: list[str] = field(default_factory=list)


@dataclass
class WorkingMemoryState:
    """Feature values and trace strengths for object, target, and relational features.

    Buffer layout (11 features):
      [0:5]   visual object features  (x, y, angle, width, height)
      [5:8]   visual target features  (goal x, y, angle)
      [8:11]  relational features     (rel_dx, rel_dy, rel_d_angle)
    """

    memory_buffer: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))
    trace_strength: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))


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


def _trace_mean(wm: WorkingMemoryState, feature_slice: slice) -> float:
    """Mean trace strength for one feature block."""

    return float(np.mean(wm.trace_strength[feature_slice]))


def _pose_vector(x: float, y: float, angle: float) -> np.ndarray:
    """Return the 3D pose vector used by motor-control helpers."""

    return np.array([x, y, angle])


def _position_error(x: float, y: float, task: TaskConfig) -> float:
    """Euclidean distance from an object position to the task goal."""

    return math.hypot(x - task.goal_x, y - task.goal_y)


def step_gaze(
    gaze: GazeState,
    params: DevelopmentalParams,
    wm: WorkingMemoryState,
    movement_started: bool,
    current_time: int,
    rng: np.random.Generator,
) -> str:
    """
    Advance fixation by one timestep and return "object" or "target".

    The base switch probability (gaze_switch_rate x dwell-time hazard) is
    dynamically modulated each timestep by four cognitive factors drawn from
    the current working-memory state and developmental parameters:

    1. WM saturation - when the trace for the currently fixated item is
       already strong, there is little more to gain. The switch rate increases
       proportionally (diminishing returns on continued fixation).

    2. Decay urgency - when the unfixated item's trace has fallen below a
       threshold, the need to refresh that representation boosts the rate of
       switching back.

    3. Perceptual noise / acuity - higher noise or lower mean acuity
       stretches the effective fixation duration so that more dwell time is
       needed before the hazard reaches its peak, slowing voluntary switching.

    4. Planning horizon - a longer lookahead shifts the effective target
       bias upward, making proactive target fixations more likely when the
       agent does decide to switch.

    5. Pre-movement scanning - before motor output begins, a high
       initiation threshold combined with a low current WM strength creates an
       information gap that inflates the scan rate; the boost decays to zero as
       WM strength approaches the threshold.

    Args:
        gaze: mutable fixation state, updated in place.
        params: developmental parameters governing switch behaviour.
        wm: current working memory state.
        movement_started: whether motor output has begun.
        current_time: current timestep index.
        rng: random generator for the switch decision.

    Returns:
        "object" or "target", the fixation after this timestep.
    """

    gaze.dwell_time += 1

    # WM driven urgency
    if gaze.current_fixation == "object":
        fixated_trace = _trace_mean(wm, OBJECT_FEATURES)
        unfixated_trace = _trace_mean(wm, TARGET_FEATURES)
    else:
        fixated_trace = _trace_mean(wm, TARGET_FEATURES)
        unfixated_trace = _trace_mean(wm, OBJECT_FEATURES)

    # Saturation: diminishing return from staying on a well-sampled fixation
    saturation_drive = fixated_trace
    # Decay urgency: need to refresh the other item when its trace has faded
    decay_urgency = max(0.0, (0.5 - unfixated_trace) * 2.0)

    wm_multiplier = 1.0 + 0.35 * saturation_drive + 0.45 * decay_urgency

    # Perceptual noise / acuity: stretch fixation duration
    acuity_mean = (params.location_acuity + params.orientation_acuity) / 2.0
    noise_load = params.perceptual_noise * (1.0 - acuity_mean)
    effective_duration = params.fixation_duration_mean * (1.0 + noise_load)

    # Planning horizon: boost target bias (proactive gaze)
    horizon_boost = (params.planning_horizon / 6.0) * 0.20
    effective_target_bias = min(0.95, params.target_bias + horizon_boost)

    # Pre-movement scanning: info-gap drives faster search
    if not movement_started:
        mean_strength = float(np.mean(wm.trace_strength))
        info_gap = max(0.0, params.initiation_threshold - mean_strength)
        threshold_drive = info_gap / max(params.initiation_threshold, 1e-6)
        scan_multiplier = 1.0 + threshold_drive * 0.50
    else:
        scan_multiplier = 1.0

    # Combined switch decision
    hazard = 1.0 - np.exp(-gaze.dwell_time / effective_duration)
    effective_rate = params.gaze_switch_rate * wm_multiplier * scan_multiplier

    if rng.random() < effective_rate * hazard:
        if gaze.current_fixation == "object":
            gaze.current_fixation = "target"
        else:
            gaze.current_fixation = (
                "object" if rng.random() < (1.0 - effective_target_bias) else "target"
            )
        gaze.dwell_time = 0
        gaze.switch_count += 1

    gaze.fixation_history.append(gaze.current_fixation)
    if gaze.current_fixation == "object":
        gaze.object_fixation_count += 1
    else:
        gaze.target_fixation_count += 1
        if gaze.first_target_fixation is None:
            gaze.first_target_fixation = current_time

    if not movement_started:
        gaze.pre_movement_steps += 1
        if gaze.current_fixation == "target":
            gaze.pre_movement_target_count += 1

    return gaze.current_fixation


def sample_percept(
    params: DevelopmentalParams,
    object_state: np.ndarray,
    target_state: np.ndarray,
    gaze_target: str,
    both_recently_fixated: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample noisy visual feature values from the fixated entity.

    Object features fill [0:5], target features fill [5:8], and relational
    features fill [8:11]. Relation features are only sampled when both
    entities are still recently active.

    Args:
        params: developmental parameters governing sampling and noise.
        object_state: current object pose and size, [x, y, angle, width,
            height].
        target_state: goal pose, [x, y, angle].
        gaze_target: "object" or "target", the entity currently fixated.
        both_recently_fixated: whether both entities were fixated recently
            enough to sample relational features.
        rng: random generator for sampling noise and masks.

    Returns:
        Tuple of (percept, sampled), each an 11-D array; ``sampled`` marks
        which entries of ``percept`` were actually sampled this step.
    """

    percept = np.zeros(FEATURE_COUNT)
    sampled = np.zeros(FEATURE_COUNT)

    def sample_block(
        values: np.ndarray, acuity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = rng.random(len(values)) < params.sampling_rate
        noise = rng.normal(0, params.perceptual_noise, len(values)) * (1 - acuity)
        return (values + noise) * mask, mask.astype(float)

    if gaze_target == "object":
        acuity = np.array(
            [params.location_acuity] * 2 + [params.orientation_acuity] * 3
        )
        percept[OBJECT_FEATURES], sampled[OBJECT_FEATURES] = sample_block(
            object_state[OBJECT_FEATURES],
            acuity,
        )
    else:
        acuity = np.array([params.location_acuity] * 2 + [params.orientation_acuity])
        percept[TARGET_FEATURES], sampled[TARGET_FEATURES] = sample_block(
            target_state[:3],
            acuity,
        )

    if both_recently_fixated and rng.random() < params.simultaneous_rate:
        acuity = np.array([params.relation_acuity] * 3)
        relation = target_state[:3] - object_state[:3]
        percept[RELATION_FEATURES], sampled[RELATION_FEATURES] = sample_block(
            relation,
            acuity,
        )

    return percept, sampled


def update_wm(
    wm: WorkingMemoryState,
    params: DevelopmentalParams,
    percept: np.ndarray,
    sampled: np.ndarray,
    gaze_target: str,
) -> np.ndarray:
    """Integrate percepts into working memory and return trace-weighted values.

    Attended features decay slowly, unattended features decay faster, and
    relational features decay at the unattended rate because they depend on
    maintaining two separately sampled entities. Capacity pressure weakens
    the least active traces when too many features are being held at once.

    Args:
        wm: working-memory state, updated in place.
        params: developmental parameters governing decay and capacity.
        percept: 11-D sampled feature values.
        sampled: 11-D mask of which features were sampled this step.
        gaze_target: "object" or "target", the entity currently fixated.

    Returns:
        11-D array of trace-weighted working-memory values.
    """

    decay = np.full(FEATURE_COUNT, params.wm_decay)
    decay[TARGET_FEATURES] = (
        params.wm_unfixated_decay if gaze_target == "object" else params.wm_decay
    )
    decay[OBJECT_FEATURES] = (
        params.wm_decay if gaze_target == "object" else params.wm_unfixated_decay
    )
    decay[RELATION_FEATURES] = params.wm_unfixated_decay
    wm.trace_strength *= 1 - decay

    for index in np.flatnonzero(sampled > 0.5):
        if wm.trace_strength[index] > 0.1:
            wm.memory_buffer[index] = (
                0.4 * wm.memory_buffer[index] + 0.6 * percept[index]
            )
        else:
            wm.memory_buffer[index] = percept[index]
        wm.trace_strength[index] = min(1.0, wm.trace_strength[index] + 0.5)

    if np.sum(wm.trace_strength > 0.1) > params.wm_capacity:
        threshold = np.sort(wm.trace_strength)[::-1][params.wm_capacity]
        wm.trace_strength[wm.trace_strength < threshold] *= 0.3

    return wm.memory_buffer * wm.trace_strength


def estimate_affordances(
    weights: AffordanceWeights,
    params: DevelopmentalParams,
    working_memory: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Map working memory onto reach, grasp, rotate, and translate affordances.

    Args:
        weights: per-trial affordance weight matrix and action bias.
        params: developmental parameters governing affordance noise.
        working_memory: 11-D trace-weighted working-memory values.
        rng: random generator for affordance noise.

    Returns:
        4D array of affordance activations for reach, grasp, rotate,
        translate.
    """

    activations = (
        np.dot(working_memory, weights.weight_matrix) + weights.action_bias * 0.1
    )
    activations += rng.normal(0, params.affordance_noise, 4)
    return 1 / (1 + np.exp(-5 * (activations - 0.3)))


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

    Args:
        weights: per-trial motor weight matrix.
        params: developmental parameters governing planning horizon and noise.
        affordances: 4D affordance activations.
        goal_motor_state: target [x, y, angle].
        current_motor_state: current [x, y, angle].
        rng: random generator for motor noise.

    Returns:
        3D motor command, clipped to [-1, 1].
    """

    affordance_command = np.dot(affordances, weights.weight_matrix)
    goal_error = goal_motor_state - current_motor_state
    planning_weight = min(1.0, params.planning_horizon / 6.0)
    command = (
        (1 - planning_weight) * affordance_command
        + planning_weight * goal_error * 0.3
        + rng.normal(0, params.motor_noise, 3)
    )
    return np.clip(command * min(1.0, np.sqrt(np.sum(goal_error**2)) / 0.5), -1, 1)


def blend_habit(
    habit: HabitState, params: DevelopmentalParams, command: np.ndarray
) -> np.ndarray:
    """Blend a translate-first habit with the goal-directed command.

    Early in a trial the habit pushes the object laterally before rotation, but later
    it shifts toward rotation.

    Args:
        habit: habit progression state, updated in place.
        params: developmental parameters governing habit and goal-directed
            strength.
        command: goal-directed motor command to blend with the habit.

    Returns:
        3D blended motor command.
    """

    habit.step_count += 1
    phase_length = int(4 + 12 * params.habit_strength)
    habitual = (
        np.array([0.7, 0.7, 0.0])
        if habit.step_count < phase_length
        else np.array([0.0, 0.0, 0.8])
    )
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
    """Apply delayed online correction from recent motor error.

    Less mature profiles keep acting on older error estimates,
    while mature profiles can correct more immediately during
    the movement.

    Args:
        correction: correction state holding the error history, updated in
            place.
        params: developmental parameters governing correction rate and delay.
        current_motor_state: current [x, y, angle].
        goal_motor_state: target [x, y, angle].
        current_time: current timestep index.
        rng: random generator for correction noise.

    Returns:
        3D correction command
    """

    current_error = goal_motor_state - current_motor_state
    correction.error_history.append(current_error.copy())
    if current_time < params.correction_delay:
        return np.zeros(3)

    delayed_error = correction.error_history[
        len(correction.error_history) - 1 - params.correction_delay
    ]
    command = params.correction_rate * delayed_error + rng.normal(
        0, params.motor_noise * 0.5, 3
    )
    return command * min(1.0, np.sqrt(np.sum(current_error**2)) / 0.4)


def compute_near_goal_command(
    command: np.ndarray,
    object_x: float,
    object_y: float,
    object_angle: float,
    task: TaskConfig,
) -> np.ndarray:
    """Add fine-grained proportional control as the object nears the goal.

    Args:
        command: motor command to refine.
        object_x, object_y, object_angle: current object pose.
        task: task geometry, providing the goal pose.

    Returns:
        3D refined motor command, clipped to [-1, 1].
    """

    goal_error = _pose_vector(
        task.goal_x - object_x,
        task.goal_y - object_y,
        task.goal_angle - object_angle,
    )
    distance = np.sqrt(np.sum(goal_error**2))
    if distance < 0.6:
        closeness = 1.0 - min(distance / 0.6, 1.0)
        command = command * (1 - closeness) + goal_error * 2.0 * closeness
    return np.clip(command * min(1.0, distance * 2.5), -1, 1)


def _path_lengths(trajectory: list[TimestepRecord]) -> tuple[float, float, float]:
    """Return (translational, rotational, combined) path length actually travelled.

    Args:
        trajectory: ordered list of TimestepRecord for one trial.

    Returns:
        Tuple of (translational, rotational, combined) path lengths.
    """

    translational = rotational = combined = 0.0
    for index in range(1, len(trajectory)):
        previous, current = trajectory[index - 1], trajectory[index]
        dx = current.obj_x - previous.obj_x
        dy = current.obj_y - previous.obj_y
        da = current.obj_angle - previous.obj_angle
        translational += math.hypot(dx, dy)
        rotational += abs(da)
        combined += math.sqrt(dx**2 + dy**2 + da**2)
    return translational, rotational, combined


def _efficiency(optimal: float, actual: float) -> float | None:
    """Ratio of optimal to actual path, or None where no path was travelled.

    Args:
        optimal: shortest possible path length.
        actual: path length actually travelled.

    Returns:
        ``optimal / actual`` capped at 1.0, or None if ``actual`` is ~0.
    """

    if actual <= MOVEMENT_EPSILON:
        return None
    return float(min(optimal / actual, 1.0))


def _trial_weights(
    params: DevelopmentalParams,
    rng: np.random.Generator,
) -> tuple[AffordanceWeights, MotorWeights]:
    """Build per-trial affordance and motor weights."""

    base_matrix = get_affordance_matrix(params.affordance_matrix_variant)
    affordance_matrix = np.clip(
        base_matrix * params.affordance_coupling
        + rng.normal(0, params.affordance_jitter_sd, base_matrix.shape),
        0,
        1,
    )
    motor_matrix = MOTOR_BASE + rng.normal(0, params.motor_jitter_sd, MOTOR_BASE.shape)
    return AffordanceWeights(affordance_matrix, ACTION_BIAS), MotorWeights(motor_matrix)


def run_trial(
    params: DevelopmentalParams,
    task: TaskConfig,
    seed: int = 42,
    trial_id: int = 0,
    rng: np.random.Generator | None = None,
) -> TrialResult:
    """
    Simulate one 2D object-manipulation trial.

    Each timestep runs the full cascade:
      1. Gaze advance - switches fixation with a dwell-time hazard model.
      2. Visual percept sampling - noisy features from the fixated entity.
      3. Working-memory update - integrates percepts with trace-strength decay
         and capacity competition.
      4. Movement initiation check - motor output is gated until WM is strong enough.
      5. Affordance estimation - maps 11-D working memory to action affordances.
      6. Motor planning - blends affordance-driven and goal-error commands.
      7. Habit blending - mixes habitual translate-first routine with goal-directed cmd.
      8. Delayed visual correction - applies lagged visual error feedback.
      9. Near-goal fine control - proportional approach as XY distance falls.
     10. Pose execution - advances object_x, object_y, and object_angle.
     11. Success check - breaks when position and angle errors are within tolerance.

    Args:
        params: developmental parameter set for this agent.
        task: task geometry, tolerances and step limit.
        seed: seed used when ``rng`` is not supplied.
        trial_id: index of this trial within its condition.
        rng: an explicit generator. Supplying one lets the caller choose whether
            conditions share a common random-number stream or draw from
            independent streams; see :func:`model_utils.make_rng`.

    Returns:
        A TrialResult with the final outcome and full per-timestep trajectory.
    """

    rng = resolve_rng(rng, seed)

    affordance_weights, motor_weights = _trial_weights(params, rng)

    gaze = GazeState()
    wm = WorkingMemoryState()
    habit = HabitState()
    correction = CorrectionState()

    object_x, object_y, object_angle = task.start_x, task.start_y, task.start_angle
    target_state = _pose_vector(task.goal_x, task.goal_y, task.goal_angle)
    goal_motor_state = _pose_vector(task.goal_x, task.goal_y, task.goal_angle)

    trajectory = []
    movement_started = False
    movement_onset = None
    rotation_onset = None
    translation_onset = None
    last_object_fixation = last_target_fixation = -10

    for current_time in range(task.max_timesteps):
        position_error = _position_error(object_x, object_y, task)
        angle_error = angular_error(
            object_angle, task.goal_angle, task.angular_symmetry
        )

        gaze_target = step_gaze(gaze, params, wm, movement_started, current_time, rng)
        if gaze_target == "object":
            last_object_fixation = current_time
        else:
            last_target_fixation = current_time

        object_state = np.array(
            [object_x, object_y, object_angle, task.obj_width, task.obj_height]
        )
        both_recently_fixated = (
            current_time - last_object_fixation <= 3
            and current_time - last_target_fixation <= 3
        )
        percept, sampled = sample_percept(
            params,
            object_state,
            target_state,
            gaze_target,
            both_recently_fixated,
            rng,
        )

        working_memory = update_wm(wm, params, percept, sampled, gaze_target)
        strengths = wm.trace_strength.copy()
        object_info = float(np.mean(strengths[OBJECT_FEATURES]))
        target_info = float(np.mean(strengths[TARGET_FEATURES]))
        relational_info = float(np.mean(strengths[RELATION_FEATURES]))

        if not movement_started and np.mean(strengths) >= params.initiation_threshold:
            movement_started = True
            movement_onset = current_time

        affordances = estimate_affordances(
            affordance_weights, params, working_memory, rng
        )
        current_motor_state = _pose_vector(object_x, object_y, object_angle)
        planned_command = plan_motor_command(
            motor_weights,
            params,
            affordances,
            goal_motor_state,
            current_motor_state,
            rng,
        )
        command = blend_habit(habit, params, planned_command)
        command += apply_correction(
            correction, params, current_motor_state, goal_motor_state, current_time, rng
        )

        final_command = (
            compute_near_goal_command(command, object_x, object_y, object_angle, task)
            if movement_started
            else np.zeros(3)
        )

        object_x += final_command[0] * 0.10
        object_y += final_command[1] * 0.10
        object_angle += final_command[2] * 0.10

        rotation_started = abs(final_command[2]) > ONSET_COMMAND_THRESHOLD
        translation_started = (
            abs(final_command[0]) > ONSET_COMMAND_THRESHOLD
            or abs(final_command[1]) > ONSET_COMMAND_THRESHOLD
        )
        if movement_started and rotation_started and rotation_onset is None:
            rotation_onset = current_time
        if movement_started and translation_started and translation_onset is None:
            translation_onset = current_time

        trajectory.append(
            TimestepRecord(
                current_time,
                object_x,
                object_y,
                object_angle,
                gaze_target,
                movement_started,
                object_info,
                target_info,
                relational_info,
                position_error,
                angle_error,
                gaze.switch_count,
            )
        )

        if (
            position_error < task.position_tolerance
            and angle_error < task.angle_tolerance
        ):
            break

    final_position_error = float(_position_error(object_x, object_y, task))
    final_angle_error = angular_error(
        object_angle, task.goal_angle, task.angular_symmetry
    )
    total_fixations = len(gaze.fixation_history)
    success = bool(
        final_position_error < task.position_tolerance
        and final_angle_error < task.angle_tolerance
    )

    optimal_translation = math.hypot(
        task.goal_x - task.start_x, task.goal_y - task.start_y
    )
    optimal_rotation = abs(task.goal_angle - task.start_angle)
    optimal_combined = math.sqrt(optimal_translation**2 + optimal_rotation**2)
    actual_translation, actual_rotation, actual_combined = _path_lengths(trajectory)

    return TrialResult(
        params_name=params.name,
        task_name=task.name,
        trial_id=trial_id,
        success=success,
        timesteps_used=len(trajectory),
        timed_out=not success,
        final_pos_error=final_position_error,
        final_angle_error=final_angle_error,
        trajectory=trajectory,
        movement_initiated=movement_started,
        movement_onset=movement_onset,
        rotation_onset=rotation_onset,
        translation_onset=translation_onset,
        efficiency=_efficiency(optimal_combined, actual_combined),
        translational_efficiency=_efficiency(optimal_translation, actual_translation),
        rotational_efficiency=_efficiency(optimal_rotation, actual_rotation),
        total_gaze_switches=gaze.switch_count,
        object_fixation_pct=gaze.object_fixation_count / max(total_fixations, 1),
        target_fixation_pct=gaze.target_fixation_count / max(total_fixations, 1),
        pre_movement_target_fixation_pct=(
            gaze.pre_movement_target_count / gaze.pre_movement_steps
            if gaze.pre_movement_steps > 0
            else None
        ),
        time_to_first_target_fixation=gaze.first_target_fixation,
        gaze_history=gaze.fixation_history,
    )
