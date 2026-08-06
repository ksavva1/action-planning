"""Core gaze planning model.

The model is intentionally small, but each layer maps to a psychological claim:
looking controls what can be sampled, working memory determines what survives
across fixations, affordances translate perception into action options, and the
motor layer blends goal-directed control with a developmentally weaker or
stronger translate-first habit.

The model is 2D: position is described by (x, y) and orientation by a single
angle. Motor commands produce changes in these three dimensions only.
"""

from dataclasses import dataclass, field

import numpy as np

from prev_iterations.gaze_model.affordance_matrices import AFFORDANCE_MATRIX_VARIANTS, get_affordance_matrix
from prev_iterations.gaze_model.model_utils import (
    DevelopmentalParams,
    TaskConfig,
    TimestepRecord,
    TrialResult,
)

__all__ = [
    "DevelopmentalParams", "TaskConfig",
    "TimestepRecord", "TrialResult", "GazeState", "WorkingMemoryState",
    "AffordanceWeights", "MotorWeights", "HabitState", "CorrectionState",
    "AFFORDANCE_MATRIX_VARIANTS", "run_trial",
]

ACTION_BIAS = np.array([.15, .1, .05, .2])
# 4 affordances (reach, grasp, rotate, translate) → 3 motor dimensions (x, y, angle).
MOTOR_BASE = np.array([
    [.9, 0,  0 ],   # reach    → x
    [0,  .9, 0 ],   # grasp    → y
    [0,  0,  .9],   # rotate   → angle
    [.5, .5, 0 ],   # translate → x and y
])


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
    """Feature values and trace strengths for object, target, and relational features.

    Buffer layout (11 features):
      [0:5]   visual object features  (x, y, angle, width, height)
      [5:8]   visual target features  (goal x, y, angle)
      [8:11]  relational features     (rel_dx, rel_dy, rel_d_angle)
    """

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


def step_gaze(
    gaze: GazeState,
    params: DevelopmentalParams,
    wm: WorkingMemoryState,
    movement_started: bool,
    rng: np.random.Generator,
) -> str:
    """
    Advance fixation by one timestep and return "object" or "target".

    The base switch probability (gaze_switch_rate × dwell-time hazard) is
    dynamically modulated each timestep by four cognitive factors drawn from
    the current working-memory state and developmental parameters:

    1. **WM saturation** — when the trace for the currently fixated item is
       already strong, there is little more to gain; the switch rate increases
       proportionally (diminishing returns on continued fixation).

    2. **Decay urgency** — when the unfixated item's trace has fallen below a
       threshold, the need to refresh that representation boosts the rate of
       switching back.

    3. **Perceptual noise / acuity** — higher noise or lower mean acuity
       stretches the effective fixation duration so that more dwell time is
       needed before the hazard reaches its peak, slowing voluntary switching.

    4. **Planning horizon** — a longer lookahead shifts the effective target
       bias upward, making proactive target fixations more likely when the
       agent does decide to switch.

    5. **Pre-movement scanning** — before motor output begins, a high
       initiation threshold combined with a low current WM strength creates an
       information gap that inflates the scan rate; the boost decays to zero as
       WM strength approaches the threshold.
    """

    gaze.dwell_time += 1

    # ── 1 & 2. WM-driven urgency ──────────────────────────────────────
    if gaze.current_fixation == "object":
        fixated_trace   = float(np.mean(wm.trace_strength[:5]))
        unfixated_trace = float(np.mean(wm.trace_strength[5:8]))
    else:
        fixated_trace   = float(np.mean(wm.trace_strength[5:8]))
        unfixated_trace = float(np.mean(wm.trace_strength[:5]))

    # Saturation: diminishing return from staying on a well-sampled fixation
    saturation_drive = fixated_trace                                  # [0, 1]
    # Decay urgency: need to refresh the other item when its trace has faded
    decay_urgency    = max(0.0, (0.5 - unfixated_trace) * 2.0)       # [0, 1]

    wm_multiplier = 1.0 + 0.35 * saturation_drive + 0.45 * decay_urgency  # [1.0, 1.80]

    # ── 3. Perceptual noise / acuity: stretch fixation duration ───────
    acuity_mean      = (params.location_acuity + params.orientation_acuity) / 2.0
    noise_load       = params.perceptual_noise * (1.0 - acuity_mean)
    effective_duration = params.fixation_duration_mean * (1.0 + noise_load)

    # ── 4. Planning horizon: boost target bias (proactive gaze) ───────
    horizon_boost         = (params.planning_horizon / 6.0) * 0.20
    effective_target_bias = min(0.95, params.target_bias + horizon_boost)

    # ── 5. Pre-movement scanning: info-gap drives faster search ───────
    if not movement_started:
        mean_strength   = float(np.mean(wm.trace_strength))
        info_gap        = max(0.0, params.initiation_threshold - mean_strength)
        threshold_drive = info_gap / max(params.initiation_threshold, 1e-6)
        scan_multiplier = 1.0 + threshold_drive * 0.50                # [1.0, 1.5]
    else:
        scan_multiplier = 1.0

    # ── Combined switch decision ───────────────────────────────────────
    hazard         = 1.0 - np.exp(-gaze.dwell_time / effective_duration)
    effective_rate = params.gaze_switch_rate * wm_multiplier * scan_multiplier

    if rng.random() < effective_rate * hazard:
        if gaze.current_fixation == "object":
            gaze.current_fixation = "target"
        else:
            gaze.current_fixation = "object" if rng.random() < (1.0 - effective_target_bias) else "target"
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
    Sample noisy visual feature values from the fixated entity.

    Returns 11-D percept and sampled arrays. Object features fill [0:5],
    target features fill [5:8], and relational features fill [8:11].
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
    maintaining two separately sampled entities. Capacity pressure weakens
    the least active traces when too many features are being held at once.
    """

    decay = np.full(11, params.wm_decay)
    decay[5:8] = params.wm_unfixated_decay if gaze_target == "object" else params.wm_decay
    decay[:5] = params.wm_decay if gaze_target == "object" else params.wm_unfixated_decay
    decay[8:11] = params.wm_unfixated_decay
    wm.trace_strength *= 1 - decay

    for index in np.flatnonzero(sampled > .5):
        if wm.trace_strength[index] > .1:
            wm.memory_buffer[index] = .4 * wm.memory_buffer[index] + .6 * percept[index]
        else:
            wm.memory_buffer[index] = percept[index]
        wm.trace_strength[index] = min(1.0, wm.trace_strength[index] + .5)

    if np.sum(wm.trace_strength > .1) > params.wm_capacity:
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
    goal_error = goal_motor_state - current_motor_state
    planning_weight = min(1.0, params.planning_horizon / 6.0)
    command = (
        (1 - planning_weight) * affordance_command
        + planning_weight * goal_error * .3
        + rng.normal(0, params.motor_noise, 3)
    )
    return np.clip(command * min(1.0, np.sqrt(np.sum(goal_error ** 2)) / .5), -1, 1)


def blend_habit(habit: HabitState, params: DevelopmentalParams, command: np.ndarray) -> np.ndarray:
    """
    Blend a translate-first habit with the goal-directed command.

    Early in a trial the habit pushes the object laterally before rotation; later
    it shifts toward rotation. The habit fades over time so mature, goal-directed
    control can take over even in high-habit profiles.
    """

    habit.step_count += 1
    phase_length = int(4 + 12 * params.habit_strength)
    habitual = np.array([0.7, 0.7, 0.0]) if habit.step_count < phase_length else np.array([0.0, 0.0, 0.8])
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

    current_error = goal_motor_state - current_motor_state
    correction.error_history.append(current_error.copy())
    if current_time < params.correction_delay:
        return np.zeros(3)

    delayed_error = correction.error_history[len(correction.error_history) - 1 - params.correction_delay]
    command = params.correction_rate * delayed_error + rng.normal(0, params.motor_noise * .5, 3)
    return command * min(1.0, np.sqrt(np.sum(current_error ** 2)) / .4)


def compute_near_goal_command(command: np.ndarray, object_x: float, object_y: float, object_angle: float, task: TaskConfig) -> np.ndarray:
    """
    Add fine-grained proportional control as the object nears the goal.

    This models a shift from coarse, ballistic movement to local error correction
    as position and angle differences become small.
    """

    dx, dy, da = task.goal_x - object_x, task.goal_y - object_y, task.goal_angle - object_angle
    distance = np.sqrt(dx ** 2 + dy ** 2 + da ** 2)
    if distance < .6:
        closeness = 1.0 - min(distance / .6, 1.0)
        command = command * (1 - closeness) + np.array([dx, dy, da]) * 2.0 * closeness
    return np.clip(command * min(1.0, distance * 2.5), -1, 1)


def run_trial(params: DevelopmentalParams, task: TaskConfig, seed: int = 42, trial_id: int = 0) -> TrialResult:
    """
    Simulate one 2D object-manipulation trial.

    Each timestep runs the full cascade:
      1. Gaze advance — switches fixation with a dwell-time hazard model.
      2. Visual percept sampling — noisy features from the fixated entity.
      3. Working-memory update — integrates percepts with trace-strength decay
         and capacity competition.
      4. Movement initiation check — motor output is gated until WM is strong enough.
      5. Affordance estimation — maps 11-D working memory to action affordances.
      6. Motor planning — blends affordance-driven and goal-error commands.
      7. Habit blending — mixes habitual translate-first routine with goal-directed cmd.
      8. Delayed visual correction — applies lagged visual error feedback.
      9. Near-goal fine control — proportional approach as XY distance falls.
     10. Pose execution — advances object_x, object_y, and object_angle.
     11. Success check — breaks when position and angle errors are within tolerance.
    """

    rng = np.random.default_rng(seed)

    base_matrix = get_affordance_matrix(params.affordance_matrix_variant)
    matrix = np.clip(base_matrix * params.affordance_coupling + rng.normal(0, .03, base_matrix.shape), 0, 1)
    affordance_weights = AffordanceWeights(matrix, ACTION_BIAS)
    motor_weights = MotorWeights(MOTOR_BASE + rng.normal(0, .02, MOTOR_BASE.shape))

    gaze = GazeState()
    wm = WorkingMemoryState()
    habit = HabitState()
    correction = CorrectionState()

    object_x, object_y, object_angle = task.start_x, task.start_y, task.start_angle
    target_state = np.array([task.goal_x, task.goal_y, task.goal_angle])
    goal_motor_state = np.array([task.goal_x, task.goal_y, task.goal_angle])

    trajectory = []
    movement_started = False
    movement_onset = rotation_onset = translation_onset = task.max_timesteps
    last_object_fixation = last_target_fixation = -10

    for current_time in range(task.max_timesteps):
        position_error = np.sqrt((object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2)
        angle_error = abs(object_angle - task.goal_angle)

        gaze_target = step_gaze(gaze, params, wm, movement_started, rng)
        if gaze_target == "object":
            last_object_fixation = current_time
        else:
            last_target_fixation = current_time

        object_state = np.array([object_x, object_y, object_angle, task.obj_width, task.obj_height])
        both_recently_fixated = current_time - last_object_fixation <= 3 and current_time - last_target_fixation <= 3
        percept, sampled = sample_percept(
            params, object_state, target_state, gaze_target, both_recently_fixated, rng,
        )

        working_memory = update_wm(wm, params, percept, sampled, gaze_target)
        strengths = wm.trace_strength.copy()
        object_info = float(np.mean(strengths[:5]))
        target_info = float(np.mean(strengths[5:8]))
        relational_info = float(np.mean(strengths[8:11]))

        if not movement_started and np.mean(strengths) >= params.initiation_threshold:
            movement_started = True
            movement_onset = current_time

        affordances = estimate_affordances(affordance_weights, params, working_memory, rng)
        current_motor_state = np.array([object_x, object_y, object_angle])
        command = blend_habit(
            habit,
            params,
            plan_motor_command(motor_weights, params, affordances, goal_motor_state, current_motor_state, rng),
        )
        command += apply_correction(correction, params, current_motor_state, goal_motor_state, current_time, rng)

        final_command = compute_near_goal_command(command, object_x, object_y, object_angle, task) if movement_started else np.zeros(3)

        object_x += final_command[0] * .10
        object_y += final_command[1] * .10
        object_angle += final_command[2] * .10

        if movement_started and abs(final_command[2]) > .1 and rotation_onset == task.max_timesteps:
            rotation_onset = current_time
        if movement_started and (abs(final_command[0]) > .1 or abs(final_command[1]) > .1) and translation_onset == task.max_timesteps:
            translation_onset = current_time

        trajectory.append(TimestepRecord(
            current_time, object_x, object_y, object_angle, gaze_target, movement_started,
            object_info, target_info, relational_info, position_error, angle_error, gaze.switch_count,
        ))

        if position_error < task.position_tolerance and angle_error < task.angle_tolerance:
            break

    final_position_error = np.sqrt((object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2)
    final_angle_error = abs(object_angle - task.goal_angle)
    total_fixations = len(gaze.fixation_history)

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
    efficiency = min(optimal / max(actual, .01), 1.0)

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
        efficiency,
        gaze.switch_count,
        gaze.object_fixation_count / max(total_fixations, 1),
        gaze.target_fixation_count / max(total_fixations, 1),
        gaze.fixation_history,
    )
