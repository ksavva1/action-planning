
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Helpers

def wrap_to_pi(angle: float) -> float:
    """
    Map angle (rad) to [-π, π].

    Args:
        angle: Angle in radians
    Returns:
        Wrapped angle in radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def orientation_vector(theta: float, length: float = 0.18) -> np.ndarray:
    """
    Convert orientation angle to 2D arrow vector.

    Args:
        theta: Orientation angle (rad)
        length: Arrow length for visualisation

    Returns:
        np.ndarray shape (2,): [dx, dy] direction vector.
    """
    return np.array([length * np.cos(theta), length * np.sin(theta)], dtype=float)


# Data classes
# ---------------------------------------------------------------------

@dataclass
class ManipulationTask:
    """
    Defines task geometry and difficulty.

    Attributes:
        start_x, start_y: Initial object position.
        target_x, target_y: Target position.
        start_theta: Initial orientation (rad).
        target_theta: Desired orientation (rad).
        precision_demand: Weight on orientation accuracy.
        visibility: Scales perceptual uncertainty.
    """
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    start_theta: float
    target_theta: float
    precision_demand: float = 1.0
    visibility: float = 1.0

    @property
    def start_position(self) -> np.ndarray:
        return np.array([self.start_x, self.start_y], dtype=float)

    @property
    def target_position(self) -> np.ndarray:
        return np.array([self.target_x, self.target_y], dtype=float)


@dataclass
class DevelopmentalParameters:
    """
    Parameters controlling motor behaviour.

    Attributes:
        label: Profile name.
        movement_mode: 'translate_then_rotate' or 'coupled_move_rotate'.
        perception_noise: Noise in initial target estimate.
        working_memory_noise: placeholder for extensions.
        motor_noise: Noise in action execution.
        online_correction_gain: Strength of feedback correction.
        dt: Time step.
    """
    label: str
    movement_mode: str
    perception_noise: float
    working_memory_noise: float
    motor_noise: float
    online_correction_gain: float
    dt: float = 0.05


@dataclass
class BeliefState:
    """
    Internal estimate of target state.

    Attributes:
        est_target_pos: Estimated target position (2D).
        est_target_theta: Estimated target orientation.
        unc_target: Scalar uncertainty.
    """
    est_target_pos: np.ndarray
    est_target_theta: float
    unc_target: float


@dataclass
class SimulationResult:
    """
    Output of one trial.

    Attributes:
        label: Profile label.
        movement_mode: Control strategy used.
        total_time: Duration of movement.
        success: Whether final state meets thresholds.
        final_position_error: Distance to target.
        final_orientation_error: Orientation error.
        corrective_events: Large updates indicating correction.
        trajectory_xy: (T,2) position trajectory.
        trajectory_theta: (T,) orientation trajectory.
        unc_target_trace: Uncertainty over time.
    """
    label: str
    movement_mode: str
    total_time: float
    success: bool
    final_position_error: float
    final_orientation_error: float
    corrective_events: int
    trajectory_xy: np.ndarray
    trajectory_theta: np.ndarray
    unc_target_trace: np.ndarray


# Model
# ---------------------------------------------------------------------

class ActionPlanner:
    """
    Developmental controller for object manipulation.
    """

    def __init__(self, params: DevelopmentalParameters, rng: np.random.Generator | None = None):
        """
        Args:
            params: Developmental parameter set
            rng: Optional random generator (for reproducibility)
        """
        self.params = params
        self.rng = np.random.default_rng(0) if rng is None else rng

    def initialise_belief(self, task: ManipulationTask) -> BeliefState:
        """
        Initialise noisy estimate of target state.

        Returns:
            BeliefState with noisy position, orientation, and uncertainty
        """
        est_target_pos = task.target_position + self.rng.normal(0.0, self.params.perception_noise, size=2)
        est_target_theta = wrap_to_pi(task.target_theta + self.rng.normal(0.0, self.params.perception_noise))
        unc_target = max(0.03, self.params.perception_noise * (1.2 - 0.2 * task.visibility))
        return BeliefState(est_target_pos=est_target_pos, est_target_theta=est_target_theta, unc_target=unc_target)

    def _translate_step(self, pos: np.ndarray, goal: np.ndarray, theta: float, task: ManipulationTask) -> np.ndarray:
        """
        Single damped translation update.
            - Step size scales with distance to prevent overshoot
            - Penalised when misaligned with target orientation
            - Noise decreases near target
            - Snap-to-goal when sufficiently close

        Args:
            pos: Current position (2D)
            goal: Target position
            theta: Current orientation
            task: Task definition

        Returns:
            Updated position (2D).
        """
        vec = goal - pos
        dist = np.linalg.norm(vec)
        if dist < 0.025:
            return goal.copy()

        direction = vec / max(dist, 1e-6)
        alignment_penalty = abs(wrap_to_pi(task.target_theta - theta))
        base_step = 0.12 + 0.04 * np.tanh(dist)
        step_scale = min(base_step, 0.5 * dist)
        step_scale = max(0.015, step_scale - 0.025 * alignment_penalty)

        correction = self.params.online_correction_gain * 0.035 * direction
        noise_scale = self.params.motor_noise * min(1.0, dist / 0.25)
        noise = self.rng.normal(0.0, noise_scale, size=2)

        new_pos = pos + step_scale * direction + correction + noise

        if np.linalg.norm(goal - new_pos) > dist:
            new_pos = pos + 0.5 * vec

        if np.linalg.norm(goal - new_pos) < 0.035:
            new_pos = goal.copy()

        return new_pos

    def _rotate_step(self, theta: float, goal_theta: float) -> float:
        """
        Single damped rotation update.

        Args:
            theta: Current orientation
            goal_theta: Target orientation

        Returns:
            Updated orientation (rad)
        """
        error = wrap_to_pi(goal_theta - theta)

        if abs(error) < 0.025:
            return float(goal_theta)
        
        update = 0.32 * error + self.rng.normal(0.0, self.params.motor_noise)
        new_theta = wrap_to_pi(theta + update)

        if abs(wrap_to_pi(goal_theta - new_theta)) < 0.04:
            new_theta = float(goal_theta)
        return float(new_theta)

    def simulate_trial(self, task: ManipulationTask) -> SimulationResult:
        """
        Simulate one reach-to-target episode.

        Returns:
            SimulationResult w/ full trajectory and metrics.
        """
        belief = self.initialise_belief(task)
        pos = task.start_position.copy()
        theta = float(task.start_theta)
        t = 0.0
        corrective_events = 0

        xy_path: List[np.ndarray] = [pos.copy()]
        theta_path: List[float] = [theta]
        unc_trace: List[float] = [belief.unc_target]

        if self.params.movement_mode == "translate_then_rotate":
            # Younger-like
            for _ in range(45):
                prev = pos.copy()
                pos = self._translate_step(pos, task.target_position, theta, task)

                if np.linalg.norm(pos - prev) > 0.16:
                    corrective_events += 1
                xy_path.append(pos.copy())
                theta_path.append(theta)
                t += self.params.dt
                unc_trace.append(max(0.02, unc_trace[-1] * 0.985))

                if np.linalg.norm(task.target_position - pos) < 0.025:
                    pos = task.target_position.copy()
                    xy_path[-1] = pos.copy()
                    break

            for _ in range(25):
                prev_theta = theta
                theta = self._rotate_step(theta, task.target_theta)

                if abs(wrap_to_pi(theta - prev_theta)) > 0.10:
                    corrective_events += 1

                xy_path.append(pos.copy())
                theta_path.append(theta)
                t += self.params.dt
                unc_trace.append(max(0.02, unc_trace[-1] * 0.99))

                if abs(wrap_to_pi(task.target_theta - theta)) < 0.025:
                    theta = float(task.target_theta)
                    theta_path[-1] = theta
                    break

        elif self.params.movement_mode == "coupled_move_rotate":
            # Older-like
            for _ in range(55):
                prev = pos.copy()
                prev_theta = theta
                pos = self._translate_step(pos, task.target_position, theta, task)
                theta = self._rotate_step(theta, task.target_theta)

                if np.linalg.norm(pos - prev) > 0.16 or abs(wrap_to_pi(theta - prev_theta)) > 0.10:
                    corrective_events += 1
                xy_path.append(pos.copy())
                theta_path.append(theta)
                t += self.params.dt
                unc_trace.append(max(0.02, unc_trace[-1] * 0.98))

                if np.linalg.norm(task.target_position - pos) < 0.025 and abs(wrap_to_pi(task.target_theta - theta)) < 0.025:
                    pos = task.target_position.copy()
                    theta = float(task.target_theta)
                    xy_path[-1] = pos.copy()
                    theta_path[-1] = theta
                    break
        else:
            raise ValueError(f"Unknown movement mode: {self.params.movement_mode}")

        final_pos_err = float(np.linalg.norm(task.target_position - pos))
        final_theta_err = float(abs(wrap_to_pi(task.target_theta - theta)))
        success = (final_pos_err < 0.10) and (final_theta_err < 0.15)

        return SimulationResult(
            label=self.params.label,
            movement_mode=self.params.movement_mode,
            total_time=t,
            success=success,
            final_position_error=final_pos_err,
            final_orientation_error=final_theta_err,
            corrective_events=corrective_events,
            trajectory_xy=np.vstack(xy_path),
            trajectory_theta=np.array(theta_path),
            unc_target_trace=np.array(unc_trace),
        )


# Presets
# ---------------------------------------------------------------------

def make_younger_profile() -> DevelopmentalParameters:
    """Younger-like profile"""
    return DevelopmentalParameters(
        label="younger_like",
        movement_mode="translate_then_rotate",
        perception_noise=0.36,
        working_memory_noise=0.10,
        motor_noise=0.020,
        online_correction_gain=0.28,
    )


def make_older_profile() -> DevelopmentalParameters:
    """Older-like profile"""
    return DevelopmentalParameters(
        label="older_like",
        movement_mode="coupled_move_rotate",
        perception_noise=0.16,
        working_memory_noise=0.05,
        motor_noise=0.010,
        online_correction_gain=0.75,
    )


# HTML visualisation
# ---------------------------------------------------------------------

def build_animation_html(task: ManipulationTask, younger_result: SimulationResult, older_result: SimulationResult, save_path: str) -> None:
    """
    Create interactive Plotly HTML animation.
        - object trajectories
        - object orientation arrows
        - target orientation (dashed)
        - uncertainty traces over time

    Args:
        task: Task definition.
        younger_result: Younger simulation output.
        older_result: Older simulation output.
        save_path: Output HTML file path.
    """
    max_frames = max(len(younger_result.trajectory_xy), len(older_result.trajectory_xy))

    def pad2(arr, n):
        if len(arr) >= n:
            return arr[:n]
        return np.vstack([arr, np.repeat(arr[-1][None, :], n - len(arr), axis=0)])

    def pad1(arr, n):
        if len(arr) >= n:
            return arr[:n]
        return np.concatenate([arr, np.repeat(arr[-1], n - len(arr))])

    y_xy = pad2(younger_result.trajectory_xy, max_frames)
    o_xy = pad2(older_result.trajectory_xy, max_frames)
    y_th = pad1(younger_result.trajectory_theta, max_frames)
    o_th = pad1(older_result.trajectory_theta, max_frames)
    y_unc = pad1(younger_result.unc_target_trace, max_frames)
    o_unc = pad1(older_result.unc_target_trace, max_frames)

    all_xy = np.vstack([y_xy, o_xy, np.array([task.start_position, task.target_position])])
    margin = 0.4
    x_min = float(np.min(all_xy[:, 0]) - margin)
    x_max = float(np.max(all_xy[:, 0]) + margin)
    y_min = float(np.min(all_xy[:, 1]) - margin)
    y_max = float(np.max(all_xy[:, 1]) + margin)

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.55, 0.45],
        subplot_titles=("Younger-like", "Older-like", "Target uncertainty over time", ""),
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter", "colspan": 2}, None]],
        horizontal_spacing=0.13,
        vertical_spacing=0.22,
    )

    target_vec = orientation_vector(task.target_theta, 0.20)

    # Younger panel
    fig.add_trace(go.Scatter(x=[task.start_x], y=[task.start_y], mode="markers", marker=dict(symbol="square", size=10), name="Start"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[task.target_x], y=[task.target_y], mode="markers", marker=dict(symbol="star", size=14), name="Target"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[task.target_x, task.target_x + target_vec[0]], y=[task.target_y, task.target_y + target_vec[1]], mode="lines", line=dict(width=4, dash="dash"), name="Target orientation"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[y_xy[0, 0]], y=[y_xy[0, 1]], mode="lines", name="Younger path"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[y_xy[0, 0]], y=[y_xy[0, 1]], mode="markers", marker=dict(size=10), name="Younger object"), row=1, col=1)
    y_vec0 = orientation_vector(y_th[0], 0.18)
    fig.add_trace(go.Scatter(x=[y_xy[0, 0], y_xy[0, 0] + y_vec0[0]], y=[y_xy[0, 1], y_xy[0, 1] + y_vec0[1]], mode="lines", line=dict(width=4), name="Younger orientation"), row=1, col=1)

    # Older panel
    fig.add_trace(go.Scatter(x=[task.start_x], y=[task.start_y], mode="markers", marker=dict(symbol="square", size=10), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[task.target_x], y=[task.target_y], mode="markers", marker=dict(symbol="star", size=14), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[task.target_x, task.target_x + target_vec[0]], y=[task.target_y, task.target_y + target_vec[1]], mode="lines", line=dict(width=4, dash="dash"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[o_xy[0, 0]], y=[o_xy[0, 1]], mode="lines", name="Older path"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[o_xy[0, 0]], y=[o_xy[0, 1]], mode="markers", marker=dict(size=10), name="Older object"), row=1, col=2)
    o_vec0 = orientation_vector(o_th[0], 0.18)
    fig.add_trace(go.Scatter(x=[o_xy[0, 0], o_xy[0, 0] + o_vec0[0]], y=[o_xy[0, 1], o_xy[0, 1] + o_vec0[1]], mode="lines", line=dict(width=4), name="Older orientation"), row=1, col=2)

    # Uncertainty panel
    fig.add_trace(go.Scatter(x=[0], y=[y_unc[0]], mode="lines", name="Younger uncertainty"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[o_unc[0]], mode="lines", name="Older uncertainty"), row=2, col=1)

    frames = []
    for f in range(max_frames):
        y_vec = orientation_vector(y_th[f], 0.18)
        o_vec = orientation_vector(o_th[f], 0.18)
        xs = np.arange(f + 1)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[task.start_x], y=[task.start_y]),
                    go.Scatter(x=[task.target_x], y=[task.target_y]),
                    go.Scatter(x=[task.target_x, task.target_x + target_vec[0]], y=[task.target_y, task.target_y + target_vec[1]]),
                    go.Scatter(x=y_xy[:f+1, 0], y=y_xy[:f+1, 1]),
                    go.Scatter(x=[y_xy[f, 0]], y=[y_xy[f, 1]]),
                    go.Scatter(x=[y_xy[f, 0], y_xy[f, 0] + y_vec[0]], y=[y_xy[f, 1], y_xy[f, 1] + y_vec[1]]),

                    go.Scatter(x=[task.start_x], y=[task.start_y]),
                    go.Scatter(x=[task.target_x], y=[task.target_y]),
                    go.Scatter(x=[task.target_x, task.target_x + target_vec[0]], y=[task.target_y, task.target_y + target_vec[1]]),
                    go.Scatter(x=o_xy[:f+1, 0], y=o_xy[:f+1, 1]),
                    go.Scatter(x=[o_xy[f, 0]], y=[o_xy[f, 1]]),
                    go.Scatter(x=[o_xy[f, 0], o_xy[f, 0] + o_vec[0]], y=[o_xy[f, 1], o_xy[f, 1] + o_vec[1]]),

                    go.Scatter(x=xs, y=y_unc[:f+1]),
                    go.Scatter(x=xs, y=o_unc[:f+1]),
                ],
                name=str(f),
                layout=go.Layout(title_text=f"Frame {f+1}/{max_frames} | Younger = translate → rotate | Older = move + rotate")
            )
        )

    fig.frames = frames

    fig.update_xaxes(range=[x_min, x_max], title_text="x", row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], title_text="y", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[x_min, x_max], title_text="x", row=1, col=2)
    fig.update_yaxes(range=[y_min, y_max], title_text="y", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(range=[0, max_frames], title_text="Frame", row=2, col=1)
    fig.update_yaxes(title_text="Uncertainty", row=2, col=1)

    fig.update_layout(
        title=dict(text="Planning Model", x=0.5),
        width=1200,
        height=900,
        margin=dict(t=130, b=120, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(size=11)),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "direction": "left",
            "x": 0.5,
            "y": 1.05,
            "xanchor": "center",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "currentvalue": {"prefix": "Frame: "},
            "steps": [
                {"label": str(k+1), "method": "animate", "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]}
                for k in range(max_frames)
            ],
            "x": 0.12,
            "y": -0.08,
            "len": 0.76,
        }],
    )

    fig.write_html(save_path, include_plotlyjs=True, full_html=True)


def run_demo(output_dir: str = ".") -> None:
    """Run demo and save HTML and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    task = ManipulationTask(
        start_x=0.0,
        start_y=0.0,
        target_x=1.2,
        target_y=0.9,
        start_theta=-1.2,
        target_theta=0.95,
        precision_demand=1.2,
        visibility=0.9,
    )

    younger = ActionPlanner(make_younger_profile(), rng=np.random.default_rng(10))
    older = ActionPlanner(make_older_profile(), rng=np.random.default_rng(20))

    younger_result = younger.simulate_trial(task)
    older_result = older.simulate_trial(task)

    html_path = os.path.join(output_dir, "action_animation.html")
    build_animation_html(task, younger_result, older_result, html_path)

    metrics = {
        "task": asdict(task),
        "younger": {
            "success": younger_result.success,
            "movement_mode": younger_result.movement_mode,
            "corrective_events": younger_result.corrective_events,
            "final_position_error": younger_result.final_position_error,
            "final_orientation_error": younger_result.final_orientation_error,
            "total_time": younger_result.total_time,
        },
        "older": {
            "success": older_result.success,
            "movement_mode": older_result.movement_mode,
            "corrective_events": older_result.corrective_events,
            "final_position_error": older_result.final_position_error,
            "final_orientation_error": older_result.final_orientation_error,
            "total_time": older_result.total_time,
        },
    }

    with open(os.path.join(output_dir, "trial_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    run_demo(".")
