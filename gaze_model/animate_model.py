"""Animate one trial for each developmental stage.

The visualisation is diagnostic rather than decorative: each panel shows how
developmental parameters change gaze allocation, movement onset, trajectory
shape, and final insertion accuracy under the same task and seed.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, transforms
from matplotlib.animation import FuncAnimation

try:
    from .planning_cascade_model import DEVELOPMENTAL_STAGES, TASKS, DevelopmentalParams, TaskConfig, run_trial
except ImportError:
    from planning_cascade_model import DEVELOPMENTAL_STAGES, TASKS, DevelopmentalParams, TaskConfig, run_trial

STAGES = ["A", "B", "C", "D"]

# Visualisation colours
STAGE_COLOURS = {"A": "#fb7185", "B": "#fbbf24", "C": "#6ee7b7", "D": "#60a5fa"}

BACKGROUND_COLOUR = "#0c0e12"
SURFACE_COLOUR = "#1a1e28"
BORDER_COLOUR = "#2a3148"
TEXT_COLOUR = "#e4e8f1"
DIM_COLOUR = "#636b83"
ACCENT_COLOUR = "#6ee7b7"
GAZE_COLOUR_OBJECT = "#fbbf24"
GAZE_COLOUR_TARGET = "#a78bfa"
GOAL_COLOUR = "#fbbf24"

# Position of the eye icon in each panel (top-left)
EYE_X, EYE_Y = -0.55, 1.05


def build_panel_label(params: DevelopmentalParams) -> str:
    """Build the compact parameter summary shown above each stage panel."""

    # Habit is shown as a percentage because it is easier to compare across
    # stage panels than two separate habit and goal-directed weights.
    habit = int(100 * params.habit_strength / (params.habit_strength + params.goal_directed_strength))
    return (
        f"samp={params.sampling_rate:.2f}  noise={params.perceptual_noise:.2f}  "
        f"gaze_sw={params.gaze_switch_rate:.2f}\n"
        f"WM={params.wm_capacity}  horizon={params.planning_horizon}  "
        f"habit={habit}%  corr={params.correction_rate:.2f}"
    )


def run_one_trial_per_stage(task_name: str, seed: int = 42) -> tuple[dict, TaskConfig]:
    """
    Run the same task and seed for all four developmental stages.

    Holding task and seed fixed makes differences in the animation attributable
    to the stage parameters rather than random trial setup.
    """

    # Reusing the same seed across stages makes the visual comparison about
    # developmental parameters, not about different random trajectories.
    task = TASKS[task_name]
    return {stage: run_trial(DEVELOPMENTAL_STAGES[stage], task, seed=seed) for stage in STAGES}, task


def draw_rotated_rectangle(ax, x, y, width, height, angle, colour, **kwargs):
    """Draw a rectangle centered on (x, y) and rotated by angle radians."""

    # The patch is drawn around the origin first so rotation is around its centre.
    rect = patches.FancyBboxPatch(
        (-width / 2, -height / 2), width, height,
        boxstyle="round,pad=0.012",
        facecolor=kwargs.get("fc", colour),
        edgecolor=colour,
        linewidth=kwargs.get("lw", 1.5),
        linestyle=kwargs.get("ls", "-"),
        alpha=kwargs.get("alpha", 1),
        zorder=kwargs.get("z", 3),
    )
    rect.set_transform(transforms.Affine2D().rotate(angle).translate(x, y) + ax.transData)
    ax.add_patch(rect)
    return rect


def setup_figure(task: TaskConfig):
    """Create the four-panel figure and static target-slot elements."""

    # Each stage gets the same axes and task geometry so differences are visual
    # consequences of the parameter profile rather than plotting scale.
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5), facecolor=BACKGROUND_COLOUR)
    fig.subplots_adjust(left=.03, right=.98, top=.82, bottom=.04, wspace=.15)
    fig.text(
        .5, .97, "Planning Model Simulation",
        ha="center", va="top", fontsize=16, fontweight="bold",
        color=ACCENT_COLOUR, fontfamily="serif",
        path_effects=[patheffects.withStroke(linewidth=2, foreground=BACKGROUND_COLOUR)],
    )
    fig.text(
        .5, .935,
        f"Task: {task.name.replace('_', ' ').title()}  |  "
        f"Fit into slot at ({task.goal_x},{task.goal_y}) rotated {np.degrees(task.goal_angle):.0f}°",
        ha="center", va="top", fontsize=9, color=DIM_COLOUR,
    )
    timestep_text = fig.text(.5, .905, "", ha="center", fontsize=9, color=DIM_COLOUR)
    trails, status_texts = {}, {}

    for ax, stage in zip(axes, STAGES):
        # Static panel elements are drawn once; moving objects and gaze markers
        # are tagged and redrawn each frame.
        ax.set_facecolor(SURFACE_COLOUR)
        ax.set_xlim(-.7, 1.1)
        ax.set_ylim(-.6, 1.25)
        ax.set_aspect("equal")
        ax.tick_params(colors=DIM_COLOUR, labelsize=5)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOUR)

        ax.set_title(
            build_panel_label(DEVELOPMENTAL_STAGES[stage]),
            fontsize=6.2, fontfamily="monospace", color=STAGE_COLOURS[stage],
            pad=5, linespacing=1.3,
        )
        draw_rotated_rectangle(
            ax, task.goal_x, task.goal_y, task.obj_width + .05, task.obj_height + .05,
            task.goal_angle, GOAL_COLOUR, fc="none", lw=1.8, ls="--", alpha=.5, z=1,
        )
        ax.text(task.goal_x, task.goal_y + task.obj_height / 2 + .08, "target",
                fontsize=6, ha="center", color=f"{GOAL_COLOUR}88", zorder=1)
        trails[stage], = ax.plot([], [], color=STAGE_COLOURS[stage], alpha=.4, lw=1.2, zorder=2)
        status_texts[stage] = ax.text(
            .97, .03, "", transform=ax.transAxes, fontsize=7.5, ha="right",
            va="bottom", fontfamily="monospace", color=TEXT_COLOUR, zorder=15,
        )

    return fig, axes, trails, status_texts, timestep_text


def mark_dynamic(artist):
    """Tag an artist so it can be removed before drawing the next frame."""

    # Matplotlib artists do not have a built-in frame lifecycle marker, so a
    # small private flag keeps cleanup simple without storing every handle.
    artist._d = True
    return artist


def remove_dynamic_artists(ax) -> None:
    """Remove frame-specific artists while leaving static panel content intact."""

    # Dynamic patches, lines, and labels are recreated from the latest trajectory
    # state; static target outlines and axes remain in place.
    for artist in list(ax.patches) + list(ax.texts) + list(ax.lines):
        if getattr(artist, "_d", False):
            artist.remove()


def draw_frame_for_stage(ax, stage, trial, frame, task, trails, status_texts) -> None:
    """
    Draw object pose, gaze target, movement trail, and status for one stage.

    The gaze line is intentionally prominent because the model's psychological
    claim is that looking controls what information can enter working memory.
    """

    trajectory = trial.trajectory
    # Hold the final pose during the extra end frames after a trial finishes.
    index = min(frame, len(trajectory) - 1)
    step = trajectory[index]
    remove_dynamic_artists(ax)

    # The movement trail shows cumulative path efficiency and overshoot.
    trails[stage].set_data(
        [item.obj_x for item in trajectory[:index + 1]],
        [item.obj_y for item in trajectory[:index + 1]],
    )
    mark_dynamic(draw_rotated_rectangle(
        ax, step.obj_x, step.obj_y, task.obj_width, task.obj_height,
        step.obj_angle, STAGE_COLOURS[stage], alpha=.88,
    ))

    looking_at_object = step.gaze_target == "object"
    colour = GAZE_COLOUR_OBJECT if looking_at_object else GAZE_COLOUR_TARGET
    # Gaze lands on the object centre or target slot centre, matching the model's
    # object-vs-target fixation state.
    fixation_x = step.obj_x if looking_at_object else task.goal_x
    fixation_y = step.obj_y if looking_at_object else task.goal_y

    for circle in (
        plt.Circle((EYE_X, EYE_Y), .035, fc=colour, ec="none", alpha=.85, zorder=10),
        plt.Circle((EYE_X, EYE_Y), .015, fc=BACKGROUND_COLOUR, ec="none", alpha=.9, zorder=11),
        plt.Circle((fixation_x, fixation_y), .025, fc=colour, ec="white", lw=.4, alpha=.7, zorder=8),
    ):
        ax.add_patch(mark_dynamic(circle))

    gaze_line, = ax.plot([EYE_X, fixation_x], [EYE_Y, fixation_y], color=colour, lw=1, alpha=.45, zorder=5)
    mark_dynamic(gaze_line)
    mark_dynamic(ax.text(
        EYE_X + .08, EYE_Y, "obj" if looking_at_object else "tgt",
        fontsize=5.5, color=colour, va="center", fontfamily="monospace", alpha=.8, zorder=12,
    ))

    if frame >= len(trajectory):
        status_texts[stage].set_text("FITTED" if trial.success else "MISSED")
        status_texts[stage].set_color(ACCENT_COLOUR if trial.success else "#fb7185")
    else:
        status_texts[stage].set_text(f"pos={step.pos_error:.2f} ang={step.angle_error:.2f}")
        status_texts[stage].set_color(DIM_COLOUR)


def update_animation_frame(frame, axes, results, task, trails, status_texts, timestep_text, max_length):
    """Draw one animation frame across all stage panels."""

    # The figure-level time label is clamped so the held ending frames still show
    # the final simulated timestep rather than a nonexistent model step.
    timestep_text.set_text(f"t = {min(frame, max_length - 1)}")
    for ax, stage in zip(axes, STAGES):
        draw_frame_for_stage(ax, stage, results[stage], frame, task, trails, status_texts)
    return []


def animate(task_name: str = "rotate_insert", save: bool = False, seed: int = 42):
    """
    Build and show or save the developmental comparison animation.

    Args:
        task_name: key into TASKS.
        save: save a GIF when True; otherwise open an interactive window.
        seed: random seed passed to each stage's trial.

    Returns:
        The matplotlib FuncAnimation object.
    """

    results, task = run_one_trial_per_stage(task_name, seed)
    max_length = max(len(trial.trajectory) for trial in results.values())
    fig, axes, trails, status_texts, timestep_text = setup_figure(task)

    # A few extra frames hold the final pose long enough to compare outcomes.
    animation = FuncAnimation(
        fig,
        update_animation_frame,
        frames=max_length + 6,
        fargs=(axes, results, task, trails, status_texts, timestep_text, max_length),
        interval=200,
        blit=False,
        repeat=True,
    )
    if save:
        output_path = Path(__file__).with_name(f"planning_cascade_{task_name}.gif")
        print(f"Saving -> {output_path}")
        animation.save(output_path, writer="pillow", fps=5, savefig_kwargs={"facecolor": BACKGROUND_COLOUR})
        print(f"Done: {output_path}")
    else:
        plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="rotate_insert", choices=list(TASKS))
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")
    animate(args.task, args.save, args.seed)
