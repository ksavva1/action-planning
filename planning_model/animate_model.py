"""Animate one trial for each developmental stage."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, transforms
from matplotlib.animation import FuncAnimation

from model_utils import DevelopmentalParams, TaskConfig
from planning_cascade_model import run_trial

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

EYE_X, EYE_Y = -0.55, 1.05


def build_panel_label(params: DevelopmentalParams) -> str:
    """Build the compact parameter summary shown above each stage panel.

    Args:
        params: developmental parameters for the stage being labelled.

    Returns:
        Two-line string of formatted parameter values.
    """

    habit = int(100 * params.habit_strength / (params.habit_strength + params.goal_directed_strength))
    return (
        f"samp={params.sampling_rate:.2f}  noise={params.perceptual_noise:.2f}  "
        f"gaze_sw={params.gaze_switch_rate:.2f}\n"
        f"WM={params.wm_capacity}  horizon={params.planning_horizon}  "
        f"habit={habit}%  corr={params.correction_rate:.2f}"
    )


def draw_rotated_rectangle(ax, x, y, width, height, angle, colour, **kwargs):
    """Draw a rectangle centered on (x, y) and rotated by angle radians.

    Args:
        ax: matplotlib axes to draw on.
        x, y: centre of the rectangle in data coordinates.
        width, height: rectangle dimensions.
        angle: rotation in radians.
        colour: default edge/face colour.
        **kwargs: optional overrides for face colour (``fc``), line width
            (``lw``), line style (``ls``), alpha (``alpha``), and z-order
            (``z``).

    Returns:
        The created FancyBboxPatch, added to ``ax``.
    """

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


def setup_figure(task: TaskConfig, stages: dict):
    """Create the multi-panel figure and static target elements.

    Args:
        task: TaskConfig describing the task geometry.
        stages: dict of {name: DevelopmentalParams}, one panel per entry.

    Returns:
        Tuple of (figure, axes, trails, status_texts, timestep_text).
    """

    stage_names = list(stages)
    n = len(stage_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6.5), facecolor=BACKGROUND_COLOUR)
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
        f"Goal ({task.goal_x},{task.goal_y}) angle={np.degrees(task.goal_angle):.0f}°",
        ha="center", va="top", fontsize=9, color=DIM_COLOUR,
    )
    timestep_text = fig.text(.5, .905, "", ha="center", fontsize=9, color=DIM_COLOUR)
    trails, status_texts = {}, {}

    axes = np.atleast_1d(axes)
    for ax, (name, params) in zip(axes, stages.items()):
        ax.set_facecolor(SURFACE_COLOUR)
        ax.set_xlim(-.7, 1.1)
        ax.set_ylim(-.6, 1.25)
        ax.set_aspect("equal")
        ax.tick_params(colors=DIM_COLOUR, labelsize=5)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOUR)

        colour = STAGE_COLOURS.get(name, "#aaaaaa")
        ax.set_title(
            build_panel_label(params),
            fontsize=6.2, fontfamily="monospace", color=colour,
            pad=5, linespacing=1.3,
        )
        draw_rotated_rectangle(
            ax, task.goal_x, task.goal_y, task.obj_width + .05, task.obj_height + .05,
            task.goal_angle, GOAL_COLOUR, fc="none", lw=1.8, ls="--", alpha=.5, z=1,
        )
        ax.text(task.goal_x, task.goal_y + task.obj_height / 2 + .08, "target",
                fontsize=6, ha="center", color=f"{GOAL_COLOUR}88", zorder=1)
        trails[name], = ax.plot([], [], color=colour, alpha=.4, lw=1.2, zorder=2)
        status_texts[name] = ax.text(
            .97, .03, "", transform=ax.transAxes, fontsize=7.5, ha="right",
            va="bottom", fontfamily="monospace", color=TEXT_COLOUR, zorder=15,
        )

    return fig, axes, trails, status_texts, timestep_text


def mark_dynamic(artist):
    """Tag an artist so it can be removed before drawing the next frame.

    Args:
        artist: a matplotlib artist.

    Returns:
        The same artist, tagged.
    """

    artist._d = True
    return artist


def remove_dynamic_artists(ax) -> None:
    """Remove frame-specific artists while leaving static panel content intact.

    Args:
        ax: matplotlib axes to clear dynamic artists from.
    """

    for artist in list(ax.patches) + list(ax.texts) + list(ax.lines):
        if getattr(artist, "_d", False):
            artist.remove()


def draw_frame_for_stage(ax, name, trial, frame, task, trails, status_texts) -> None:
    """Draw object pose, gaze target, movement trail, and status for one stage.

    Args:
        ax: matplotlib axes for this stage's panel.
        name: stage name, used to look up its colour and trail.
        trial: TrialResult holding the trajectory to draw.
        frame: current animation frame index.
        task: TaskConfig describing the task geometry.
        trails: mapping from stage name to its trail Line2D.
        status_texts: mapping from stage name to its status Text artist.
    """

    trajectory = trial.trajectory
    index = min(frame, len(trajectory) - 1)
    step = trajectory[index]
    remove_dynamic_artists(ax)

    trails[name].set_data(
        [item.obj_x for item in trajectory[:index + 1]],
        [item.obj_y for item in trajectory[:index + 1]],
    )
    colour = STAGE_COLOURS.get(name, "#aaaaaa")
    mark_dynamic(draw_rotated_rectangle(
        ax, step.obj_x, step.obj_y, task.obj_width, task.obj_height,
        step.obj_angle, colour, alpha=.88,
    ))

    looking_at_object = step.gaze_target == "object"
    gaze_colour = GAZE_COLOUR_OBJECT if looking_at_object else GAZE_COLOUR_TARGET
    fixation_x = step.obj_x if looking_at_object else task.goal_x
    fixation_y = step.obj_y if looking_at_object else task.goal_y

    for circle in (
        plt.Circle((EYE_X, EYE_Y), .035, fc=gaze_colour, ec="none", alpha=.85, zorder=10),
        plt.Circle((EYE_X, EYE_Y), .015, fc=BACKGROUND_COLOUR, ec="none", alpha=.9, zorder=11),
        plt.Circle((fixation_x, fixation_y), .025, fc=gaze_colour, ec="white", lw=.4, alpha=.7, zorder=8),
    ):
        ax.add_patch(mark_dynamic(circle))

    gaze_line, = ax.plot([EYE_X, fixation_x], [EYE_Y, fixation_y], color=gaze_colour, lw=1, alpha=.45, zorder=5)
    mark_dynamic(gaze_line)
    mark_dynamic(ax.text(
        EYE_X + .08, EYE_Y, "obj" if looking_at_object else "tgt",
        fontsize=5.5, color=gaze_colour, va="center", fontfamily="monospace", alpha=.8, zorder=12,
    ))

    if frame >= len(trajectory):
        status_texts[name].set_text("SUCCESS" if trial.success else "MISSED")
        status_texts[name].set_color(ACCENT_COLOUR if trial.success else "#fb7185")
    else:
        status_texts[name].set_text(f"pos={step.pos_error:.2f} ang={step.angle_error:.2f}")
        status_texts[name].set_color(DIM_COLOUR)


def update_animation_frame(frame, axes, results, task, trails, status_texts, timestep_text, max_length, stage_names):
    """Draw one animation frame across all stage panels.

    Args:
        frame: current animation frame index.
        axes: sequence of matplotlib axes, one per stage.
        results: mapping from stage name to its TrialResult.
        task: TaskConfig describing the task geometry.
        trails: mapping from stage name to its trail Line2D.
        status_texts: mapping from stage name to its status Text artist.
        timestep_text: shared Text artist showing the current timestep.
        max_length: length of the longest trajectory, for the timestep label.
        stage_names: ordered stage names matching ``axes``.

    Returns:
        Empty list (blitting is disabled, so no artist list is required).
    """

    timestep_text.set_text(f"t = {min(frame, max_length - 1)}")
    for ax, name in zip(axes, stage_names):
        draw_frame_for_stage(ax, name, results[name], frame, task, trails, status_texts)
    return []


def animate(
    task: TaskConfig,
    stages: dict,
    save: bool = False,
    seed: int = 42,
) -> object:
    """
    Build and show or save the developmental comparison animation.

    Args:
        task: TaskConfig describing the task geometry.
        stages: dict of {name: DevelopmentalParams} to compare.
        save: save a GIF when True; otherwise open an interactive window.
        seed: random seed passed to each stage's trial.

    Returns:
        The matplotlib FuncAnimation object.
    """

    stage_names = list(stages)
    results = {name: run_trial(params, task, seed=seed) for name, params in stages.items()}
    max_length = max(len(trial.trajectory) for trial in results.values())
    fig, axes, trails, status_texts, timestep_text = setup_figure(task, stages)

    animation = FuncAnimation(
        fig,
        update_animation_frame,
        frames=max_length + 6,
        fargs=(axes, results, task, trails, status_texts, timestep_text, max_length, stage_names),
        interval=200,
        blit=False,
        repeat=True,
    )
    if save:
        output_path = Path(__file__).with_name(f"planning_cascade_{task.name}.gif")
        print(f"Saving -> {output_path}")
        animation.save(output_path, writer="pillow", fps=5, savefig_kwargs={"facecolor": BACKGROUND_COLOUR})
        print(f"Done: {output_path}")
    else:
        plt.show()
    return animation
