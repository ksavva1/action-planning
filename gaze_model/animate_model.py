"""
Animated Visualisation

Displays 4 parameter configurations side-by-side, each showing:
  - The object (coloured rectangle) moving and rotating
  - The target slot (dashed gold outline) the object must fit into
  - A gaze line from an eye icon to whichever entity the infant is fixating
  - The movement trace left behind by the object
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, transforms
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planning_cascade_model import (
    DEVELOPMENTAL_STAGES, TASKS,
    DevelopmentalParams, TaskConfig, run_trial,
)

# The four developmental stages
STAGES = ["A", "B", "C", "D"]
STAGE_COLOURS = {"A": "#fb7185", "B": "#fbbf24", "C": "#6ee7b7", "D": "#60a5fa"}

# Background colours
BACKGROUND_COLOUR = "#0c0e12"
SURFACE_COLOUR = "#1a1e28"
BORDER_COLOUR = "#2a3148"
TEXT_COLOUR = "#e4e8f1"
DIM_COLOUR = "#636b83"
ACCENT_COLOUR = "#6ee7b7"

GAZE_COLOUR_OBJECT = "#fbbf24"   # yellow when looking at the object
GAZE_COLOUR_TARGET = "#a78bfa"   # purple when looking at the target slot
GOAL_COLOUR = "#fbbf24"

# Position of the eye icon in each panel (top-left)
EYE_X = -0.55
EYE_Y = 1.05


def build_panel_label(params: DevelopmentalParams) -> str:
    """Build a compact parameter summary string for use as a panel title.

    Args:
        params: developmental stage parameters to summarise.

    Returns:
        Multi-line string showing key parameter values.
    """
    habit_percentage = int(
        100 * params.habit_strength / (params.habit_strength + params.goal_directed_strength)
    )
    return (
        f"samp={params.sampling_rate:.2f}  noise={params.perceptual_noise:.2f}  "
        f"gaze_sw={params.gaze_switch_rate:.2f}\n"
        f"WM={params.wm_capacity}  horizon={params.planning_horizon}  "
        f"habit={habit_percentage}%  corr={params.correction_rate:.2f}"
    )


def run_one_trial_per_stage(task_name: str, seed: int = 42) -> tuple:
    """Run one trial per developmental stage and return all results.

    Running all four stages with the same seed and task makes the animation
    a direct developmental comparison: any differences in trajectory, timing,
    or gaze behaviour are attributable to the parameter differences across stages.

    Args:
        task_name: key into the TASKS dict.
        seed: random seed for reproducibility.

    Returns:
        (results, task):
            results: dict mapping stage name ("A"–"D") to TrialResult.
            task: the TaskConfig used for all trials.
    """
    task = TASKS[task_name]
    results = {}
    for stage_name in STAGES:
        results[stage_name] = run_trial(DEVELOPMENTAL_STAGES[stage_name], task, seed=seed)
    return results, task


def draw_rotated_rectangle(ax, centre_x, centre_y, width, height, angle, colour, **kwargs):
    """Draw a rotated rounded rectangle centred at (centre_x, centre_y).

    Args:
        ax: matplotlib Axes to draw on.
        centre_x, centre_y: centre coordinates of the rectangle.
        width, height: dimensions of the rectangle.
        angle: rotation angle in radians.
        colour: edge colour.
        **kwargs: optional overrides — fc (facecolor), lw (linewidth),
                  ls (linestyle), alpha, z (zorder).

    Returns:
        The FancyBboxPatch that was added to the axes.
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
    rect.set_transform(
        transforms.Affine2D().rotate(angle).translate(centre_x, centre_y) + ax.transData
    )
    ax.add_patch(rect)
    return rect


def setup_figure(task: TaskConfig) -> tuple:
    """Create the figure, axes, and static elements shared across all frames.

    Returns:
        (fig, axes, trails, status_texts, timestep_text)
            fig: the Figure object.
            axes: list of 4 Axes, one per stage.
            trails: dict mapping stage name to the trail Line2D object.
            status_texts: dict mapping stage name to the status Text object.
            timestep_text: the figure-level Text showing current timestep.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5), facecolor=BACKGROUND_COLOUR)
    fig.subplots_adjust(left=.03, right=.98, top=.82, bottom=.04, wspace=.15)

    fig.text(
        0.5, 0.97, "Planning Model Simulation",
        ha="center", va="top", fontsize=16, fontweight="bold",
        color=ACCENT_COLOUR, fontfamily="serif",
        path_effects=[patheffects.withStroke(linewidth=2, foreground=BACKGROUND_COLOUR)],
    )
    fig.text(
        0.5, 0.935,
        f"Task: {task.name.replace('_', ' ').title()}  |  "
        f"Fit into slot at ({task.goal_x},{task.goal_y}) "
        f"rotated {np.degrees(task.goal_angle):.0f}°",
        ha="center", va="top", fontsize=9, color=DIM_COLOUR,
    )
    timestep_text = fig.text(0.5, 0.905, "", ha="center", fontsize=9, color=DIM_COLOUR)

    trails = {}
    status_texts = {}

    for panel_index, stage_name in enumerate(STAGES):
        ax = axes[panel_index]
        ax.set_facecolor(SURFACE_COLOUR)
        ax.set_xlim(-0.7, 1.1)
        ax.set_ylim(-0.6, 1.25)
        ax.set_aspect("equal")
        ax.tick_params(colors=DIM_COLOUR, labelsize=5)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOUR)

        ax.set_title(
            build_panel_label(DEVELOPMENTAL_STAGES[stage_name]),
            fontsize=6.2, fontfamily="monospace",
            color=STAGE_COLOURS[stage_name], pad=5, linespacing=1.3,
        )

        # Static goal slot drawn once — the object must end up here.
        draw_rotated_rectangle(
            ax, task.goal_x, task.goal_y,
            task.obj_width + 0.05, task.obj_height + 0.05,
            task.goal_angle, GOAL_COLOUR,
            fc="none", lw=1.8, ls="--", alpha=0.5, z=1,
        )
        ax.text(
            task.goal_x, task.goal_y + task.obj_height / 2 + 0.08, "target",
            fontsize=6, ha="center", color=f"{GOAL_COLOUR}88", zorder=1,
        )

        # Trail line: accumulates the object's path across all frames.
        trail_line, = ax.plot([], [], color=STAGE_COLOURS[stage_name], alpha=0.4, lw=1.2, zorder=2)
        trails[stage_name] = trail_line

        # Status text in the bottom-right corner of each panel.
        status_text = ax.text(
            0.97, 0.03, "", transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom",
            fontfamily="monospace", color=TEXT_COLOUR, zorder=15,
        )
        status_texts[stage_name] = status_text

    return fig, axes, trails, status_texts, timestep_text


def remove_dynamic_artists(ax) -> None:
    """Remove all artists tagged with _d=True from the axes.

    Dynamic artists (gaze lines, eye icons, object rectangles) are recreated
    every frame. Tagging them with _d=True and removing them at the start of
    each frame is simpler than storing and updating references to each one.

    Args:
        ax: matplotlib Axes to clean.
    """
    for artist in list(ax.patches) + list(ax.texts) + list(ax.lines):
        if getattr(artist, "_d", False):
            artist.remove()


def draw_frame_for_stage(
    ax, stage_name, trial, frame_index, task, trails, status_texts
) -> None:
    """Draw all dynamic elements for one stage panel at a given frame.

    Args:
        ax: the Axes for this stage panel.
        stage_name: e.g. "A", "B", "C", "D".
        trial: the TrialResult for this stage.
        frame_index: the current animation frame.
        task: the TaskConfig (for goal position).
        trails: dict of stage_name → trail Line2D.
        status_texts: dict of stage_name → status Text.
    """
    trajectory = trial.trajectory
    clamped_index = min(frame_index, len(trajectory) - 1)
    timestep = trajectory[clamped_index]

    remove_dynamic_artists(ax)

    # Update movement trail with all positions up to current frame.
    trail_x = [trajectory[i].obj_x for i in range(clamped_index + 1)]
    trail_y = [trajectory[i].obj_y for i in range(clamped_index + 1)]
    trails[stage_name].set_data(trail_x, trail_y)

    # Draw the object rectangle at its current pose.
    object_rect = draw_rotated_rectangle(
        ax, timestep.obj_x, timestep.obj_y,
        task.obj_width, task.obj_height,
        timestep.obj_angle, STAGE_COLOURS[stage_name], alpha=0.88,
    )
    object_rect._d = True

    # Draw the gaze visualisation: eye icon, gaze line, and fixation dot.
    # Colour encodes which entity is being fixated — yellow for object, purple for target.
    looking_at_object = timestep.gaze_target == "object"
    gaze_colour = GAZE_COLOUR_OBJECT if looking_at_object else GAZE_COLOUR_TARGET
    fixation_x = timestep.obj_x if looking_at_object else task.goal_x
    fixation_y = timestep.obj_y if looking_at_object else task.goal_y

    eye_circle = plt.Circle((EYE_X, EYE_Y), 0.035, fc=gaze_colour, ec="none", alpha=0.85, zorder=10)
    eye_circle._d = True
    ax.add_patch(eye_circle)

    pupil = plt.Circle((EYE_X, EYE_Y), 0.015, fc=BACKGROUND_COLOUR, ec="none", alpha=0.9, zorder=11)
    pupil._d = True
    ax.add_patch(pupil)

    gaze_line, = ax.plot(
        [EYE_X, fixation_x], [EYE_Y, fixation_y],
        color=gaze_colour, lw=1.0, ls="-", alpha=0.45, zorder=5,
    )
    gaze_line._d = True

    fixation_dot = plt.Circle(
        (fixation_x, fixation_y), 0.025,
        fc=gaze_colour, ec="white", lw=0.4, alpha=0.7, zorder=8,
    )
    fixation_dot._d = True
    ax.add_patch(fixation_dot)

    gaze_label = ax.text(
        EYE_X + 0.08, EYE_Y,
        "obj" if looking_at_object else "tgt",
        fontsize=5.5, color=gaze_colour, va="center",
        fontfamily="monospace", alpha=0.8, zorder=12,
    )
    gaze_label._d = True

    # Update the status text: show live error during the trial, outcome after it ends.
    if frame_index >= len(trajectory):
        if trial.success:
            status_texts[stage_name].set_text("FITTED")
            status_texts[stage_name].set_color(ACCENT_COLOUR)
        else:
            status_texts[stage_name].set_text("MISSED")
            status_texts[stage_name].set_color("#fb7185")
    else:
        status_texts[stage_name].set_text(
            f"pos={timestep.pos_error:.2f} ang={timestep.angle_error:.2f}"
        )
        status_texts[stage_name].set_color(DIM_COLOUR)


def animate(task_name: str = "rotate_insert", save: bool = False, seed: int = 42):
    """Build and run the matplotlib animation.

    Args:
        task_name: key into the TASKS dict.
        save: if True, save as GIF; if False, show an interactive window.
        seed: random seed.

    Returns:
        The FuncAnimation object.
    """
    results, task = run_one_trial_per_stage(task_name, seed)

    max_trial_length = max(len(trial.trajectory) for trial in results.values())
    num_frames = max_trial_length + 6  # a few extra frames to hold the final state

    fig, axes, trails, status_texts, timestep_text = setup_figure(task)

    def update(frame):
        """Draw one frame of the animation."""
        timestep_text.set_text(f"t = {min(frame, max_trial_length - 1)}")
        for panel_index, stage_name in enumerate(STAGES):
            draw_frame_for_stage(
                axes[panel_index], stage_name, results[stage_name],
                frame, task, trails, status_texts,
            )
        return []

    animation = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False, repeat=True)

    if save:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"planning_cascade_{task_name}.gif",
        )
        print(f"Saving -> {output_path}")
        animation.save(output_path, writer="pillow", fps=5, savefig_kwargs={"facecolor": BACKGROUND_COLOUR})
        print(f"Done: {output_path}")
    else:
        plt.show()

    return animation


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task", default="rotate_insert", choices=list(TASKS.keys()))
    arg_parser.add_argument("--save", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    animate(args.task, args.save, args.seed)
