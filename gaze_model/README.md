# Planning Model

A connectionist computational model of infant action planning in object manipulation tasks. Simulates how infants gather information by looking back and forth between an object and a target, then plan and execute movement.

## Background

The model captures:

- **Translation bias**: younger parameter configurations default to translating the object toward the target before rotating it, whereas more mature configurations rotate and translate simultaneously.
- **Relational information**: displacement and angular difference between object and target can only be extracted when both have been recently fixated, governed by a `simultaneous_rate` parameter.
- **Gradual development**: a shift from reactive correction to partial planning to full anticipatory control, driven by increases in planning horizon, gaze switch rate, and working memory capacity.

## Architecture

The network consists of six interconnected modules:

```
LOOK                    PROCESS                 ACT
┌──────────┐           ┌──────────────┐        ┌──────────────┐
│  Gaze    │──────────▶│   Working    │───────▶│    Motor     │
│Controller│           │   Memory     │        │   Planning   │
│          │           └──────┬───────┘        └──────┬───────┘
│ object ◀─┼─▶ target         │                       │
└──────────┘           ┌──────▼───────┐        ┌──────▼───────┐
      │                │  Affordance  │        │   Habitual   │
      │                │  Estimation  │        │     Bias     │
      │                └──────────────┘        └──────┬───────┘
      │                                               │
      │                                        ┌──────▼───────┐
      │                                        │    Online    │
      └────────────────────────────────────────│  Correction  │
                     feedback                  └──────────────┘
```

**Gaze Controller** - switches fixation between the held object and the target with a hazard-function model. Parameters: `gaze_switch_rate`, `fixation_duration_mean`, `target_bias`.

**Perceptual Sampling** - Object features (x, y, angle, width, height) are sampled when looking at the object; target features (goal x, y, angle) when looking at the target. Relational features (dx, dy, d_angle) require recent fixation of both entities.

**Working Memory** - Maintains separate traces for object, target, and relational features with capacity limits. Traces of the non-fixated entity decay faster (`wm_unfixated_decay`).

**Affordance Estimation** - A learned weight matrix mapping the working memory state to four affordances: reach, grasp, rotate, translate.

**Motor Planning** - Generates motor commands with a variable lookahead horizon. Near the goal, blends toward a direct proportional controller to prevent overshoot.

**Habitual Bias** - Encodes the translate-first default. The habitual phase length scales with `habit_strength`, producing the shift from sequential to simultaneous movement.

**Online Correction** — delayed, error-driven adjustments that improve with development.

## Files

| File | Description |
|------|-------------|
| `planning_cascade_model.py` | Network layers, task definitions, simulation engine, and batch runner. Generates `simulation_results.json` and should be run first. |
| `animate_model.py` | Matplotlib animation showing 4 parameter configurations performing a task simultaneously. Displays the object, target slot, gaze line, and movement trace. |
| `hyperparam_sweep.py` | Systematic parameter sweep. Supports 1D sweeps, 2D sweeps (with heatmaps), and full grid sweeps. |
| `planning_cascade_dashboard.html` | HTML dashboard with architecture diagram, parameter profiles, simulation results, gaze statistics, and trajectory views. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```


## Usage

### Run simulation

```bash
python planning_cascade_model.py
```

This runs 25 trials per condition across 4 parameter configurations (A–D) and 4 tasks, writing results to `simulation_results.json` in the same directory.

### Animate the model

Animated output:
```bash
python animate_model.py
```

Save as GIF:
```bash
python animate_model.py --save
```

### Hyperparameter sweep

**Single parameter sweep** - vary one parameter and plot its effect:
```bash
python hyperparam_sweep.py --param gaze_switch_rate --values 0.1 0.3 0.5 0.7
```

**Two-param sweep** - produces heatmaps:
```bash
python hyperparam_sweep.py \
    --param gaze_switch_rate --values 0.1 0.25 0.4 0.55 0.7 \
    --param2 habit_strength --values2 0.1 0.3 0.5 0.7 0.9
```

**Default 4-param grid** (gaze_switch_rate × habit_strength × planning_horizon × wm_capacity):
```bash
python hyperparam_sweep.py --trials 15
```

## Parameter Configurations

Four default configs:

| Config | Strategy | Habit | Horizon | WM | Gaze Switch | 
|--------|----------|-------|---------|-----|-------------|
| A | Translate then rotate | High (78%) | 1 | 2 | Low (0.12) |
| B | Mostly sequential | 70% | 2 | 3 | 0.25 |
| C | Partially simultaneous | 40% | 4 | 4 | 0.40 |
| D | Fully simultaneous | Low (10%) | 6 | 5 | High (0.55) |


## Custom sweep params

Any field of `DevelopmentalParams` can be swept:

```bash
python hyperparam_sweep.py --param correction_rate --values 0.05 0.1 0.15 0.2 0.25 0.3
python hyperparam_sweep.py --param wm_decay --values 0.05 0.1 0.2 0.3 --param2 sampling_rate --values2 0.2 0.4 0.6 0.8
```
