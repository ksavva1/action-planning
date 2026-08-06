"""Affordance-layer weight matrix variants.

Rows are working-memory features and columns are action affordances. Use to compare 
assumptions about which information should drive reach, grasp, rotate,
and translate actions.

Feature vector layout (11 features total):
visual object features   — object_x, object_y, object_angle,
                           object_width, object_height
visual target features   — target_x, target_y, target_angle
relational features      — rel_dx, rel_dy, rel_d_angle
"""

import numpy as np

FEATURE_LABELS = [
    # Visual object features, sampled when fixating the object.
    "object_x",
    "object_y",
    "object_angle",
    "object_width",
    "object_height",
    # Visual target features, sampled when fixating the target slot.
    "target_x",
    "target_y",
    "target_angle",
    # Relational features, sampled when both entities have been recently fixated.
    "rel_dx",
    "rel_dy",
    "rel_d_angle",
]
ACTION_LABELS = ["reach", "grasp", "rotate", "translate"]

# Baseline matrix
BASELINE_MATRIX = np.array([
    # visual object features
    [.6, .1, 0,  .5],   # object_x
    [.6, .1, 0,  .5],   # object_y
    [0,  .2, .7, 0 ],   # object_angle
    [.1, .6, .3, .1],   # object_width
    [.1, .6, .3, .1],   # object_height
    # visual target features
    [.3, 0,  0,  .4],   # target_x
    [.3, 0,  0,  .4],   # target_y
    [0,  .1, .5, 0 ],   # target_angle
    # relational features
    [.2, 0,  .1, .8],   # rel_dx
    [.2, 0,  .1, .8],   # rel_dy
    [0,  .1, .9, 0 ],   # rel_d_angle
])

# Alternative matrices
AFFORDANCE_MATRIX_VARIANTS = {
    "baseline": BASELINE_MATRIX,

    # Object-dominant: local object position and size drive action more than
    # object-target comparison, modelling a less relational action strategy.
    "object_dominant": np.array([
        [.8, .1, 0,  .7],   # object_x
        [.8, .1, 0,  .7],   # object_y
        [0,  .2, .9, 0 ],   # object_angle
        [.1, .8, .3, .1],   # object_width
        [.1, .8, .3, .1],   # object_height
        [.2, 0,  0,  .25],  # target_x
        [.2, 0,  0,  .25],  # target_y
        [0,  .05,.25, 0],   # target_angle
        [.1, 0,  .05,.35],  # rel_dx
        [.1, 0,  .05,.35],  # rel_dy
        [0,  .05,.35, 0],   # rel_d_angle
    ]),

    # Relational-dominant: gap features strongly drive translate/rotate,
    # testing a more comparison-based planning strategy.
    "relational_dominant": np.array([
        [.35,.05, 0,  .25],  # object_x
        [.35,.05, 0,  .25],  # object_y
        [0,  .1, .35, 0 ],   # object_angle
        [.05,.45,.15,.05],   # object_width
        [.05,.45,.15,.05],   # object_height
        [.2, 0,  0,  .25],   # target_x
        [.2, 0,  0,  .25],   # target_y
        [0,  .05,.35, 0 ],   # target_angle
        [.3, 0,  .1, 1.0],   # rel_dx
        [.3, 0,  .1, 1.0],   # rel_dy
        [0,  .05,1.0, 0 ],   # rel_d_angle
    ]),

    # Diffuse: features activate several actions weakly, modelling a less
    # differentiated perception-action mapping.
    "diffuse": np.array([
        [.45,.25,.2, .45],   # object_x
        [.45,.25,.2, .45],   # object_y
        [.15,.25,.45,.2 ],   # object_angle
        [.25,.45,.35,.25],   # object_width
        [.25,.45,.35,.25],   # object_height
        [.35,.15,.15,.4 ],   # target_x
        [.35,.15,.15,.4 ],   # target_y
        [.15,.2, .4, .15],   # target_angle
        [.3, .15,.25,.55],   # rel_dx
        [.3, .15,.25,.55],   # rel_dy
        [.15,.2, .6, .15],   # rel_d_angle
    ]),

    # Goal-directed: relational gap features dominate, but object features
    # still provide a grounding signal for reach and grasp initiation.
    # Models a strategy that uses strong goal-comparison for movement control.
    "goal_directed": np.array([
        [.5, .1, 0,  .4],    # object_x
        [.5, .1, 0,  .4],    # object_y
        [0,  .15,.5, 0 ],    # object_angle
        [.1, .5, .2, .1],    # object_width
        [.1, .5, .2, .1],    # object_height
        [.25, 0, 0,  .3],    # target_x
        [.25, 0, 0,  .3],    # target_y
        [0,  .1, .4, 0 ],    # target_angle
        [.15, 0, .1, .6],    # rel_dx
        [.15, 0, .1, .6],    # rel_dy
        [0,  .1, .7, 0 ],    # rel_d_angle
    ]),
}


def get_affordance_matrix(variant: str) -> np.ndarray:
    """Look up an affordance weight matrix by name.

    Args:
        variant: key into ``AFFORDANCE_MATRIX_VARIANTS``.

    Returns:
        The corresponding weight matrix.

    Raises:
        ValueError: if ``variant`` is not a known matrix name.
    """

    if variant not in AFFORDANCE_MATRIX_VARIANTS:
        variants = ", ".join(sorted(AFFORDANCE_MATRIX_VARIANTS))
        raise ValueError(f"Unknown affordance_matrix_variant {variant!r}; choose one of: {variants}")
    return AFFORDANCE_MATRIX_VARIANTS[variant]