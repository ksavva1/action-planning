"""Affordance-layer weight matrix variants.

Rows are working-memory features and columns are action affordances. Keeping the
variants in one file makes it easier to compare psychological assumptions about
which information should drive reach, grasp, rotate, and translate actions.
"""

import numpy as np

FEATURE_LABELS = [
    "object_x",
    "object_y",
    "object_angle",
    "object_width",
    "object_height",
    "target_x",
    "target_y",
    "target_angle",
    "rel_dx",
    "rel_dy",
    "rel_d_angle",
]
ACTION_LABELS = ["reach", "grasp", "rotate", "translate"]

# Baseline mapping: relational rows strongly drive translate and rotate because
# they represent the gap between current object pose and target pose.
BASELINE_MATRIX = np.array([
    [.6, .1, 0, .5],
    [.6, .1, 0, .5],
    [0, .2, .7, 0],
    [.1, .6, .3, .1],
    [.1, .6, .3, .1],
    [.3, 0, 0, .4],
    [.3, 0, 0, .4],
    [0, .1, .5, 0],
    [.2, 0, .1, .8],
    [.2, 0, .1, .8],
    [0, .1, .9, 0],
])

# Alternative matrices support targeted experiments in what information the
# affordance layer treats as action-relevant. They keep the same shape so only
# the psychological mapping assumption changes between runs.
AFFORDANCE_MATRIX_VARIANTS = {
    "baseline": BASELINE_MATRIX,
    # Object-dominant mapping: local object pose and size drive action more than
    # object-target comparison, modelling a less relational action strategy.
    "object_dominant": np.array([
        [.8, .1, 0, .7],
        [.8, .1, 0, .7],
        [0, .2, .9, 0],
        [.1, .8, .3, .1],
        [.1, .8, .3, .1],
        [.2, 0, 0, .25],
        [.2, 0, 0, .25],
        [0, .05, .25, 0],
        [.1, 0, .05, .35],
        [.1, 0, .05, .35],
        [0, .05, .35, 0],
    ]),
    # Relational-dominant mapping: gap features strongly drive translate/rotate,
    # testing a more comparison-based planning strategy.
    "relational_dominant": np.array([
        [.35, .05, 0, .25],
        [.35, .05, 0, .25],
        [0, .1, .35, 0],
        [.05, .45, .15, .05],
        [.05, .45, .15, .05],
        [.2, 0, 0, .25],
        [.2, 0, 0, .25],
        [0, .05, .35, 0],
        [.3, 0, .1, 1.0],
        [.3, 0, .1, 1.0],
        [0, .05, 1.0, 0],
    ]),
    # Diffuse mapping: features activate several actions weakly, modelling a less
    # differentiated perception-action mapping.
    "diffuse": np.array([
        [.45, .25, .2, .45],
        [.45, .25, .2, .45],
        [.15, .25, .45, .2],
        [.25, .45, .35, .25],
        [.25, .45, .35, .25],
        [.35, .15, .15, .4],
        [.35, .15, .15, .4],
        [.15, .2, .4, .15],
        [.3, .15, .25, .55],
        [.3, .15, .25, .55],
        [.15, .2, .6, .15],
    ]),
}


def get_affordance_matrix(variant: str) -> np.ndarray:
    """Return the matrix for a named variant or raise a helpful error."""

    if variant not in AFFORDANCE_MATRIX_VARIANTS:
        variants = ", ".join(sorted(AFFORDANCE_MATRIX_VARIANTS))
        raise ValueError(f"Unknown affordance_matrix_variant {variant!r}; choose one of: {variants}")
    return AFFORDANCE_MATRIX_VARIANTS[variant]
