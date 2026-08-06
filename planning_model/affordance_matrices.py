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
BASELINE_MATRIX = np.array(
    [
        # visual object features
        [0.6, 0.1, 0, 0.5],  # object_x
        [0.6, 0.1, 0, 0.5],  # object_y
        [0, 0.2, 0.7, 0],  # object_angle
        [0.1, 0.6, 0.3, 0.1],  # object_width
        [0.1, 0.6, 0.3, 0.1],  # object_height
        # visual target features
        [0.3, 0, 0, 0.4],  # target_x
        [0.3, 0, 0, 0.4],  # target_y
        [0, 0.1, 0.5, 0],  # target_angle
        # relational features
        [0.2, 0, 0.1, 0.8],  # rel_dx
        [0.2, 0, 0.1, 0.8],  # rel_dy
        [0, 0.1, 0.9, 0],  # rel_d_angle
    ]
)

# Alternative matrices
AFFORDANCE_MATRIX_VARIANTS = {
    "baseline": BASELINE_MATRIX,
    # Object-dominant: local object position and size drive action more than
    # object-target comparison, modelling a less relational action strategy.
    "object_dominant": np.array(
        [
            [0.8, 0.1, 0, 0.7],  # object_x
            [0.8, 0.1, 0, 0.7],  # object_y
            [0, 0.2, 0.9, 0],  # object_angle
            [0.1, 0.8, 0.3, 0.1],  # object_width
            [0.1, 0.8, 0.3, 0.1],  # object_height
            [0.2, 0, 0, 0.25],  # target_x
            [0.2, 0, 0, 0.25],  # target_y
            [0, 0.05, 0.25, 0],  # target_angle
            [0.1, 0, 0.05, 0.35],  # rel_dx
            [0.1, 0, 0.05, 0.35],  # rel_dy
            [0, 0.05, 0.35, 0],  # rel_d_angle
        ]
    ),
    # Relational-dominant: gap features strongly drive translate/rotate,
    # testing a more comparison-based planning strategy.
    "relational_dominant": np.array(
        [
            [0.35, 0.05, 0, 0.25],  # object_x
            [0.35, 0.05, 0, 0.25],  # object_y
            [0, 0.1, 0.35, 0],  # object_angle
            [0.05, 0.45, 0.15, 0.05],  # object_width
            [0.05, 0.45, 0.15, 0.05],  # object_height
            [0.2, 0, 0, 0.25],  # target_x
            [0.2, 0, 0, 0.25],  # target_y
            [0, 0.05, 0.35, 0],  # target_angle
            [0.3, 0, 0.1, 1.0],  # rel_dx
            [0.3, 0, 0.1, 1.0],  # rel_dy
            [0, 0.05, 1.0, 0],  # rel_d_angle
        ]
    ),
    # Diffuse: features activate several actions weakly, modelling a less
    # differentiated perception-action mapping.
    "diffuse": np.array(
        [
            [0.45, 0.25, 0.2, 0.45],  # object_x
            [0.45, 0.25, 0.2, 0.45],  # object_y
            [0.15, 0.25, 0.45, 0.2],  # object_angle
            [0.25, 0.45, 0.35, 0.25],  # object_width
            [0.25, 0.45, 0.35, 0.25],  # object_height
            [0.35, 0.15, 0.15, 0.4],  # target_x
            [0.35, 0.15, 0.15, 0.4],  # target_y
            [0.15, 0.2, 0.4, 0.15],  # target_angle
            [0.3, 0.15, 0.25, 0.55],  # rel_dx
            [0.3, 0.15, 0.25, 0.55],  # rel_dy
            [0.15, 0.2, 0.6, 0.15],  # rel_d_angle
        ]
    ),
    # Goal-directed: relational gap features dominate, but object features
    # still provide a grounding signal for reach and grasp initiation.
    # Models a strategy that uses strong goal-comparison for movement control.
    "goal_directed": np.array(
        [
            [0.5, 0.1, 0, 0.4],  # object_x
            [0.5, 0.1, 0, 0.4],  # object_y
            [0, 0.15, 0.5, 0],  # object_angle
            [0.1, 0.5, 0.2, 0.1],  # object_width
            [0.1, 0.5, 0.2, 0.1],  # object_height
            [0.25, 0, 0, 0.3],  # target_x
            [0.25, 0, 0, 0.3],  # target_y
            [0, 0.1, 0.4, 0],  # target_angle
            [0.15, 0, 0.1, 0.6],  # rel_dx
            [0.15, 0, 0.1, 0.6],  # rel_dy
            [0, 0.1, 0.7, 0],  # rel_d_angle
        ]
    ),
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
        raise ValueError(
            f"Unknown affordance_matrix_variant {variant!r}; choose one of: {variants}"
        )
    return AFFORDANCE_MATRIX_VARIANTS[variant]
