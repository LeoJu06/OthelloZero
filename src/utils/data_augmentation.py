import numpy as np

def rotate_board(board, k):
    """Rotates the board by 90° * k counter-clockwise."""
    return np.rot90(board, k)

def flip_board_horizontal(board):
    """Flips the board horizontally (left ↔ right)."""
    return np.fliplr(board)

def flip_board_vertical(board):
    """Flips the board vertically (top ↔ bottom)."""
    return np.flipud(board)

def transform_policy_with_pass(policy, transform_fn):
    """
    Transforms the policy (64 board values + 1 pass value),
    keeping the pass value unchanged.
    """
    board_policy = policy[:-1].reshape(8, 8)        # Only board policy
    transformed = transform_fn(board_policy)       # e.g., rotate or flip
    transformed_flat = transformed.flatten()
    pass_prob = policy[-1]                         # Last entry remains unchanged
    return np.append(transformed_flat, pass_prob)

def augment_data(data):
    """
    Augments a list of (state, policy, value) triples through symmetry transformations.
    Returns a new list with 8× as many entries.
    """
    augmented = []

    for state, policy, value in data:
        for k in range(4):  # Rotation 0°, 90°, 180°, 270°
            rot_state = rotate_board(state, k)
            rot_policy = transform_policy_with_pass(policy, lambda b: rotate_board(b, k))
            augmented.append((rot_state, rot_policy, value))

            # Horizontally flipped after rotation
            flip_state = flip_board_horizontal(rot_state)
            flip_policy = transform_policy_with_pass(rot_policy, flip_board_horizontal)
            augmented.append((flip_state, flip_policy, value))

    return augmented