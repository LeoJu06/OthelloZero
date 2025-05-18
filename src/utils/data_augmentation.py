import numpy as np
import random

def rotate_board(board, k):
    """Rotates the board by 90° * k counter-clockwise."""
    return np.rot90(board, k)

def flip_board(board, axis):
    """Flips the board along the specified axis (0=horizontal, 1=vertical)."""
    return np.flip(board, axis=axis)

def transform_policy_with_pass(policy, transform_fn):
    """
    Transforms the policy (64 board values + 1 pass value),
    keeping the pass value unchanged.
    """
    board_policy = policy[:-1].reshape(8, 8)
    transformed = transform_fn(board_policy)
    return np.append(transformed.flatten(), policy[-1])

def random_augment_data(data, augment_prob=0.75):
    """
    Augments data randomly through symmetry transformations.
    Each sample has `augment_prob` chance of being augmented.
    Returns a new list with the same length as input.
    """
    augmented = []
    
    for state, policy, value in data:
        if random.random() < augment_prob:
            # Randomly choose a transformation
            k = random.randint(0, 3)  # Rotation (0-3)
            flip = random.choice([True, False])  # Whether to flip
            
            # Apply rotation
            state = rotate_board(state, k)
            policy = transform_policy_with_pass(policy, lambda b: rotate_board(b, k))
            
            # Optionally apply flip
            if flip:
                axis = random.choice([0, 1])  # 0=horizontal, 1=vertical
                state = flip_board(state, axis)
                policy = transform_policy_with_pass(policy, lambda b: flip_board(b, axis))
        
        augmented.append((state, policy, value))
    
    return augmented