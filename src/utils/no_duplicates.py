def filter_duplicates(examples):
    """
    Removes duplicate training examples from a list of examples.
    Each example is a list of [board, policy, value], where:
        - board: 8x8 numpy array or nested list representing the Othello board state.
        - policy: List/array of move probabilities.
        - value: Float representing the game outcome (-1 to 1).

    Args:
        examples: List of training examples to filter.

    Returns:
        List of unique training examples with duplicates removed.
    """
    
    combined = []  # Will store the deduplicated examples
    seen = set()   # Tracks unique examples via hashable tuples

    for example in examples:
        # Convert the example to a hashable tuple format:
        # 1. Convert the board (example[0]) to a tuple of tuples for hashability
        board_tuple = tuple(map(tuple, example[0]))
        
        # 2. Convert the policy (example[1]) to a tuple
        policy_tuple = tuple(example[1])
        
        # 3. Value (example[2]) remains as-is (float/int is already hashable)
        value = example[2]

        # Combine all components into a single tuple
        example_tuple = (board_tuple, policy_tuple, value)

        # Check if this unique configuration has been seen before
        if example_tuple not in seen:
            # If new, add to tracking set and keep the original example
            seen.add(example_tuple)
            combined.append(example)  # Keep original format (not the tuple)

    return combined