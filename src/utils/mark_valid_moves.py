def mark_valid_moves(action_probs, valid_moves_flattened):
    """
    Marks the valid moves in the action probabilities array by setting 
    the probability of invalid moves to zero.

    Args:
        action_probs (numpy.ndarray): A 1D array containing action probabilities.
                                       The length of the array should correspond to the total number of possible actions.
        valid_moves_flattened (numpy.ndarray): A 1D array of the same length as `action_probs`,
                                                where each element is either 1 (valid move) or 0 (invalid move).

    Returns:
        numpy.ndarray: A 1D array where the action probabilities are updated such that 
                        invalid moves (corresponding to 0s in `valid_moves_flattened`) 
                        are set to 0, while valid moves (corresponding to 1s in `valid_moves_flattened`) 
                        retain their original probability.

    Example:
        action_probs = np.array([0.1, 0.3, 0.2, 0.4])
        valid_moves_flattened = np.array([1, 0, 1, 1])
        marked_probs = mark_valid_moves(action_probs, valid_moves_flattened)
        print(marked_probs)  # Output: [0.1 0.  0.2 0.4]
    """
    
    return action_probs * valid_moves_flattened
