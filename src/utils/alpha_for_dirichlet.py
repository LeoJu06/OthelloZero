def alpha_for_dirichlet(n_valid_moves: int) -> float:
    """
    Computes the alpha parameter for Dirichlet noise based on the number of valid moves.

    The alpha parameter controls the concentration of the Dirichlet distribution, which is used
    to add exploration noise to the action probabilities during the MCTS search. This helps
    encourage exploration of less-visited actions.

    Args:
        n_valid_moves (int): The number of valid moves available in the current game state.

    Returns:
        float: The alpha parameter for the Dirichlet distribution. If there are valid moves,
               the alpha is set to the minimum of 1 or 10 divided by the number of valid moves.
               If there are no valid moves, the alpha is set to 10.

    Notes:
        - The alpha parameter is inversely proportional to the number of valid moves. This ensures
          that the noise is more concentrated when there are fewer valid moves, promoting exploration.
        - When no valid moves are available, the alpha is set to a default value of 10 to avoid
          division by zero and ensure meaningful noise is added.
    """
    if n_valid_moves > 0:
        return min(1, 10 / n_valid_moves)
    else:
        return 10