def coordinates_to_index(row, col, board_size=8):
    """
    Converts 2D coordinates on a board into a single linear index.

    Parameters:
        row (int): The row coordinate (0-based).
        col (int): The column coordinate (0-based).
        board_size (int): The size of one side of the board (default is 8 for Othello).

    Returns:
        int: The linear index of the cell.
    """
    return row * board_size + col

if __name__ == "__main__":
    # Examples
    print(coordinates_to_index(0, 0))  # Output: 0
    print(coordinates_to_index(7, 7))  # Output: 63
