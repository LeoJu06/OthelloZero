from src.othello.game_constants import PASSING_MOVE

def index_to_coordinates(index, board_size=8):
    """
    Converts a single number into 2D coordinates on a board of given size.

    Parameters:
        index (int): The linear index of the cell (0-based).
        board_size (int): The size of one side of the board (default is 8 for Othello).

    Returns:
        tuple: A tuple (row, col) representing the 2D coordinates.
    """
    if index == -1:
        return PASSING_MOVE, PASSING_MOVE

    row = index // board_size
    col = index % board_size
    return row, col


if __name__ == "__main__":
    # Examples
    print(index_to_coordinates(0))  # Output: (0, 0)
    print(index_to_coordinates(63))  # Output: (7. 7)
