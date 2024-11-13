from src.othello.board import Board
import src.othello.game_constants as const


def test_board():
    # Init empty board
    board = Board(const.EMPTY_BOARD)

    # List of starting fields
    start_positions = [(3, 4), (3, 3), (4, 3), (4, 4)]

    # Check whether the starting positions are empty at the beginning and do not contain any player pieces
    for x, y in start_positions:
        assert board.board[x][y] == const.EMPTY_CELL, f"({x}|{y}) should be empty"
        assert board.board[x][y] not in [
            const.PlayerColor.BLACK.value,
            const.PlayerColor.WHITE.value,
        ], f"({x}|{y}) should not be occupied by any player"

        # Apply move
        board.apply_move(x, y)

    # Berechne die gültigen Züge nach den gesetzten Startpositionen
    valid_moves = board.valid_moves()

    # Expected valid moves
    expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]

    # Assertion to ensure that the calculated valid moves are correct
    assert (
        valid_moves == expected_moves
    ), f"Expected valid moves {expected_moves}, but got {valid_moves}"

    # New game situation: An additional move that changes the game situation
    new_move = (2, 3)  # Beispielzug, der auf das Board angewendet wird
    board.apply_move(*new_move)

    # Calculate new valid moves
    new_valid_moves = board.valid_moves()

    # Expected valid movess after adding a peace
    new_expected_moves = [(2, 2), (2, 4), (4, 2)]

    # Assertion for new board state
    assert (
        new_valid_moves == new_expected_moves
    ), f"Expected new valid moves {new_expected_moves}, but got {new_valid_moves}"
