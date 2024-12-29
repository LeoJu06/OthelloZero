from src.othello.board import Board
import src.othello.game_constants as const


def test_board():
    # Init empty board
    board = Board(const.EMPTY_BOARD)

    # New game situation: An additional move that changes the game situation
    new_move = (2, 3)  # Beispielzug, der auf das Board angewendet wird
    board.apply_move(*new_move)
    board.update(*new_move)

    # Calculate new valid moves
    new_valid_moves = board.valid_moves()

    # Expected valid movess after adding a peace
    new_expected_moves = [(2, 2), (2, 4), (4, 2)]

    # Assertion for new board state
    assert (
        new_valid_moves == new_expected_moves
    ), f"Expected new valid moves {new_expected_moves}, but got {new_valid_moves}"


def test_board_is_terminal_state():
    board = Board(const.EMPTY_BOARD)

    for x in range(8):
        for y in range(8):
            board.board[x][y] = const.PlayerColor.BLACK.value

    assert board.is_terminal_state() == True, "Board should be terminal"
