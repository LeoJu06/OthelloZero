import sys
import numpy as np
import src.othello.game_constants as const
import src.utils.logger_config as lg
import time


class Board:
    """The Board class represents the Othello game board and manages all game-related operations."""

    def __init__(
        self, board=None, player=const.PlayerColor.BLACK.value, empty_cells=None
    ):
        if board is None:
            self.board = np.array(const.EMPTY_BOARD)
        else:
            self.board = np.array(board)

        self.player = player

        if empty_cells is None:
            self.empty_cells = [
                (x, y)
                for x in range(8)
                for y in range(8)
                if self.board[x][y] == const.EMPTY_CELL
            ]
        else:
            self.empty_cells = empty_cells

    def apply_move(self, x_pos=None, y_pos=None):
        """Places a piece on the board, or handles passing if no move is possible."""
        if x_pos is None and y_pos is None:  # Passing the turn
            lg.logger_board.debug(f"Player {self.player} passes the turn.")
            return

        self.board[x_pos][y_pos] = self.player
        self._remove_empty_cell(x_pos, y_pos)

        lg.logger_board.debug(
            "Player (%s) places at (%s|%s)", self.player, x_pos, y_pos
        )
        lg.logger_board.debug("Remaining empty fields:\n%s", self.print_empty_cells())
        lg.logger_board.debug("Board:\n%s", self.print_board(to_console=False))

    def stones_to_flip(self, x_pos, y_pos):
        """Calculates which stones must be flipped after a move is played."""
        stones_to_flip = []
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Top-left, Up, Top-right
            (0, -1),
            (0, 1),  # Left, Right
            (1, -1),
            (1, 0),
            (1, 1),  # Bottom-left, Down, Bottom-right
        ]

        for dx, dy in directions:
            nx, ny = x_pos + dx, y_pos + dy
            turn = []

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == -self.player:
                    turn.append((nx, ny))
                elif self.board[nx][ny] == self.player:
                    if turn:
                        stones_to_flip.extend(turn)
                    break
                else:
                    break
                nx += dx
                ny += dy

        return stones_to_flip

    def _flip_stones(self, stones_to_flip):
        """Flips stones on the board for the current player."""
        for flip_x, flip_y in stones_to_flip:
            self.board[flip_x][flip_y] = self.player

    def switch_player(self):
        """Switches the current player to the opponent."""
        self.player = -self.player

    def _remove_empty_cell(self, x_pos, y_pos):
        """Removes a cell from the list of empty cells."""
        self.empty_cells.remove((x_pos, y_pos))

    def update(self, x_pos=None, y_pos=None):
        """Updates the board state after a move, Note, A pass move is also a move."""
        if x_pos is None and y_pos is None:  # Passing the turn
            self.switch_player()
            return []

        stones_to_flip = self.stones_to_flip(x_pos, y_pos)
        self._flip_stones(stones_to_flip)
        self.switch_player()
        return stones_to_flip

    def valid_moves(self):
        """Finds all valid moves for the current player."""
        valid_moves = []
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Top-left, Up, Top-right
            (0, -1),
            (0, 1),  # Left, Right
            (1, -1),
            (1, 0),
            (1, 1),  # Bottom-left, Down, Bottom-right
        ]

        for x, y in self.empty_cells:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                found_opponent = False

                while (
                    0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == -self.player
                ):
                    found_opponent = True
                    nx += dx
                    ny += dy

                if (
                    found_opponent
                    and 0 <= nx < 8
                    and 0 <= ny < 8
                    and self.board[nx][ny] == self.player
                ):
                    valid_moves.append((x, y))
                    break

        return valid_moves

    def must_pass(self):
        """Determines if the current player has no valid moves and must pass."""
        return not self.valid_moves()

    def is_terminal_state(self):
        """Determines if the current board state is terminal (game over)."""
        if self.must_pass():
            self.switch_player()
            if self.must_pass():
                self.switch_player()  # Restore original player
                return True
            self.switch_player()
        return False

    def determine_winner(self):
        """Determines the winner of the game."""
        total_sum = np.sum(self.board)

        if total_sum > 0:
            return const.PlayerColor.WHITE.value
        elif total_sum < 0:
            return const.PlayerColor.BLACK.value
        else:
            return 0

    def print_board(self, to_console=True):
        """Prints or returns a formatted string representation of the board."""
        board_str = "\n    " + "  ".join(str(i) for i in range(8)) + "\n"
        board_str += "   " + "-" * 25 + "\n"

        for row_idx, row in enumerate(self.board):
            row_str = f"{row_idx} | "
            for col in row:
                if col == const.PlayerColor.BLACK.value:
                    row_str += "B  "
                elif col == const.PlayerColor.WHITE.value:
                    row_str += "W  "
                else:
                    row_str += ".  "
            board_str += row_str.strip() + "\n"

        if to_console:
            print(board_str)
        else:
            return board_str

    def print_empty_cells(self):
        """Returns a formatted string of all empty cells for logging."""
        cells_per_line = 6
        cells_str = ""

        for i, ec in enumerate(self.empty_cells, start=1):
            cells_str += f"{ec}"
            if i % cells_per_line == 0:
                cells_str += "\n"
            else:
                cells_str += " - "

        return cells_str.strip(" -\n")


if __name__ == "__main__":
    board = Board()
    board.print_board()
