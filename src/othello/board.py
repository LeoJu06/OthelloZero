"""This File contains the board class, 
which represents an othello board with all it's functions
"""
import src.othello.game_constants as const
import src.logger_config as lg


class Board:
    """The Board Class represents an Othello board"""

    def __init__(self, board, player=const.PlayerColor.BLACK.value, empty_cells=None):
        self.board = board
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

    def apply_move(self, x_pos: int, y_pos: int):
        """Function to apply the given move into self.board.
        Assumes that only valid moves are passed to this method.
        """

       

        # placing the piece
        self.board[x_pos][y_pos] = self.player

        # removing it from the empty fields
        self.empty_cells.remove((x_pos, y_pos))

        # Flipping the stones
        self._flip_stones()
        
        # simple debug logging
        lg.logger_board.debug(
            "Player (%s) places at (%s|%s)", self.player, x_pos, y_pos
        )
        lg.logger_board.debug("Remaining empty fields: {%s}", self.print_empty_cells())
        lg.logger_board.debug("Board: %s", self.print_board(to_console=False))

        # switch player
        self.switch_player()

    def _flip_stones(self):
        pass

    def switch_player(self):
        """Function to switch to the other player (e.g from black to white)"""
        self.player = -self.player

    def valid_moves(self):
        """Returns a list with all valid moves, which can be made"""
        # list with valid moves
        valid_moves = []

        # all directions you could possibly place a peace
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # iterate over empty fields
        for x, y in self.empty_cells:
            # move over the fields around you
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                found_opponent = False

                # move further if there is an enemy and the field is still on the board
                while (
                    0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == -self.player
                ):
                    found_opponent = True
                    nx += dx
                    ny += dy

                # look if the field is still on the board and a stone of mine is there
                if (
                    found_opponent
                    and 0 <= nx < 8
                    and 0 <= ny < 8
                    and self.board[nx][ny] == self.player
                ):
                    # a valid move was found, append it to the list
                    valid_moves.append((x, y))
                    # you don't have to look further
                    break
        # return valid moves
        return valid_moves

    def is_terminal_state(self):
        """Function to check wheter the board is a draw or someone has won."""

        # Check whether the current player does not have a valid move
        if not self.valid_moves():
            # In this case, the current player would have to pass

            # Change to the opponents view
            self.switch_player()

            # Check whether the opponent no longer has a valid move
            if not self.valid_moves():
                # it is a draw, wether no one has a valid move or the board is full.
                return True

            # change back to the original player
            self.switch_player()

        # There is no draw yet
        return False

    def print_board(self, to_console=True):
        """Function to return the board as a formatted string with row and column numbers."""
        board_str = "\n    0  1  2  3  4  5  6  7 \n"  # Column numbers header
        board_str += (
            "   -----------------------\n"  # Separator between header and board
        )

        # Print the line number (row number) and the corresponding playing field values
        for row_idx, row in enumerate(self.board):
            row_str = f"{row_idx} |"  # Row number, e.g. 0 | ...

            for col in row:
                if col == const.PlayerColor.BLACK.value:
                    row_str += " B"  # Black piece
                elif col == const.PlayerColor.WHITE.value:
                    row_str += " W"  # White piece
                else:
                    row_str += " . "  # Empty space (represented by a dot)

            board_str += row_str + "\n"
        
        if to_console:
            print(board_str)
        else:
            return board_str

    def print_empty_cells(self):
        """This method prints all empty cells for a proper logging format"""
        cells_str = ""
        for i, ec in enumerate(self.empty_cells):
            cells_str += str(ec)

            if i % 6 == 0:
                cells_str += "\n"
            else:
                cells_str += " - "

        return cells_str
