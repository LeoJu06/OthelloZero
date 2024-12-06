"""
This file contains the Board class, which represents an Othello board with all its functions.
"""

import sys
import numpy as np
import src.othello.game_constants as const
import src.logger_config as lg


class Board:
    """The Board class represents the Othello game board and manages all game-related operations, such as:
    - Applying moves
    - Checking for valid moves
    - Flipping pieces
    - Switching players
    - Determining game status (e.g., checking if the game is over and determining the winner).

    This class interacts with the game state by keeping track of the current board configuration,
    player turns, and the available empty cells.

    Attributes:
        board (np.array): The current state of the board.
        player (int): The current player (BLACK or WHITE).
        empty_cells (list): List of tuples representing empty cells on the board.
    """

    def __init__(
        self, board=None, player=const.PlayerColor.BLACK.value, empty_cells=None
    ):
        """Initializes the Board instance.

        Args:
            board (np.array): Initial board state.
            player (int, optional): The starting player. Defaults to BLACK.
            empty_cells (list, optional): List of initial empty cells. Defaults to calculation based on the board state.
        """
        if board is None:
            self.board = np.array(const.EMPTY_BOARD)
        else:
            self.board = np.array(board) 
            
        self.player = player

        # Calculate empty cells if not provided
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
        """Places a piece on the board and removes the corresponding empty cell.

        This method assumes that only valid moves are passed. After applying the move,
        the `update()` method must be called to flip stones and change the active player.

        Args:
            x_pos (int): The x-coordinate of the move.
            y_pos (int): The y-coordinate of the move.
        """
        self.board[x_pos][y_pos] = self.player
        self._remove_empty_cell(x_pos, y_pos)

        # Debug logging
        lg.logger_board.debug(
            "Player (%s) places at (%s|%s)", self.player, x_pos, y_pos
        )
        lg.logger_board.debug("Remaining empty fields:\n%s", self.print_empty_cells())
        lg.logger_board.debug("Board:\n%s", self.print_board(to_console=False))

    def stones_to_flip(self, x_pos, y_pos):
        """Calculates which stones must be flipped after a move is played.

        Args:
            x_pos (int): The x-coordinate of the move.
            y_pos (int): The y-coordinate of the move.

        Returns:
            list: A list of tuples representing coordinates of stones to be flipped.
        """
        stones_to_flip = []
        directions = [
            (-1, -1),  # Top-left
            (-1, 0),  # Up
            (-1, 1),  # Top-right
            (0, -1),  # Left
            (0, 1),  # Right
            (1, -1),  # Bottom-left
            (1, 0),  # Down
            (1, 1),  # Bottom-right
        ]
        for dx, dy in directions:
            nx, ny = x_pos + dx, y_pos + dy
            turn = (
                []
            )  # Temporary list to hold coordinates of opponent's pieces to be flipped

            while 0 <= nx < 8 and 0 <= ny < 8:  # Continue until the edge of the board
                if self.board[nx][ny] == -self.player:  # Opponent's piece found
                    turn.append((nx, ny))
                elif self.board[nx][ny] == self.player:  # Current player's piece found
                    if turn:  # If we found opponent pieces to flip
                        stones_to_flip.extend(
                            turn
                        )  # Add the opponent pieces to the flip list
                    break  # No need to check further in this direction
                else:  # Empty cell, stop checking this direction
                    break
                nx += dx  # Move to the next cell in the current direction
                ny += dy

        return stones_to_flip

    def _flip_stones(self, stones_to_flip):
        """Flips stones on the board for the current player.

        Args:
            stones_to_flip (list): List of (x, y) coordinates of stones to be flipped.
        """
        for flip_x, flip_y in stones_to_flip:
            self.board[flip_x][flip_y] = self.player

    def switch_player(self):
        """Switches the current player to the opponent."""
        self.player = -self.player

    def _remove_empty_cell(self, x_pos, y_pos):
        """Removes a cell from the list of empty cells.

        Args:
            x_pos (int): The x-coordinate of the move.
            y_pos (int): The y-coordinate of the move.
        """
        self.empty_cells.remove((x_pos, y_pos))

    def update(self, x_pos, y_pos):
        """Updates the board by flipping stones and switching the player.

        This method must be called after applying a move to ensure the board state is updated correctly.

        Args:
            x_pos (int): The x-coordinate of the move.
            y_pos (int): The y-coordinate of the move.

        Returns:
            list: A list of coordinates representing flipped stones for potential animations.
        """
        stones_to_flip = self.stones_to_flip(x_pos, y_pos)
        self._flip_stones(stones_to_flip)
        self.switch_player()
        return stones_to_flip

    def valid_moves(self):
        """Finds all valid moves for the current player.

        A valid move occurs if, by placing the current player's piece in an empty cell, one or more of the opponent's pieces
        get "flipped" in any direction. A move is only valid if there is at least one opponent's piece sandwiched between
        the current player's piece and another of the current player's pieces.

        Returns:
            list: A list of (x, y) tuples representing valid move coordinates for the current player.
        """
        valid_moves = []  # List to store all valid move coordinates
        # Define all possible directions to check for valid moves (8 directions)
        directions = [
            (-1, -1),  # Top-left
            (-1, 0),  # Up
            (-1, 1),  # Top-right
            (0, -1),  # Left
            (0, 1),  # Right
            (1, -1),  # Bottom-left
            (1, 0),  # Down
            (1, 1),  # Bottom-right
        ]

        # Loop through all empty cells on the board (the possible places for a move)
        for x, y in self.empty_cells:
            # Check each direction from the current empty cell
            for dx, dy in directions:
                nx, ny = x + dx, y + dy  # Start checking in the given direction
                found_opponent = (
                    False  # Flag to track if we encounter any opponent pieces
                )

                # Check cells in the current direction while within the board boundaries
                while (
                    0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == -self.player
                ):
                    found_opponent = (
                        True  # We've found an opponent piece in this direction
                    )
                    nx += dx  # Move to the next cell in this direction
                    ny += dy

                # After we've moved along the direction, check if we found a valid sandwich of opponent pieces
                if (
                    found_opponent  # We found at least one opponent's piece
                    and 0 <= nx < 8  # Ensure the new position is within board bounds
                    and 0 <= ny < 8
                    and self.board[nx][ny]
                    == self.player  # The current player's piece should be at the new position
                ):
                    valid_moves.append(
                        (x, y)
                    )  # Add the current empty cell (x, y) as a valid move
                    break  # No need to check further directions for this empty cell, move to the next one

        return valid_moves  # Return the list of all valid moves for the current player

    def is_terminal_state(self):
        """Determines if the current board state is terminal (game over).

        Returns:
            bool: True if the game is over (no valid moves for both players), False otherwise.
        """
        if not self.valid_moves():
            self.switch_player()
            if not self.valid_moves():
                # Switch back to original player, if both players have no valid move
                self.switch_player()
                return True
            self.switch_player()
        return False

    def determine_winner(self):
        """Determines the winner of the game.

        Returns:
            int: -1 if BLACK wins, 1 if WHITE wins, 0 if it's a draw.
        """
        total_sum = np.sum(self.board)

        if total_sum > 0:
            return const.PlayerColor.WHITE.value
        elif total_sum < 0:
            return const.PlayerColor.BLACK.value
        else:
            return 0

    def print_board(self, to_console=True):
        """Prints or returns a formatted string representation of the board.

        Args:
            to_console (bool, optional): If True, prints the board to the console. If False, returns it as a string.

        Returns:
            str: Formatted string of the board if `to_console` is False.
        """
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
        """Returns a formatted string of all empty cells for logging.

        Returns:
            str: Formatted string showing empty cells.
        """
        cells_per_line = 6
        cells_str = ""

        for i, ec in enumerate(self.empty_cells, start=1):
            cells_str += f"{ec}"
            if i % cells_per_line == 0:
                cells_str += "\n"
            else:
                cells_str += " - "

        return cells_str.strip(" -\n")


def play():
    """Runs a console-based Othello game."""
    board = Board(const.EMPTY_BOARD)

    while not board.is_terminal_state():
        board.print_board()
        moves = board.valid_moves()

        if not moves:
            print("No valid moves available. Switching player...")
            board.switch_player()
            continue

        print("Your valid moves are:")
        print(moves)

        while True:
            try:
                x = int(input("Enter the x-coordinate for your move: "))
                y = int(input("Enter the y-coordinate for your move: "))

                if (x, y) in moves:
                    break
                else:
                    print(f"({x}, {y}) is not a valid move. Please try again.")
            except ValueError:
                print("Invalid input. Coordinates must be integers. Please try again.")
            except KeyboardInterrupt:
                sys.exit()

        board.apply_move(x, y)
        flipped_stones = board.update(x, y)
        print(f"{len(flipped_stones)} stones were flipped.")

    winner = board.determine_winner()
    if winner == const.PlayerColor.BLACK.value:
        print("Congrats! Player Black (B) wins!")
    elif winner == const.PlayerColor.WHITE.value:
        print("Congrats! Player White (W) wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play()
