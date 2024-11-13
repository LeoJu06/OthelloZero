"""
This file contains constants for the Othello game,
such as player color values and the empty field representation,
which remain consistent throughout the game.
"""

from enum import Enum


class PlayerColor(Enum):
    """
    Enum for player colors.
    This ensures that the player color values are fixed and cannot be changed.
    """

    BLACK = -1
    WHITE = 1


# Constant for empty fields (no player assigned to this position)
EMPTY_CELL = 0
EMPTY_BOARD = [[EMPTY_CELL for x in range(8)] for y in range(8)]

# Placing the starting peaces
EMPTY_BOARD[3][4] = PlayerColor.BLACK.value
EMPTY_BOARD[3][3] = PlayerColor.WHITE.value
EMPTY_BOARD[4][3] = PlayerColor.BLACK.value
EMPTY_BOARD[4][4] = PlayerColor.WHITE.value

    


if __name__ == "__main__":
    # prints -1 to the console (representing the black player)
    print(PlayerColor.BLACK.value)

    # prints 1 to the console (representing the white player)
    print(PlayerColor.WHITE.value)

    # prints 0 to the console (representing an empty field)
    print(EMPTY_CELL)
