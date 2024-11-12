from src.othello.board import Board
import src.othello.game_constants as const
from src.utils import print_green

import timeit


board = Board(const.EMPTY_BOARD)
n = 10**3

exe_time = timeit.timeit(board.valid_moves, number=n)

print_green.print_green(
    f"Function {str({board.valid_moves})} needed {exe_time} seconds for {n} iterations"
)
