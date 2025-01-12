import numpy as np
import src.othello.game_constants as const
import src.utils.logger_config as lg
from src.utils.coordinates_to_index import coordinates_to_index
import random


class OthelloGame:
    """
    The OthelloGame class encapsulates the game logic for Othello/Reversi.
    It manages the board, validates moves, determines game states, and handles gameplay.
    """

    def __init__(self, rows=8, cols=8):
        """
        Initializes the Othello game with an 8x8 board by default.
        """
        self.rows = rows
        self.columns = cols

    def get_init_board(self):
        """
        Creates the initial board setup for Othello.

        Returns:
            np.ndarray: 2D array representing the board.
                        0 = empty, 1 = black stone, -1 = white stone.
        """
        board = np.zeros((self.rows, self.columns), dtype=int)
        board[3][4] = const.PlayerColor.BLACK.value
        board[3][3] = const.PlayerColor.WHITE.value
        board[4][3] = const.PlayerColor.BLACK.value
        board[4][4] = const.PlayerColor.WHITE.value
        return board

    def get_board_size(self):
        """
        Returns the dimensions of the board.

        Returns:
            tuple: (rows, columns)
        """
        return self.rows, self.columns

    def get_action_size(self):
        """
        Calculates the total number of actions (cells) on the board.

        Returns:
            int: Total number of cells (rows * columns).
        """
        return self.rows * self.columns

    def get_next_state(self, state, player, x_pos, y_pos):
        """
        Executes a player's move and returns the updated board state.

        Args:
            state (np.ndarray): Current board state.
            player (int): Current player (1 for black, -1 for white).
            x_pos (int): Row index of the move.
            y_pos (int): Column index of the move.

        Returns:
            tuple: (updated board, next player)
        """
        next_state = np.copy(state)
        next_state[x_pos, y_pos] = player
        stones_to_flip = self._find_stones_to_flip(state, player, x_pos, y_pos)
        self._flip_stones(next_state, player, stones_to_flip)
        return next_state, -player

    def _find_stones_to_flip(self, state, player, x_pos, y_pos):
        """
        Identifies the stones that need to be flipped after a move.

        Args:
            state (np.ndarray): Current board state.
            player (int): Current player.
            x_pos (int): Row index of the move.
            y_pos (int): Column index of the move.

        Returns:
            list: List of (row, column) positions of stones to flip.
        """
        stones_to_flip = []
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Diagonal and vertical directions.
            (0, -1),
            (0, 1),  # Horizontal directions.
            (1, -1),
            (1, 0),
            (1, 1),  # Diagonal and vertical directions.
        ]

        for dx, dy in directions:
            nx, ny = x_pos + dx, y_pos + dy
            stones_in_line = []

            while 0 <= nx < self.rows and 0 <= ny < self.columns:
                if state[nx][ny] == -player:
                    stones_in_line.append((nx, ny))
                elif state[nx][ny] == player:
                    stones_to_flip.extend(stones_in_line)
                    break
                else:
                    break
                nx += dx
                ny += dy

        return stones_to_flip

    def _flip_stones(self, state, player, stones_to_flip):
        """
        Flips stones for the current player.

        Args:
            state (np.ndarray): Current board state.
            player (int): Current player.
            stones_to_flip (list): Positions of stones to flip.
        """
        for x, y in stones_to_flip:
            state[x][y] = player

    def get_empty_cells(self, state):
        """
        Finds all empty cells on the board.

        Args:
            state (np.ndarray): Current board state.

        Returns:
            list: List of (row, column) positions for empty cells.
        """
        return [
            (x, y)
            for x in range(self.rows)
            for y in range(self.columns)
            if state[x, y] == 0
        ]

    def get_valid_moves(self, state, player):
        """
        Determines all valid moves for a player.

        Args:
            state (np.ndarray): Current board state.
            player (int): Current player.

        Returns:
            list: List of valid move positions (row, column).
        """
        valid_moves = []
        for x, y in self.get_empty_cells(state):
            if any(self._find_stones_to_flip(state, player, x, y)):
                valid_moves.append((x, y))
        return valid_moves

    def flatten_move_coordinates(self, state, player):
        """
        Converts a 2D game board into a 1D array and marks the valid moves.

        This function flattens the initial game board and updates the resulting 
        1D array by setting the indices corresponding to valid moves to 1. 
        The positions that do not correspond to valid moves remain 0.

        Args:
            state (object): The current state of the game, which may contain information 
                            such as the current board configuration or game status. 
                            The exact structure of `state` depends on the specific game implementation.
            player (int): The identifier of the player whose valid moves are to be marked. 
                        This could be, for example, 1 for player 1 and -1 for player 2.

        Returns:
            numpy.ndarray: A 1D array representing the flattened game board, where 
                        valid moves are marked with 1 and invalid moves are 0.

       """
        
        # Flatten the initial board
        flattened_board = np.zeros(self.get_action_size())
        
        # Get valid moves
        valid_moves = self.get_valid_moves(state, player)
        
        # Mark valid moves in the flattened board
        for x, y in valid_moves:
            index = coordinates_to_index(x, y)  # Convert 2D coordinates to index
            flattened_board[index] = 1
        
        return flattened_board


    def get_canonical_board(self, state, player):
        return player * state
    
    def get_reward_for_player(self, state, player):

        if any(self.get_valid_moves(state, player)):
            return None
        
        elif self.is_terminal_state(state):
            winner = self.determine_winner(state)

            if winner == player:
                return 1
            elif winner == -player:
                return -1
        
        return 0 


    def is_terminal_state(self, state):
        """
        Checks if the game has ended (no valid moves for both players).

        Args:
            state (np.ndarray): Current board state.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return not any(
            self.get_valid_moves(state, const.PlayerColor.BLACK.value)
        ) and not any(self.get_valid_moves(state, const.PlayerColor.WHITE.value))

    def determine_winner(self, state):
        """
        Determines the winner based on the board state.

        Args:
            state (np.ndarray): Final board state.

        Returns:
            int: 1 for black, -1 for white, 0 for a draw.
        """
        score = np.sum(state)
        if score > 0:
            return const.PlayerColor.BLACK.value
        elif score < 0:
            return const.PlayerColor.WHITE.value
        return 0

    def print_board(self, state):
        """
        Displays the current board state.

        Args:
            state (np.ndarray): Current board state.
        """
        print("\n    " + "  ".join(map(str, range(self.columns))))
        print("   " + "-" * (3 * self.columns))
        for row_idx, row in enumerate(state):
            row_str = f"{row_idx} | " + "  ".join(
                "B"
                if cell == const.PlayerColor.BLACK.value
                else "W"
                if cell == const.PlayerColor.WHITE.value
                else "."
                for cell in row
            )
            print(row_str)

    def play_random_move(self, state, player):
        """
        Plays a random valid move for the current player.

        Args:
            state (np.ndarray): The current board state.
            player (int): The current player.

        Returns:
            tuple: (updated board, next player), or None if no valid moves are available.
        """
        valid_moves = self.get_valid_moves(state, player)
        if not valid_moves:
            print(f"Player {player} has no valid moves.")
            return state, -player  # Skip turn if no valid moves available.

        # Choose a random move from the valid moves
        move = random.choice(valid_moves)
        print(f"Player {player} plays random move: {move}")
        return self.get_next_state(state, player, *move)


def play_game_with_random_moves():
    """
    Simulates a game of Othello with both players making random moves.
    """
    game = OthelloGame()
    board = game.get_init_board()
    current_player = const.PlayerColor.BLACK.value

    while not game.is_terminal_state(board):
        game.print_board(board)
        board, current_player = game.play_random_move(board, current_player)

    game.print_board(board)
    winner = game.determine_winner(board)
    print("The game is a draw!" if winner == 0 else f"Player {winner} wins!")


def play_game():
    """
    Main game loop for playing Othello via console input.
    Handles input, turn-taking, and displays the board state.
    """
    game = OthelloGame()
    board = game.get_init_board()
    current_player = const.PlayerColor.BLACK.value

    while not game.is_terminal_state(board):
        game.print_board(board)
        valid_moves = game.get_valid_moves(board, current_player)

        if not valid_moves:
            print(f"Player {current_player} has no valid moves. Skipping turn.")
            current_player = -current_player
            continue

        print(f"Player {current_player}'s turn. Valid moves: {valid_moves}")

        while True:
            try:
                move = input("Enter your move as 'row col': ").strip()
                x, y = map(int, move.split())
                if (x, y) in valid_moves:
                    board, current_player = game.get_next_state(
                        board, current_player, x, y
                    )
                    break
                print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Enter row and column as numbers.")

    game.print_board(board)
    winner = game.determine_winner(board)
    print("The game is a draw!" if winner == 0 else f"Player {winner} wins!")


if __name__ == "__main__":
    play_game_with_random_moves()
