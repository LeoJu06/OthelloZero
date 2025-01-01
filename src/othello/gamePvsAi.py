import os
import pygame
from src.othello.board import Board
import src.othello.game_constants as const
from src.othello.game_settings import (
    WIDTH,
    HEIGHT,
    SQUARE_SIZE,
    BACKGROUND_COLOR,
    GRID_COLOR,
    FPS,
)
from src.othello.game_visuals import GameVisuals
from random import choice
from src.mcts.mcts import MCTS
from src.mcts.node import Node
from src.utils.index_to_coordinates import index_to_coordinates


class GamePvsAi:
    """
    Player-vs-AI Othello Game

    This class manages the game loop, player interactions, and AI moves.
    """

    def __init__(self, screen, board=None):
        """
        Initialize the game instance.

        Args:
            screen: Pygame screen object for rendering.
            board: Optional Board object; creates a new one if not provided.
        """
        self.clock = pygame.time.Clock()
        self.board = Board(board)
        self.screen = screen
        self.visuals = GameVisuals(screen, self.clock)
        self.ai = MCTS(Node(board=board))

        self.running = True
        self.is_ai_turn = choice([True, False])  # Randomly decide if AI starts

    def run_game_loop(self, fps=FPS):
        """
        Main game loop that handles turns and rendering.

        Args:
            fps: Frames per second for the game loop.
        """
        while self.running:
            if self.check_both_players_cannot_move():
                self.running = False
                continue

            self.process_turn()
            self.update_display()
            self.clock.tick(fps)

        self.display_winner()

    def process_turn(self):
        """Process the current turn based on whose turn it is."""
        if self.is_ai_turn:
            self.execute_ai_turn()
        else:
            self.handle_player_input()

    def check_both_players_cannot_move(self):
        """
        Check if neither player nor AI has valid moves.

        Returns:
            True if both players cannot move, False otherwise.
        """
        no_moves_player = self.board.must_pass()
        self.board.switch_player()  # Switch to the other player
        no_moves_ai = self.board.must_pass()
        self.board.switch_player()  # Switch back to the original player

        # If both players must pass, the game should end
        if no_moves_player and no_moves_ai:
            self.display_winner()  # Display the winner (or draw)
            self.running = False  # End the game loop
            return True

        return False

    def handle_player_input(self):
        """Handle player actions during their turn."""
        if self.board.must_pass():
            print("Player has no valid moves. Passing turn.")
            self.switch_to_ai()
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.attempt_player_move(event.pos):
                    self.switch_to_ai()

    def attempt_player_move(self, position):
        """
        Attempt to execute the player's move.

        Args:
            position: Tuple (x, y) of mouse click coordinates.

        Returns:
            True if the move was valid, False otherwise.
        """
        x, y = position
        row, col = y // SQUARE_SIZE, x // SQUARE_SIZE

        if (row, col) in self.board.valid_moves():
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.visuals.play_flip_animation(
                self.board.board, flipped_stones, self.board.player
            )
            return True

        print("Invalid move attempted.")
        return False

    def execute_ai_turn(self):
        """Execute the AI's move."""
        if self.board.must_pass():
            print("AI has no valid moves. Passing turn.")
            self.switch_to_player()
            return
        self.ai = MCTS(Node(board=Board(board=self.board.board.copy())))
        self.ai.search()
        action, child = self.ai.get_best_move()
        self.ai.root_node = child

        row, col = index_to_coordinates(action)
        self.board.apply_move(row, col)
        flipped_stones = self.board.update(row, col)
        self.visuals.play_flip_animation(
            self.board.board, flipped_stones, self.board.player
        )

        self.switch_to_player()

    def switch_to_player(self):
        """Switch the turn to the player."""
        self.is_ai_turn = False

    def switch_to_ai(self):
        """Switch the turn to the AI."""
        self.is_ai_turn = True

    def update_display(self):
        """Redraw the game board and update the screen."""
        self.visuals.draw_board(self.board.board)
        self.visuals.mark_valid_fields(self.board.valid_moves())
        pygame.display.flip()

    def display_winner(self):
        """Determine and display the game winner."""
        winner = self.board.determine_winner()
        print(
            f"Game Over! Winner: {'Black' if winner == -1 else 'White' if winner == 1 else 'Draw'}"
        )


def main(board=None):
    """Entry point to start the game."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    game = GamePvsAi(screen, board)
    game.run_game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
