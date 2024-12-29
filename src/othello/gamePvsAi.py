import os
import pygame
import pygame.locals
from src.othello.board import Board
import src.othello.game_constants as const
from src.othello.game_settings import WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE
from src.othello.game_settings import BACKGROUND_COLOR, GRID_COLOR, FPS
from src.othello.game_visuals import GameVisuals
from random import choice


class GamePvsAi:
    """
    Extension of the GamePvsP class for a Player-vs-AI Othello game.

    This class adds AI functionality by overriding and extending move logic.
    The player and AI take turns, with the AI automatically executing its moves.
    """

    def __init__(self, screen, board=None):
        """
        Initialize the Player-vs-AI game.

        Args:
            screen: The Pygame screen where the game will be rendered.
            board: Optionally, an existing board instance. A new board is created if not provided.
        """
        self.clock = pygame.time.Clock()
        self.board = Board(board)
        self.screen = screen
        self.game_renderer = GameVisuals(screen, self.clock)

        self.running = True
        self.is_ai_turn = choice([True, False])  # Randomly decide whether AI goes first

    def handle_events(self):
        """
        Handle game events for player or AI based on whose turn it is.
        """
        if self.is_ai_turn:
            self.handle_ai_turn()  # Handle AI's move when it's AI's turn
        else:
            self.handle_player_turn()  # Handle player's move when it's player's turn

    def handle_player_turn(self):
        """
        Handle player’s interaction with the game board during their turn.
        """
        if self.board.must_pass():
            print("Player passes the turn.")
            self.board.switch_player()
            self.activate_ai()
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Exit the game
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.process_player_move(event.pos):
                    self.activate_ai()  # Once the player moves, activate the AI

    def handle_ai_turn(self):
        """
        Execute the AI's move by selecting a random valid move.
        """
        if self.board.must_pass():
            print("AI passes the turn.")
            self.board.update
            self.deactivate_ai()
            return

        valid_moves = self.board.valid_moves()
        if valid_moves:
            row, col = choice(valid_moves)  # Choose a random move
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.game_renderer.play_flip_animation(
                self.board.board, flipped_stones, self.board.player
            )
        self.deactivate_ai()

    def deactivate_ai(self):
        """Deactivate AI's turn and activate the player's turn."""
        self.is_ai_turn = False

    def activate_ai(self):
        """Activate AI's turn."""
        self.is_ai_turn = True

    def draw(self):
        """
        Redraw the game screen and update the display.
        """
        self.game_renderer.draw_board(self.board.board)
        pygame.display.flip()  # Update the display

    def process_player_move(self, position):
        """
        Processes a mouse click on the board.

        Args:
            position (tuple): The (x, y) position of the mouse click.

        Returns:
            True if the move was valid, False otherwise
        """
        x, y = position
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE

        if (row, col) in self.board.valid_moves():
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.game_renderer.play_flip_animation(
                self.board.board, flipped_stones, self.board.player
            )
            return True
        return False

    def run(self, fps=FPS):
        """
        Main game loop with player and AI support.

        Args:
            fps: The game's frame rate (frames per second).
        """
        while self.running:
            if self.board.is_terminal_state():
                winner = self.board.determine_winner()
                print(
                    f"Game Over! Winner: {'Black' if winner == -1 else 'White' if winner == 1 else 'Draw'}"
                )
                self.running = False
                continue

            self.handle_events()
            self.draw()
            self.clock.tick(fps)


def run_game(board=None, activate_ai=False):
    """Function to run a Demo of the game."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    game = GamePvsAi(screen, board=board)
    if activate_ai:
        game.activate_ai()
    else:
        game.deactivate_ai()
    game.run()

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    game = GamePvsAi(screen)
    game.run()

    pygame.quit()
