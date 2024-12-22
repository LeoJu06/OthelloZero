import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Deactivate Pygame's welcome message

import pygame
import pygame.locals
from src.othello.board import Board
import src.othello.game_constants as const
from src.othello.game_settings import WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE
from src.othello.game_settings import BACKGROUND_COLOR, GRID_COLOR, FPS
from src.othello.game_visuals import GameVisuals
from random import choice  # For random moves


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
            animation_manager: Manager for animations, such as flipping stones.
            board: Optionally, an existing board instance. A new board is created if not provided.
        """
        self.clock = pygame.time.Clock()
        self.board = Board(board)
        self.screen = screen
        self.game_renderer = GameVisuals(screen, self.clock)

        self.running = True

        self.is_ai_turn = choice(
            [True, False]
        )  # The AI does not start; the player goes first.

    def handle_events(self):
        """
        Handle game events.

        - Processes player interactions when it is their turn.
        - Automatically executes the AI's move when it is its turn.
        """
        if not self.is_ai_turn:
            # Handle player interactions
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False  # Exit the game
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle player's move and then activate AI's turn
                    self.handle_mouse_click(event.pos)
                    self.activate_ai()
        else:
            # Handle AI's move
            self.handle_ai_turn()

    def handle_ai_turn(self):
        """
        Execute the AI's move.

        The AI selects a valid move and applies it. Control is then returned to the player.
        """
        valid_moves = self.board.valid_moves()
        if valid_moves:
            # Example: Choose a random valid move
            row, col = choice(valid_moves)
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.game_renderer.play_flip_animation(
                self.board.board, flipped_stones, self.board.player
            )
        self.deactivate_ai()  # After the AI's move, activate the player's turn

    def deactivate_ai(self):
        """Deactivate AI's turn and activate the player's turn."""
        self.is_ai_turn = False

    def activate_ai(self):
        """Activate AI's turn."""
        self.is_ai_turn = True

    def draw(self):
        """
        Redraws the game screen and updates the display.
        """
        self.game_renderer.draw_board(self.board.board)
        pygame.display.flip()  # Update the display

    def handle_mouse_click(self, position):
        """
        Processes a mouse click on the board.

        Args:
            position (tuple): The (x, y) position of the mouse click.
        """
        x, y = position
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE

        # Check if the clicked position is a valid move
        if (row, col) in self.board.valid_moves():
            self.board.apply_move(row, col)  # Update the board state
            flipped_stones = self.board.update(row, col)  # Flip stones

            # Trigger animation for flipped stones
            self.game_renderer.play_flip_animation(
                self.board.board, flipped_stones, self.board.player
            )

    def run(self, fps=FPS):
        """
        Main game loop with player and AI support.

        Args:
            fps: The game's frame rate (frames per second).
        """
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(fps)


# Run the game
if __name__ == "__main__":
    # Initialize Pygame and game settings
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    # Initialize the animation manager

    # Create the game instance
    game = GamePvsAi(screen)
    game.run()

    # Quit Pygame when the game ends
    pygame.quit()
