import os
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
        valid_moves = self.board.valid_moves()
        if valid_moves:
            # Example: Choose a random valid move
            row, col = choice(valid_moves)
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.game_renderer.play_flip_animation(self.board.board, flipped_stones, self.board.player)
        self.deactivate_ai()  # After the AI's move, it's the player's turn

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

        # Check if the clicked position is a valid move
        if (row, col) in self.board.valid_moves():
            self.board.apply_move(row, col)  # Update the board state
            flipped_stones = self.board.update(row, col)  # Flip stones
            # Trigger animation for flipped stones
            self.game_renderer.play_flip_animation(self.board.board, flipped_stones, self.board.player)

            return True
        return False

    def run(self, fps=FPS):
        """
        Main game loop with player and AI support.

        Args:
            fps: The game's frame rate (frames per second).
        """
        while self.running:
            self.handle_events()  # Handle events like player moves and AI turns
            self.draw()  # Redraw the board
            self.clock.tick(fps)  # Maintain consistent frame rate

def run_game(board=None, activate_ai = False):

    """Function to run a Demo of the game. This function should not be 
    used for real purpose"""
    
    
        

    # Initialize Pygame and game settings
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    if board is not None:
        b = board
    

    # Create the game instance
    game = GamePvsAi(screen, board=b)
    
    if activate_ai:
        game.activate_ai()
    else:
        game.deactivate_ai()
    game.run()

    # Quit Pygame when the game ends
    pygame.quit()
# Run the game
if __name__ == "__main__":
    # Initialize Pygame and game settings
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Othello")

    # Create the game instance
    game = GamePvsAi(screen)
    game.run()

    # Quit Pygame when the game ends
    pygame.quit()
