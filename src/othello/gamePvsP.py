import os

# Suppresses the default pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import pygame.locals
from src.othello.board import Board
import src.othello.game_constants as const
from src.othello.game_settings import WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE
from src.othello.game_settings import BACKGROUND_COLOR, GRID_COLOR, FPS
from src.othello.animations import AnimationManager

class GamePvsP:
    """
    Class representing a Player vs Player Othello game.

    Attributes:
        screen (pygame.Surface): The Pygame surface to draw the game.
        animation_manager (AnimationManager): Manages animations for the game.
        board (Board): The game board managing logic and state.
        running (bool): Controls the main game loop.
        clock (pygame.time.Clock): Manages frame rate for the game loop.
        image_white_stone (pygame.Surface): Scaled image of a white stone.
        image_black_stone (pygame.Surface): Scaled image of a black stone.
    """

    def __init__(self, screen, animation_manager, board=None):
        """
        Initializes the game.

        Args:
            screen (pygame.Surface): The Pygame screen surface.
            animation_manager (AnimationManager): Animation manager instance.
            board (Board, optional): Initial board state. Defaults to None.
        """
        self.screen = screen
        self.animation_manager = animation_manager
        self.board = Board(board)
        self.running = True
        self.clock = pygame.time.Clock()

        # Load and scale images for the stones
        path_to_images = os.path.join(os.path.dirname(__file__), "images")
        image_black_stone = pygame.image.load(
            os.path.join(path_to_images, "black_stone.png")
        )
        image_white_stone = pygame.image.load(
            os.path.join(path_to_images, "white_stone.png")
        )
        self.image_white_stone = pygame.transform.smoothscale(
            image_white_stone, (SQUARE_SIZE, SQUARE_SIZE)
        )
        self.image_black_stone = pygame.transform.smoothscale(
            image_black_stone, (SQUARE_SIZE, SQUARE_SIZE)
        )

    def handle_events(self):
        """
        Handles user input events such as quitting or mouse clicks.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Stops the game loop
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)  # Handle board clicks

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
            self.animation_manager.play_flip_animation(
                flipped_stones,
                self.screen,
                self.draw_board,
                self.board.player,
                self.clock,
            )

    def draw(self):
        """
        Redraws the game screen and updates the display.
        """
        self.draw_board()  # Draw the game board and pieces
        pygame.display.flip()  # Update the display

    def draw_board(self):
        """
        Draws the game board and stones.
        """
        # Fill the screen with the background color
        self.screen.fill(BACKGROUND_COLOR)

        # Draw the grid and pieces on the board
        for row in range(ROWS):
            for col in range(COLS):
                # Draw grid squares
                pygame.draw.rect(
                    self.screen,
                    GRID_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,
                )

                # Draw stones based on board state
                piece = self.board.board[row][col]
                if piece == const.PlayerColor.BLACK.value:
                    self.screen.blit(
                        self.image_black_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )
                elif piece == const.PlayerColor.WHITE.value:
                    self.screen.blit(
                        self.image_white_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )

    def run(self, fps=FPS):
        """
        Runs the main game loop.

        Args:
            fps (int): Frames per second for the game loop.
        """
        while self.running:
            self.handle_events()  # Handle user input
            self.draw()  # Draw the updated game state
            self.clock.tick(fps)  # Cap the frame rate

# Initialize Pygame and game settings
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create game window
pygame.display.set_caption("Othello")  # Set window title

# Initialize animation manager
animation_manager = AnimationManager()

# Create the game instance
game = GamePvsP(screen, animation_manager)

# Run the game if this is the main module
if __name__ == "__main__":
    game.run()

    # Quit pygame when the game loop ends
    pygame.quit()
