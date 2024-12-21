import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Deactivates pygame's welcome message

import pygame

import pygame.locals
from src.othello.board import Board
import src.othello.game_constants as const
from src.othello.game_settings import WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE
from src.othello.game_settings import BACKGROUND_COLOR, GRID_COLOR, FPS
from src.othello.animations import AnimationManager


class GamePvsP:
    def __init__(self, screen, animation_manager, board=None):
        self.screen = screen
        self.animation_manager = animation_manager
        self.board = Board(board)
        self.running = True
        self.is_computer_turn = False
        self.clock = pygame.time.Clock()

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)

    def handle_mouse_click(self, position):
        x, y = position
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        if (row, col) in self.board.valid_moves():
            self.board.apply_move(row, col)
            flipped_stones = self.board.update(row, col)
            self.animation_manager.play_flip_animation(
                flipped_stones,
                self.screen,
                self.draw_board,
                self.board.player,
                self.clock,
            )


    def draw(self):
        self.draw_board()
        pygame.display.flip()

    def draw_board(self):
        self.screen.fill(BACKGROUND_COLOR)
        for row in range(ROWS):
            for col in range(COLS):
                pygame.draw.rect(
                    self.screen,
                    GRID_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,
                )
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
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(fps)


# Initialize Pygame and game settings
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello")


# Initialize animation manager
animation_manager = AnimationManager()

# Create the game instance
game = GamePvsP(screen, animation_manager)

# Run the game
if __name__ == "__main__":
    game.run()

    # Quit pygame when done
pygame.quit()
