from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # deactivates pygame's welcome message

import pygame
import numpy as np
import pygame.locals
from src.othello.board import Board  
import src.othello.game_constants as const
from src.othello.game_settings import WIDTH, HEIGHT,ROWS, COLS, SQUARE_SIZE
from src.othello.game_settings import BG_COLOR, GRID_COLOR, WHITE, BLACK
from src.othello.animations import AnimationManager

import os


# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()




# Load images
path_to_images = os.path.join(os.path.dirname(__file__), "images")
img_black_stone = pygame.image.load(os.path.join(path_to_images, "black_stone.png"))
img_white_stone = pygame.image.load(os.path.join(path_to_images, "white_stone.png"))

# Scale the images to 100x100 pixels
img_white_stone = pygame.transform.smoothscale(img_white_stone, (SQUARE_SIZE, SQUARE_SIZE))
img_black_stone = pygame.transform.smoothscale(img_black_stone, (SQUARE_SIZE, SQUARE_SIZE))

# Load all rotation images
rotation_images = [
    pygame.image.load(os.path.join(path_to_images, f"black_to_white{i}.png")) for i in range(1, 4)     
]

# Optionally scale the rotation images if necessary
for i in range(len(rotation_images)):
    rotation_images[i] = pygame.transform.smoothscale(rotation_images[i], (SQUARE_SIZE,SQUARE_SIZE))

animation_manager = AnimationManager(rotation_images, square_size=SQUARE_SIZE)

# Initialize the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello")

# Draw the game board
def draw_board(board: Board):
    screen.fill(BG_COLOR)
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, GRID_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
            piece = board.board[row][col]
            if piece == const.PlayerColor.BLACK.value:
                screen.blit(img_black_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE))

            elif piece == const.PlayerColor.WHITE.value:
                screen.blit(img_white_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE))






# Main game loop
def main():
    running = True
    board = Board(const.EMPTY_BOARD)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                if (row, col) in board.valid_moves():
                    print(row, col)
                    board.apply_move(row, col)
                    flipped_stones = board.update(row, col)
                    print(flipped_stones)

                    animation_manager.play_flip_animation(flipped_stones, board, screen, draw_board)
        
        draw_board(board)
        pygame.display.flip()
        clock.tick(60)
        

    pygame.quit()


if __name__ == "__main__":
    main()
