from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # deactivates pygame's welcome message

import pygame
import numpy as np
import pygame.locals
from src.othello.board import Board  
import src.othello.game_constants as const

import os


# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Set the size of the game window
WIDTH, HEIGHT = 1300, 900
ROWS, COLS = 8, 8
SQUARE_SIZE = HEIGHT // COLS

# Define colors
BG_COLOR = (50, 50, 50)  # Dark gray background
GRID_COLOR = (200, 200, 200)  # Light gray for the grid lines
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


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

# Play flip animation at the positions of flipped stones
def play_flip_animation(flipped_stones, board, delay_time=50): 
    for frame in rotation_images:
        draw_board(board)  # Redraw the base state of the board
        for row, col in flipped_stones:
            screen.blit(frame, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        pygame.display.flip()  # Update the display only after all images are drawn
        pygame.time.delay(delay_time)  # Adjust delay for smooth animation




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

                    play_flip_animation(flipped_stones, board)
        
        draw_board(board)
        pygame.display.flip()
        clock.tick(60)
        

    pygame.quit()


if __name__ == "__main__":
    main()
