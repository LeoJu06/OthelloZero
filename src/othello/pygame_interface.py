import pygame
import numpy as np
import pygame.locals
from src.othello.board import Board  # Importiere deine existierende Board-Klasse
import src.othello.game_constants as const

import os




# Initialisiere Pygame
pygame.init()

# Setze die Größe des Spielfensters
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Farben definieren
BG_COLOR = (50, 50, 50)  # Dunkelgrauer Hintergrund
GRID_COLOR = (200, 200, 200)  # Helles Grau für die Linien
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# images
path_to_images = os.path.join(os.path.dirname(__file__), "images")
img_black_stone = pygame.image.load(os.path.join(path_to_images, "black_stone.png"))
img_white_stone = pygame.image.load(os.path.join(path_to_images, "white_stone.png"))

# Skaliere das Bild auf 100x100 Pixel
img_white_stone = pygame.transform.smoothscale(img_white_stone, (SQUARE_SIZE, SQUARE_SIZE))
img_black_stone = pygame.transform.smoothscale(img_black_stone, (SQUARE_SIZE, SQUARE_SIZE))

# Lade alle Rotationsbilder
rotation_images = [
    pygame.image.load(os.path.join(path_to_images, f"black_to_white{i}.png")) for i in range(1, 4)     
]

# Optionale Skalierung der Bilder, falls notwendig
for i in range(len(rotation_images)):
    rotation_images[i] = pygame.transform.scale(rotation_images[i], (100, 100))



# Initialisiere die Spielanzeige
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello")

# Zeichne das Spielbrett
def draw_board(board: Board):
    screen.fill(BG_COLOR)
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, GRID_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
            piece = board.board[row][col]
            if piece == const.PlayerColor.BLACK.value:
                screen.blit(img_black_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE))

            elif piece == const.PlayerColor.WHITE.value:
                screen.blit(img_white_stone,(col * SQUARE_SIZE, row * SQUARE_SIZE) )


# Animationsfunktion für mehrere Steine
def animate_multiple_stones(flipped_stones,board, step=0):
    for (x, y) in flipped_stones:
        # Wenn der Fortschritt der Animation für diesen Stein noch nicht abgeschlossen ist
        if step < len(rotation_images):
            print("Drehung der Steine")
            screen.blit(rotation_images[step], (x * SQUARE_SIZE, y * SQUARE_SIZE))
            
        else:
            # Wenn der Fortschritt abgeschlossen ist, zeichne den finalen Stein
            piece = board.board[y][x]  # Verwende den aktuellen Zustand des Steins (schwarz oder weiß)
            if piece == const.PlayerColor.BLACK.value:
                screen.blit(img_black_stone, (x * SQUARE_SIZE, y * SQUARE_SIZE))
            elif piece == const.PlayerColor.WHITE.value:
                screen.blit(img_white_stone, (x * SQUARE_SIZE, y * SQUARE_SIZE))


# Hauptspielschleife
def main():
    running = True
    board = Board(const.EMPTY_BOARD)
    flipped_stones = []  # Liste der umgedrehten Steine
    animation_step = 0  # Fortschritt der Animation für alle Steine
    max_steps = len(rotation_images)  # Anzahl der Animationen für den Übergang

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                if (row, col) in board.valid_moves():
                    board.apply_move(row, col)
                    flipped_stones = board.update(row, col)

        # Wenn es Steine zum animieren gibt, führe die Animation aus
        if flipped_stones:
            animate_multiple_stones(flipped_stones,board, animation_step)  # Animieren der Steine

        # Zeichne das Spielfeld, wenn die Animation fertig ist
        
        pygame.display.flip()
        

        # Wenn alle Animationsschritte durchlaufen sind, geh zur nächsten Runde
        if animation_step < max_steps:
            animation_step += 1
        else:
            # Wenn die Animation abgeschlossen ist, setze sie zurück
            flipped_stones = []  # Keine weiteren Animationen mehr
            animation_step = 0
            draw_board(board)

        pygame.time.delay(100)  # Verzögere die Animation, damit der Fortschritt sichtbar ist

    pygame.quit()



if __name__ == "__main__":
    main()
