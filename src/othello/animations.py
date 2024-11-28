import pygame
import os
from src.othello.game_settings import SQUARE_SIZE

class AnimationManager:
    """
    Manages animations for the Othello game.

 
    """

    def __init__(self):
        
        
        
        self.square_size = SQUARE_SIZE

        # Load rotation images
        path_to_images = os.path.join(os.path.dirname(__file__), "images")
        self.rotation_images = [
            pygame.image.load(os.path.join(path_to_images, f"black_to_white{i}.png")) for i in range(1, 4)
        ]
        for i in range(len(self.rotation_images)):
            self.rotation_images[i] = pygame.transform.smoothscale(self.rotation_images[i], (self.square_size, self.square_size))

    def play_flip_animation(self, flipped_stones, screen, draw_board, speed=50):
        """
        Plays the flip animation for the specified stones.

        Args:
            flipped_stones (list): List of (row, col) positions of the flipped stones.
           
            screen (pygame.Surface): The Pygame surface for rendering.
            draw_board (function): Function to draw the game board.
            speed (int): Delay between frames in milliseconds.
        """
        if not flipped_stones:
            return  # No animation needed

        for frame in self.rotation_images:
            draw_board()  # Redraw the current state of the board
            for row, col in flipped_stones:
                screen.blit(frame, (col * self.square_size, row * self.square_size))
            pygame.display.flip()  # Update the display after all frames are drawn
            pygame.time.delay(speed)
