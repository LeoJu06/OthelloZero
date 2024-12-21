import pygame
import os
from src.othello.game_settings import SQUARE_SIZE

class AnimationManager:
    """
    Manages animations for the Othello game, such as flipping stones with an animated transition.
    
    Attributes:
        square_size (int): The size of each square on the board, used to scale images.
        rotation_images (list): Preloaded and scaled images for the flipping animation.
    """

    def __init__(self):
        """
        Initializes the AnimationManager by loading and scaling the rotation images for the flipping animation.
        """
        self.square_size = SQUARE_SIZE

        # Construct the path to the images directory
        path_to_images = os.path.join(os.path.dirname(__file__), "images")

        # Load and scale rotation images for flipping animation
        self.rotation_images = [
            pygame.image.load(os.path.join(path_to_images, f"black_to_white{i}.png"))
            for i in range(1, 4)
        ]
        for i in range(len(self.rotation_images)):
            self.rotation_images[i] = pygame.transform.smoothscale(
                self.rotation_images[i], (self.square_size, self.square_size)
            )

    def play_flip_animation(self, flipped_stones, screen, draw_board, speed=50):
        """
        Plays the flip animation for the specified stones.

        Args:
            flipped_stones (list of tuples): List of (row, col) positions of the stones to flip.
            screen (pygame.Surface): The Pygame surface where the animation will be rendered.
            draw_board (function): A function to draw the current state of the game board.
            speed (int): Delay (in milliseconds) between animation frames.
        
        Behavior:
            This method iterates through the preloaded rotation images, updating the game board and rendering
            each frame of the animation for the stones being flipped.
        """
        # Iterate over each frame in the animation sequence
        for frame in self.rotation_images:
            # Redraw the current state of the board to reflect any static elements
            draw_board()

            # Draw the current animation frame for each stone being flipped
            for row, col in flipped_stones:
                # Calculate the position and blit the current frame onto the board
                screen.blit(frame, (col * self.square_size, row * self.square_size))

            # Update the display to show the new frame
            pygame.display.flip()

            # Introduce a short delay to control animation speed
            pygame.time.delay(speed)
