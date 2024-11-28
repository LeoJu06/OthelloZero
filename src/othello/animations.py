import pygame

class AnimationManager:
    """
    Manages animations for the Othello game.

    Attributes:
        rotation_images (list): List of image frames for the flip animation.
        square_size (int): Size of a square on the board in pixels.
    """

    def __init__(self, rotation_images, square_size):
        """
        Initializes the AnimationManager.

        Args:
            rotation_images (list): List of image frames for the flip animation.
            square_size (int): Size of a square on the board in pixels.
        """
        if not rotation_images or not isinstance(rotation_images, list):
            raise ValueError("rotation_images must be a non-empty list of images")
        self.rotation_images = rotation_images
        self.square_size = square_size

    def play_flip_animation(self, flipped_stones, board, screen, draw_board, speed=50):
        """
        Plays the flip animation for the specified stones.

        Args:
            flipped_stones (list): List of (row, col) positions of the flipped stones.
            board (Board): The current game board.
            screen (pygame.Surface): The Pygame surface for rendering.
            draw_board (function): Function to draw the game board.
            speed (int): Delay between frames in milliseconds.
        """
        if not flipped_stones:
            return  # No animation needed

        for frame in self.rotation_images:
            draw_board(board)  # Redraw the current state of the board
            for row, col in flipped_stones:
                screen.blit(frame, (col * self.square_size, row * self.square_size))
            pygame.display.flip()  # Update the display after all frames are drawn
            pygame.time.delay(speed)
