# Set the size of the game window
WIDTH, HEIGHT = 1300, 900
ROWS, COLS = 8, 8
SQUARE_SIZE = HEIGHT // COLS

# Define colors
BACKGROUND_COLOR = (50-5, 50-5, 50-5)  # Dark green background for a soothing look
GRID_COLOR = (200, 200, 200)  # Slate gray grid lines to add sophistication
COLOR_VALID_FIELDS = (207, 182, 103)  # Light green for valid fields

COLOR_HEATMAP_LOW = (70, 130, 180)  # Steel Blue for low probability
COLOR_HEATMAP_HIGH = (255, 69, 0)  # Red-Orange for high probability

FPS = 60
