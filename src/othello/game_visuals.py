import pygame
import os
from src.othello.game_constants import PlayerColor
from src.othello.game_settings import FPS

from src.othello.game_settings import BACKGROUND_COLOR, ROWS, COLS
from src.othello.game_settings import GRID_COLOR, SQUARE_SIZE, COLOR_VALID_FIELDS
import src.othello.game_constants as const
from src.utils.format_time import format_time
from src.utils.scientific_notation import scientific_e_format
import matplotlib.pyplot as plt
import numpy as np
import pygame
import io

import matplotlib.pyplot as plt
import numpy as np
import io
import pygame
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import pygame.freetype

import matplotlib.pyplot as plt
import io
import pygame
from matplotlib.ticker import MaxNLocator
from src.utils.from_algebraic_to_choords import from_algebraic_to_coords

size = (4.3, 2.5)
def generate_plot_image(data_points):
    fig, ax = plt.subplots(figsize=size)
    turns = list(range(len(data_points)))

    # Plot
    ax.plot(
        turns,
        data_points,
        marker="o",
        linestyle="-",
        color="#000000",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5
    )

    # Letzter Punkt größer hervorheben
    if data_points:
        ax.plot(
            turns[-1],
            data_points[-1],
            marker="o",
            markersize=10,
            markerfacecolor="black",
            markeredgecolor="black",
            linestyle="None",
            zorder=3,
        )
        ax.text(
            turns[-1],
            data_points[-1] - 0.4,  # Text etwas höher positionieren
            f"{data_points[-1]:.2f}",
            ha='center',
            va='bottom',
            color='black',
            fontsize=10,
            fontweight='bold'
        )

    # Achsentitel und Labels
    ax.set_title("Value-Schätzung des Neuronalen Netzes", fontsize=11, pad=10)
    ax.set_xlabel("Zugnummer", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)

    # Achsenformatierung
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', steps=[5]))
    ax.set_ylim(-1.1, 1.1)
   
    # X-Achsen-Limit anpassen für bessere Ausrichtung
    if len(turns) == 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(-0.5, len(turns) - 0.5)  # Konsistent mit Entropie-Plot
    
    plt.xticks(rotation=0)

    # Farbverlauf im Hintergrund
    ax.axhspan(0.5, 1.1, facecolor='limegreen', alpha=0.4, label="Vorteilhaft")
    
    ax.axhspan(-0.5, 0.5, facecolor='gold', alpha=0.4, label="Ausgeglichen")
    ax.axhspan(-1.1, -0.5, facecolor='lightcoral', alpha=0.4, label="Verlustreich")
    
    ax.legend(loc="upper left", fontsize=7)

    # Stil
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f7f7f7")

    plt.tight_layout(pad=2)  # Etwas mehr Padding für bessere Ausrichtung

    # Speichern und in pygame laden
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=105)
    plt.close(fig)
    buf.seek(0)
    return pygame.image.load(buf)

def generate_entropy_plot_image(entropies):
    fig, ax = plt.subplots(figsize=size)
    turns = list(range(len(entropies)))

    # Plot zeichnen
  

    ax.plot(
        turns,
        entropies,
        marker="o",
        linestyle="-",
        color="#000000",
        markersize=6,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5
    )

    # Achsentitel und Beschriftungen
    ax.set_title("Normierte Entropie", fontsize=11, pad=10)  # Gleiche Schriftgröße wie oben
    ax.set_xlabel("Zugnummer", fontsize=10)
    ax.set_ylabel("Entropie", fontsize=10)

    # X-Ticks formatieren
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', steps=[5]))
    plt.xticks(rotation=0)

    # Letzten Entropie-Wert anzeigen
    if entropies:


        ax.plot(
            turns[-1],
            entropies[-1],
            marker="o",
            markersize=10,
            markerfacecolor="black",
            markeredgecolor="black",
            linestyle="None",
            zorder=3,
        )
        ax.text(
            turns[-1],
            entropies[-1] + 0.05,  # Text etwas höher positionieren
            f"{entropies[-1]:.2f}",
            ha='center',
            va='bottom',
            color='black',
            fontsize=10,
            fontweight="bold"
        )
      
    ax.set_ylim(0, 1.00)
    
    # Hintergrundfarben
    ax.axhspan(0.0, 0.3, facecolor='limegreen', alpha=0.4, label='Fokussiert')
    ax.axhspan(0.3, 0.7, facecolor='gold', alpha=0.4, label='Neutral')
    ax.axhspan(0.7, 1.0, facecolor='lightcoral', alpha=0.4, label='Unsicher')

    ax.legend(loc="upper left", fontsize=8)

    # X-Achsen-Limit konsistent mit Value-Plot
    if len(turns) == 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(-0.5, len(turns) - 0.5)

    # Stil anpassen
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f7f7f7")

    plt.tight_layout(pad=2)  # Gleiches Padding wie oben

    # In BytesIO-Buffer schreiben
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches="tight", dpi=100)
    plt.close(fig)

    buf.seek(0)
    image = pygame.image.load(buf)
    return image

class GameVisuals:
    """
    Handles the rendering and animation for the Othello game.

    Attributes:
        screen (pygame.Surface): The Pygame surface for rendering.
        board (list): Current state of the board.
        clock (pygame.time.Clock): Clock for managing frame rate.
        rotation_images_black_to_white (list): Preloaded frames for black-to-white flipping animation.
        rotation_images_white_to_black (list): Preloaded frames for white-to-black flipping animation.
        image_black_stone (pygame.Surface): Scaled image for black stones.
        image_white_stone (pygame.Surface): Scaled image for white stones.
    """

    def __init__(self, screen, clock):
        """
        Initializes the visual manager and preloads images for animations and rendering.

        Args:
            screen (pygame.Surface): The Pygame surface for rendering.
            board (list): Current board state.
            clock (pygame.time.Clock): Clock for managing frame rate.
        """
        self.square_size = SQUARE_SIZE  # Square size for each tile on the board
        self.screen = screen  # Screen to render the visuals
        self.clock = clock  # Clock to control the frame rate
        self.values = [0]

        # Load the images for black and white stones
        self.image_black_stone, self.image_white_stone = self._load_stone_images()

        # Load the transition images for the flipping animation
        (
            self.rotation_images_black_to_white,
            self.rotation_images_white_to_black,
        ) = self._load_transition_images()

        self.game_data = {"max_depth": None, "num_states": None, "num_legal_moves": None, "min_prior": None, "max_prior": None, "thinking_time":None, "move": None, "tree_nodes":None, "search_efficiency":None, "sps":None, "brute_force_time":None}
        self._last_plot_values = []
        self.entropies = [0]
        self._last_entropy_values = []
        self._plot_image = None
        self._entropy_plot_image = None
        self.moves = []
        self.cmap = "RdYlGn"

    def _load_stone_images(self):
        """
        Loads and scales the images for black and white stones.

        Returns:
            tuple: Scaled images for black and white stones.
        """
        path_to_stone_images = os.path.join(
            os.path.dirname(__file__), "assets"
        )  # Path to images folder

        # Load and scale the black stone image
        image_black_stone = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(path_to_stone_images, "black_stone.png")),
            (SQUARE_SIZE, SQUARE_SIZE),  # Scale to fit the square size
        )

        # Load and scale the white stone image
        image_white_stone = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(path_to_stone_images, "white_stone.png")),
            (SQUARE_SIZE, SQUARE_SIZE),  # Scale to fit the square size
        )

        return image_black_stone, image_white_stone

    def _load_transition_images(self):
        """
        Loads the images for the flipping animation.

        Returns:
            tuple: Two lists of images for black-to-white and white-to-black transitions.
        """
        path_to_images = os.path.join(
            os.path.dirname(__file__), "assets", "transition_images"
        )  # Path to transition images folder

        # Load and scale the black-to-white flipping images
        rotation_images_black_to_white = [
            pygame.image.load(os.path.join(path_to_images, f"black_to_white_{i}.png"))
            for i in range(14)  # Assuming 14 frames of transition
        ]
        rotation_images_black_to_white = [
            pygame.transform.smoothscale(image, (self.square_size, self.square_size))
            for image in rotation_images_black_to_white
        ]

        # Reverse and flip the images for the white-to-black transition
        rotation_images_white_to_black = [
            pygame.transform.flip(frame, False, True)
            for frame in reversed(rotation_images_black_to_white)
        ]

        return rotation_images_black_to_white, rotation_images_white_to_black

    def play_flip_animation(self, board, flipped_stones, player):
        """
        Plays the flip animation for a set of stones.

        Args:
            flipped_stones (list of tuples): Coordinates of stones being flipped.
            player (int): The player initiating the flip (BLACK or WHITE).
        """
        # Choose the correct animation images based on the player color
        animation_images = (
            self.rotation_images_white_to_black
            if player == PlayerColor.WHITE.value
            else self.rotation_images_black_to_white
        )

        # Iterate through each frame of the animation
        for frame in animation_images:
            # Redraw the board in each frame to keep static elements visible
            self.draw_board(board)
            
            self.draw_plot()
            self.display_informations()
            self.draw_colorbar()
            self.draw_move_history(None)
            self.highlight_last_move(self.moves)
            # Overlay the current animation frame on the flipping stones
            for row, col in flipped_stones:
                self.screen.blit(
                    frame, (col * self.square_size, row * self.square_size)
                )

            pygame.display.flip()  # Refresh the display to show the updated frame
            
            self.clock.tick(FPS)  # Maintain consistent frame rate as defined by FPS

    def draw_heatmap_policy(self, policy):

        policy = np.array(policy).reshape((8, 8))
        max_value = policy.max()
        cmap = cm.get_cmap(self.cmap)
        gamma = 0.4

        if max_value == 0:
            return 
        
        for row in range(ROWS):
            for col in range(COLS):
                value = policy[row][col]
                if value > 0:

                    normalized = (value / max_value) 
                    
                    r, g, b, _ = cmap(normalized)
                    color = (int(r*255), int(g*255), int(b*255))
                    
                    
                   
                    rect_position = (col * SQUARE_SIZE + 1, row*SQUARE_SIZE+1 )

                    rect_size = (
                    SQUARE_SIZE - 2,
                    SQUARE_SIZE - 2,
                    )  # subtracting a but for not overdrawing the grid lines

                    # Create the rectangle and draw it
                    pygame.draw.rect(
                        self.screen, color, pygame.Rect(rect_position, rect_size)
                    )


    def draw_colorbar(self, x=1250, y=480, height=220, width=20, label="Policy"):
        """
        Zeichnet eine vertikale Farbskala (Colorbar) als Legende für die Policy-Heatmap.

        Args:
            x, y (int): Position oben links der Skala
            height (int): Höhe der Farbskala
            width (int): Breite der Farbskala
            cmap (function): Eine Funktion, die normierte Werte (0–1) in RGB-Farben übersetzt
            label (str): Beschriftung über der Skala
        """
        font = pygame.font.SysFont("arial", 16)
        title_surface = font.render(label, True, (255, 255, 255))
        self.screen.blit(title_surface, (x - 10, y - 25))
        cmap = cm.get_cmap(self.cmap)

        # Skala von oben (1.0) nach unten (0.0)
        for i in range(height):
            value = 1.0 - i / height  # 1 oben, 0 unten

            
            r, g, b, _ = cmap(value)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)

            color = (r, g, b)
            pygame.draw.rect(self.screen, color, pygame.Rect(x, y + i, width, 1))

        # Beschriftungen unten & oben
        font_small = pygame.font.SysFont("arial", 12)
        self.screen.blit(font_small.render("0.0", True, (255, 255, 255)), (x + width + 5, y + height - 6))
        self.screen.blit(font_small.render("1.0", True, (255, 255, 255)), (x + width + 5, y - 2))

                
    def draw_board(self, board):
        """
        Draws the game board and pieces.

        Args:
            board (list): The current state of the game board.
        """
        self.screen.fill(
            BACKGROUND_COLOR
        )  # Fill the background with the specified color

        # Loop through each row and column of the board to draw the grid and pieces
        for row in range(ROWS):
            for col in range(COLS):
                # Draw the grid square (the border around each tile)
                pygame.draw.rect(
                    self.screen,
                    GRID_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,  # Thickness of the grid line
                )

                # Draw the piece (stone) based on the current board state
                piece = board[row][col]
                if piece == const.PlayerColor.BLACK.value:
                    self.screen.blit(
                        self.image_black_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )  # Draw black stone
                elif piece == const.PlayerColor.WHITE.value:
                    self.screen.blit(
                        self.image_white_stone, (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )  # Draw white stone

    
  

    def mark_valid_fields(self, valid_moves: list):
        if not valid_moves:
            return

        virtual_prob = 1 / len(valid_moves)
        gamma = 0.4  # Macht kleine Wahrscheinlichkeiten besser sichtbar
        normalized = virtual_prob ** gamma

       
        cmap = cm.get_cmap(self.cmap)
        r, g, b, _ = cmap(normalized)
        color = (int(r * 255), int(g * 255), int(b * 255))

        for x, y in valid_moves:
            rect_position = (
                y * SQUARE_SIZE + 1,
                x * SQUARE_SIZE + 1,
            )
            rect_size = (
                SQUARE_SIZE - 2,
                SQUARE_SIZE - 2,
            )

            pygame.draw.rect(self.screen, color, pygame.Rect(rect_position, rect_size))


    def draw_plot(self, position=(900, 0)):
        """
        Zeichnet den gespeicherten Plot (cached), erstellt ihn nur neu, wenn sich die Werte geändert haben.
        """
        if self.values != self._last_plot_values:
            self._plot_image = generate_plot_image(self.values)
            self._last_plot_values = list(self.values)  # neue Kopie speichern

        if self.entropies != self._last_entropy_values:
            self._entropy_plot_image = generate_entropy_plot_image(self.entropies)
            self._last_entropy_values = list(self.entropies)

        if self._plot_image:
            self.screen.blit(self._plot_image, (int(position[0]), int(position[1])))
        
        if self._plot_image:
            self.screen.blit(self._entropy_plot_image, (900, 230))
    
    
    def highlight_last_move(self, move_history=None):
        if move_history:
            move = move_history[-1]
            row, col = from_algebraic_to_coords(move.lower())

            # Transparente Surface erstellen
            size = SQUARE_SIZE + 4
            highlight_surf = pygame.Surface((size, size), pygame.SRCALPHA)

            # RGBA – leicht transparentes Rot
           
            
            border_color = (80, 160, 255, 180)  # hellblau-transparent



            pygame.draw.rect(
                highlight_surf,
                border_color,
                highlight_surf.get_rect(),
                width=5,
                border_radius=6
            )

            # Blitten auf Haupt-Screen
            pos = (col * SQUARE_SIZE - 2, row * SQUARE_SIZE - 2)
            self.screen.blit(highlight_surf, pos)

            


    def append_value(self, value):
        self.values.append(value)
    
    def append_entropy(self, entropy):
        self.entropies.append(entropy)

    def update_game_data(self, min_prior=None, max_prior=None, max_depth=None, num_states=None, num_legal_moves=None, thinking_time=None, move=None, tree_nodes=None, search_efficiency=None, sps=None, brute_force_time=None):
        """
        Aktualisiert die Spieldaten für die Anzeige.

        Args:
            min_prior (float): Minimaler Prior-Wert.
            max_prior (float): Maximaler Prior-Wert.
            max_depth (int): Maximale Tiefe des MCTS.
            num_states (int): Anzahl der untersuchten Positionen.
            num_legal_moves (int): Anzahl der legalen Züge am Root.
        """
        if min_prior is not None:
            self.game_data["min_prior"] = min_prior
        if max_prior is not None:
            self.game_data["max_prior"] = max_prior
        if max_depth is not None:
            self.game_data["max_depth"] = max_depth
        if num_states is not None:
            self.game_data["num_states"] = num_states
        if num_legal_moves is not None:
            self.game_data["num_legal_moves"] = num_legal_moves
        if thinking_time is not None:
            self.game_data["thinking_time"] = thinking_time
        if move is not None:
            self.game_data["move"] = move
        if tree_nodes is not None:
            self.game_data["tree_nodes"] = tree_nodes
        if search_efficiency is not None:
            self.game_data["search_efficiency"] = search_efficiency
        if sps is not None:
            self.game_data["sps"] = sps
        if brute_force_time is not None:
            self.game_data["brute_force_time"] = brute_force_time

    
    def display_informations(self, position=(910, 460)):
        """
        Zeigt eine Informationsbox unterhalb der Value-Plotgrafik an.

        Args:
            priors (list of float): Liste der Policy-Prio-Wahrscheinlichkeiten.
            max_depth (int): Maximal erreichte Tiefe im MCTS.
            num_states (int): Anzahl untersuchter Positionen.
            num_legal_moves (int): Anzahl legaler Züge am Root.
            eff_branching_factor (float, optional): Durchschnittlicher Verzweigungsgrad.
            position (tuple): Linke obere Ecke der Box.
        """
        x, y = map(int, position)
        line_height = 22

        
        width = 400
        height = 260 

        # Hintergrundbox übermalen (verhindert Flackern)
        pygame.draw.rect(self.screen, (30, 30, 30), (x - 10, y - 10, width, height))  # Dunkelgrauer Hintergrund
        pygame.draw.rect(self.screen, (200, 200, 200), (x - 10, y - 10, width, height), 2)  # Optional: Rahmen

        font = pygame.font.SysFont("monospace", 20)
        
        def draw_line(label, value, offset):
            text = f"{label:<20} {value}"
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (x, y + offset * line_height))

        min_prior = self.game_data.get("min_prior", None)
        max_prior = self.game_data.get("max_prior", None)
        thinking_time = self.game_data.get("thinking_time", None)
        max_depth = self.game_data.get("max_depth", None)
        num_states = self.game_data.get("num_states", None)
        num_legal_moves = self.game_data.get("num_legal_moves", None)
        move = self.game_data.get("move", None)  
        tree_nodes = self.game_data.get("tree_nodes", None)
        efficiency = self.game_data.get("search_efficiency")
        sps = self.game_data.get("sps", None)
        brute_force_time = self.game_data.get("brute_force_time", None)
        s = None

        if brute_force_time is not None:
            brute_force_time, s = format_time(brute_force_time)
            brute_force_time = float(brute_force_time)  
            brute_force_time = f"{brute_force_time:.1e}"

          

          
    

    
        
        draw_line("Bedenkzeit [s]:", thinking_time, 0)
        draw_line("Expandierte Knoten:", num_states, 1)
        draw_line("Knoten/s", sps, 2)
        draw_line("Legale Züge (Root):", num_legal_moves, 3)
        draw_line("Max. Tiefe:", max_depth, 4)
        draw_line("Gespielter Zug:", move, 5)
        draw_line("Min Prior:", f"{min_prior}", 6)
        draw_line("Max Prior:", f"{max_prior}", 7)
        draw_line("Suchraum:", tree_nodes, 8)
        draw_line("Effizienzfaktor:", efficiency, 9)
        draw_line(f"Theor. Dauer [{s}]", brute_force_time, 10 )
        

      

    def draw_move_history(self, moves=None, start_x=900, start_y=710, max_width=1300, line_height=24):
        """
        Zeichnet die Züge kompakt im Format D3–C5–D6–E3... mit automatischem Zeilenumbruch.

        Args:
            moves (list): Liste der Züge als Strings (z.B. ["D3", "C5", "D6"])
            start_x (int): Startposition X
            start_y (int): Startposition Y
            max_width (int): Maximale Breite in Pixel (danach Umbruch)
            line_height (int): Zeilenhöhe in Pixel
        """
        font = pygame.font.SysFont("monospace", 24)
        if moves is None:
            moves = self.moves
        else:
            self.moves = moves
        move_text = "–".join(moves)

        words = move_text.split("–")
        x = start_x
        y = start_y
        line = ""

        for i, move in enumerate(words):
            new_line = f"{line}{move}–" if line else f"{move}–"
            width, _ = font.size(new_line)

            if x + width > max_width:
                # zeichne alte Zeile
                rendered = font.render(line.strip("–"), True, (255, 255, 255))
                self.screen.blit(rendered, (x, y))
                y += line_height
                line = f"{move}–".upper()
            else:
                line = new_line

        # letzte Zeile zeichnen
        if line:
            rendered = font.render(line.strip("–"), True, (255, 255, 255))
            self.screen.blit(rendered, (x, y))
