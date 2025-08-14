import matplotlib.pyplot as plt
import numpy as np
from src.othello.othello_game import OthelloGame
from src.neural_net.preprocess_board import preprocess_board

def plot_input_tensor_final(input_tensor):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    titles = ['Aktueller Spieler', 'Gegner', 'Legale Züge']
    colors = ["#2C2C2C", '#AAAAAA', "#2951BF"]  # dunkelgrau, hellgrau, blau

    for i in range(3):
        ax = axs[i]

        # Rechtecke für jedes Feld zeichnen
        for x in range(8):
            for y in range(8):
                value = input_tensor[i, y, x]
                if value > 0.5:
                    color = colors[i]
                else:
                    color = 'white'
                rect = plt.Rectangle((x, y), 1, 1, facecolor=color)
                ax.add_patch(rect)

        # Raster und Achseneinstellungen
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks(np.arange(0, 9, 1))
        ax.set_yticks(np.arange(0, 9, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.grid(True, color='black', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(titles[i], fontsize=14)

    plt.suptitle('Netzwerkinput (3×8×8)', fontsize=16)
    plt.tight_layout()
    # Speichern für die Arbeit (optional):
    # plt.savefig("netzwerkinput.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    game = OthelloGame()
    board = game.get_init_board()  # Startaufstellung
    board = game.get_canonical_board(board, 1)  # Spieler -1 = Schwarz
    input_tensor = preprocess_board(board)
    plot_input_tensor_final(input_tensor)
