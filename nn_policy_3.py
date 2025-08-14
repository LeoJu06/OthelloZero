import numpy as np
from tqdm import tqdm
from src.othello.othello_game import OthelloGame
from src.othello.game_constants import PlayerColor
from src.mcts.mcts import MCTS
from src.utils.index_to_coordinates import index_to_coordinates
import random
from src.data_manager.data_manager import DataManager
from plot_policy_nn import plot_move_heatmap
from src.neural_net.model import preprocess_board


import torch
def analyze_policy_output_heatmap(model, num_samples=1000):
    heatmap = np.zeros((8, 8), dtype=np.float32)
    dm = DataManager()
      # Koordinaten der vier Startfelder (d4, e4, d5, e5)
    center_coords = {(3, 3), (3, 4), (4, 3), (4, 4)}  # 0-indiziert
    
    for _ in tqdm(range(num_samples)):
        # Zufälliger Zustand aus Replay Buffer
        data = dm.load_examples(random.randint(0, 55))
        board = random.choice(data)[0]
        
        # Netzwerkausgabe
       
        
        policy, _ = model.predict_raw(board)
        
        
        for idx, p in enumerate(policy[:64]):
            row, col = index_to_coordinates(idx)
            if (row, col) in center_coords:
                continue
            heatmap[row, col] += p

    heatmap /= heatmap.sum()
    return heatmap


if __name__ == "__main__":

    
    model = DataManager().load_model(latest_model=False, n=40)  
    model.eval()
    with torch.no_grad():
        # Beispielaufruf der Funktion
        # Hier wird die Heatmap für die Modellpolicy erstellt
        # und anschließend geplottet.
        heatmap = analyze_policy_output_heatmap(model, num_samples=10_000)

    
    plot_move_heatmap(heatmap, generation=40)
  
    #create_heatmaps_opening()
    #show_opening_heatmaps()
    #create_heatmaps_endgame()
    #show_endgame_heatmaps()

    #show_opening_heatmaps()
    #h = analyze_endgame_policy_heatmap(model=model, num_games=100, num_simulations=10, min_turn=44)
   # plot_move_heatmap(h, generation=1)


    #create_heatmaps_endgame()
    #create_heatmaps()
