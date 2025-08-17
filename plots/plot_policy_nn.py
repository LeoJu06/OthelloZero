import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
from src.data_manager.data_manager import DataManager
from src.utils.index_to_coordinates import index_to_coordinates
import random
import torch

def plot_move_heatmap(move_counts, generation, save_path=None, vmax=None):
    total = move_counts.sum()
    move_freq = move_counts / total if total > 0 else move_counts

    if vmax is None:
        vmax = np.max(move_freq)

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Schöne Normierung für bessere visuelle Dynamik
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)  # Maximaler Wert aus den Beispieldaten
    im = ax.imshow(move_freq, cmap='Blues', norm=norm)

    # Achsenbeschriftung a–h, 1–8
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([chr(ord('a') + i) for i in range(8)], fontsize=10)
    ax.set_yticklabels([str(i + 1) for i in range(8)], fontsize=10)

    # Gitterlinien
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_title(f"Feldbewertung – Gen {generation}", fontsize=14, pad=10)
    ax.set_aspect('equal')

    # Farbbalken
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    # Optional speichern
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

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
        
        max_p = float("-inf")
       
        for idx, p in enumerate(policy[:64]):
            row, col = index_to_coordinates(idx)
            if (row, col) in center_coords:
                continue

            if p > max_p:
                
                heatmap[row, col] += p

    heatmap /= heatmap.sum()
    return heatmap

def create_policy_heatmap(model, generations=[1, 15, 55], num_samples=1000):
       
        dm = DataManager()

        for gen in generations:
            model = dm.load_model(latest_model=False, n=gen)
            model.eval()
            with torch.no_grad():
                heatmap = analyze_policy_output_heatmap(model, num_samples=num_samples)
            
            plot_move_heatmap(heatmap, generation=gen)
            save_policy_heatmap(heatmap, generation=gen)
   
    
    


def save_policy_heatmap(heatmap, generation):
    path = f"data/policy_heatmaps/policy_heatmap_gen_{generation}.npy"
    np.save(path, heatmap)

def load_policy_heatmap(generation):
    path = f"data/policy_heatmaps/policy_heatmap_gen_{generation}.npy"
    return np.load(path)

def plot_policy_heatmap(heatmap, annotate=True, title="Aggregierte Policy-Heatmap"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plotten der Heatmap mit Farbcodierung
    cax = ax.imshow(heatmap, cmap='YlGnBu', interpolation='nearest')

    # Achsenbeschriftungen
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7", "8"])

    # Gitterlinien zwischen Zellen
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Beschriftung der Werte (optional)
    if annotate:
        for i in range(8):
            for j in range(8):
                value = heatmap[i, j]
                if value > 0:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")

    # Farbskala hinzufügen
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Aggregierte Priorität")

    # Titel setzen
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    model = DataManager().load_model(latest_model=False, n=35)
    h = analyze_policy_output_heatmap(model, 1000)
    plot_policy_heatmap(h)
   


    sys.exit()

    #create_policy_heatmap(model=None, generations=[5], num_samples=15000)
    heatmaps = []
    for gen in [5, 15,55]:

        h = load_policy_heatmap(gen)
        heatmaps.append(h)
     

  

   
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors


    titles = ["Generation 5", "Generation 15", "Generation 55"]
    vmax = max(h.max() for h in heatmaps)
    norm = mcolors.PowerNorm(gamma=1, vmin=0, vmax=vmax)

    # Figure und Grid erstellen
    fig = plt.figure(figsize=(9, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

    axes = [plt.subplot(gs[i]) for i in range(3)]

    for ax, heatmap, title in zip(axes, heatmaps, titles):
        im = ax.imshow(heatmap, cmap="plasma", norm=norm, extent=[0, 8, 8, 0])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Farbskala ganz rechts (extra Achse)
    cbar_ax = plt.subplot(gs[3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalisierte Zughäufigkeit", rotation=90)

    plt.tight_layout()
    plt.savefig("heatmaps_generations_clean.png", dpi=300)
    plt.savefig("policy_heatmaps_generations.png", dpi=600, bbox_inches="tight", transparent=False)

    plt.show()

        
        
        