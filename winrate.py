import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_dir = "data/elo"
data_list = []

# Gehe alle Dateien im Verzeichnis durch
for file in os.listdir(results_dir):
    if file.startswith("results_gen") and file.endswith(".json"):
        filepath = os.path.join(results_dir, file)
        with open(filepath, "r") as f:
            data = json.load(f)
            gen = data["generation"]
            edax = data["edax_level"]
            results = data["results"]
            wins = results["win"]
            losses = results["loss"]
            draws = results["draw"]
            data_list.append({
                "generation": gen,
                "edax_level": edax,
                "wins": wins,
                "losses": losses,
                "draws": draws
            })

# In DataFrame und Winrate berechnen
df = pd.DataFrame(data_list)
df["winrate"] = 100 * df["wins"] / (df["wins"] + df["losses"] + df["draws"])

# Pivot für Heatmap
heatmap_data = df.pivot(index="generation", columns="edax_level", values="winrate")

# Plot
plt.figure(figsize=(10, 6))
c = plt.imshow(heatmap_data, aspect="auto", cmap="YlGnBu", origin="lower", vmin=0, vmax=100)
plt.colorbar(label="Gewinnrate (%)")
plt.xticks(ticks=range(len(heatmap_data.columns)), labels=[f"Edax {l}" for l in heatmap_data.columns])
plt.yticks(ticks=range(len(heatmap_data.index)), labels=[f"Gen {g}" for g in heatmap_data.index])
plt.xlabel("Edax-Level")
plt.ylabel("Modellgeneration")
plt.title("Heatmap der Gewinnraten: Modellgeneration vs. Edax-Level")
plt.tight_layout()
plt.show()
