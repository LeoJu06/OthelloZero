import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# Pfad zum Verzeichnis mit den gespeicherten Matchdaten
results_dir = "data/elo"

# Alle JSON-Dateien sammeln
files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

# Matchdaten extrahieren
matches = []
for file in files:
    with open(os.path.join(results_dir, file), "r") as f:
        data = json.load(f)
        gen = f"Model_{data['generation']}"
        edax = f"Edax_{data['edax_level']}"
        res = data["results"]
        matches.append({
            "Player_A": gen,
            "Player_B": edax,
            "Wins_A": res["win"],
            "Wins_B": res["loss"],
            "Draws": res["draw"]
        })

# Parameter
K = 32
num_runs = 1000
all_players = set()
elo_accumulator = defaultdict(list)

# Mehrfache Shuffles & Mittelwertbildung
for run in range(num_runs):
    ratings = defaultdict(lambda: 1000)
    match_sequence = matches.copy()
    random.shuffle(match_sequence)

    for match in match_sequence:
        A, B = match["Player_A"], match["Player_B"]
        Ra, Rb = ratings[A], ratings[B]

        for _ in range(match["Wins_A"]):
            Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
            Ra += K * (1 - Ea)
            Rb += K * (0 - (1 - Ea))

        for _ in range(match["Wins_B"]):
            Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
            Ra += K * (0 - Ea)
            Rb += K * (1 - (1 - Ea))

        for _ in range(match["Draws"]):
            Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
            Ra += K * (0.5 - Ea)
            Rb += K * (0.5 - (1 - Ea))

        ratings[A], ratings[B] = Ra, Rb
        all_players.update([A, B])

    for player in all_players:
        elo_accumulator[player].append(ratings[player])

# Durchschnittswerte berechnen
elo_means = {
    player: round(np.mean(scores)) for player, scores in elo_accumulator.items()
}

# In DataFrame überführen
elo_df = pd.DataFrame([
    {"Player": player, "Elo": elo_means[player]} for player in sorted(elo_means, key=elo_means.get, reverse=True)
])

print(elo_df)