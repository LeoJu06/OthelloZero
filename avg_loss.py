import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Stil setzen (für sauberes, modernes Aussehen)
plt.style.use("seaborn-v0_8-muted")

folder = "data/losses_plotted"
all_policy_losses = []
all_value_losses = []
iterations = []

# Sammle Daten aus allen vorhandenen Dateien
for i in range(56):  # Iterationen 0–55
    filename = f"Training_loss_{i}.json"
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
            policy_avg = np.mean(data["policy_loss"])
            value_avg = np.mean(data["value_loss"])
            all_policy_losses.append(policy_avg)
            all_value_losses.append(value_avg)
            iterations.append(i)
    else:
        print(f"Datei nicht gefunden: {filepath}")

# Konvertiere zu NumPy für spätere Weiterverarbeitung
iterations = np.array(iterations)
policy_losses = np.array(all_policy_losses)
value_losses = np.array(all_value_losses)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Glättung mit gleitendem Mittel (Fenstergröße 5)
policy_smooth = pd.Series(policy_losses).rolling(window=5, center=True).mean()
value_smooth = pd.Series(value_losses).rolling(window=5, center=True).mean()

# Plot erstellen
fig, ax1 = plt.subplots(figsize=(10, 6))

# Erste Y-Achse: Policy Loss
color1 = "tab:blue"
ax1.set_xlabel("Generation")
ax1.set_ylabel("Policy Loss", color=color1)
p1, = ax1.plot(iterations, policy_smooth, label="Policy Loss (geglättet)", color=color1, linewidth=2)
ax1.tick_params(axis="y", labelcolor=color1)

# Zweite Y-Achse: Value Loss
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Value Loss", color=color2)
p2, = ax2.plot(iterations, value_smooth, label="Value Loss (geglättet)", color=color2, linewidth=2, linestyle="--")
ax2.tick_params(axis="y", labelcolor=color2)

# Legende kombinieren
lines = [p1, p2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

# Titel und Layout
fig.suptitle("Verlauf der Trainingsverluste über die Generationen", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.show()
