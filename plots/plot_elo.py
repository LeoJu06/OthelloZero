import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def plot_elo_balken():
    # Elo-Werte aus der gemittelten Bewertung
    elo_data = [
        ("Edax_6", 1892),
        ("Model_55", 1778),
        ("Model_50", 1681),
        ("Edax_5", 1574),
        ("Model_45", 1416),
        ("Model_40", 1332),
        ("Model_35", 1306),
        ("Model_30", 1157),
        ("Edax_4", 1082),
        ("Model_25", 1008),
        ("Model_20", 928),
        ("Model_15", 775),
        ("Edax_3", 658),
        ("Model_10", 541),
        ("Edax_2", 430),
        ("Model_5", 389),
        ("Edax_1", 103),
        ("Model_0", -50),
    ]

    # In DataFrame überführen
    df = pd.DataFrame(elo_data, columns=["Player", "Elo"])
    df_sorted = df.sort_values("Elo", ascending=True)

    # Farben: Modelle = blau, Edax = rot
    colors = ["red" if "Edax" in p else "blue" for p in df_sorted["Player"]]

    # Plot-Stil setzen
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_sorted["Player"], df_sorted["Elo"], color=colors)
    ax.set_xlabel("Elo-Wert")
    ax.set_title("Elo-Bewertung: Vergleich der Modelle mit Edax")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    plt.show()

    return fig

#fig_balken = plot_elo_balken()



import matplotlib.pyplot as plt
import pandas as pd

def plot_elo_lernkurve():
    # Modell-Elo-Werte
    elo_data = [
        ("Edax_6", 1892),
        ("Model_55", 1778),
        ("Model_50", 1681),
        ("Edax_5", 1574),
        ("Model_45", 1416),
        ("Model_40", 1332),
        ("Model_35", 1306),
        ("Model_30", 1157),
        ("Edax_4", 1082),
        ("Model_25", 1008),
        ("Model_20", 928),
        ("Model_15", 775),
        ("Edax_3", 658),
        ("Model_10", 541),
        ("Edax_2", 430),
        ("Model_5", 389),
        ("Edax_1", 103),
        ("Model_0", -50),
    ]

    # DataFrame vorbereiten
    df_models = pd.DataFrame(elo_data, columns=["Player", "Elo"])
    df_models["Generation"] = df_models["Player"].str.extract(r"Model_(\d+)").astype(int)
    df_models = df_models.sort_values("Generation")

    # Plot-Konfiguration
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df_models["Generation"],
        df_models["Elo"],
        marker="o",
        linestyle="-",
        color="#1f77b4",
        linewidth=2,
        label="Modell-Elo"
    )

    # Achsen & Titel
    ax.set_title("Lernkurve: Elo-Entwicklung über Generationen")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Elo-Wert")
    ax.set_xticks(df_models["Generation"])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    plt.show()

    return fig



import matplotlib.pyplot as plt
import pandas as pd

# Modell-Elo-Werte
elo_data = [
    ("Model_0", -50),
    ("Model_5", 389),
    ("Model_10", 541),
    ("Model_15", 775),
    ("Model_20", 928),
    ("Model_25", 1008),
    ("Model_30", 1157),
    ("Model_35", 1306),
    ("Model_40", 1332),
    ("Model_45", 1416),
    ("Model_50", 1681),
    ("Model_55", 1778),
]

# Edax-Level als horizontale Referenzlinien
edax_levels = {
    "Edax 1": 103,
    "Edax 2": 430,
    "Edax 3": 658,
    "Edax 4": 1082,
    "Edax 5": 1574,
    "Edax 6": 1892,
}

# DataFrame vorbereiten
df_models = pd.DataFrame(elo_data, columns=["Player", "Elo"])
df_models["Generation"] = df_models["Player"].str.extract(r"Model_(\d+)").astype(int)
df_models = df_models.sort_values("Generation")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_models["Generation"], df_models["Elo"], marker="o", linestyle="-", color="blue", label="Modell-Elo")

# Edax-Level als horizontale Linien mit besser positionierter Beschriftung
for label, value in edax_levels.items():
    plt.axhline(y=value, linestyle="--", color="red", alpha=0.5)
    plt.text(
        x=max(df_models["Generation"]) + -0.2,  # weiter rechts
        y=value + 15,                          # leicht oberhalb der Linie
        s=label,
        va="bottom",                          # vertikale Ausrichtung: Unterkante
        ha="left",                            # horizontale Ausrichtung
        fontsize=10,
        color="red"
    )


# Achsen und Titel
plt.title("Elo-Entwicklung über Trainingsgenerationen mit Edax-Vergleich")
plt.xlabel("Generation")
plt.ylabel("Elo-Wert")
plt.legend(loc="lower right", fontsize="small")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

