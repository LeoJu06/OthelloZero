import os 
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt



def report(self, won, lost, examples, duration, n):
    n = self.data_manager.get_iter_number()
    pdf_filename = f"data/reports/report_iteration_{n}.pdf"
    report_image = f"data/losses_plotted/Training_loss_{n}.png"

    # PDF-Dokument erstellen
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 770, f"Training Report - Iteration {n}")

    # Horizontale Linie für Struktur
    c.setStrokeColor(colors.black)
    c.line(100, 760, 500, 760)

    # Informationen über Trainingsergebnisse
    c.setFont("Helvetica", 12)
    info_text = [
        f"Games Won: {won}",
        f"Games Lost: {lost}",
        f"Number of Training Examples: {len(examples)}",
        f"Model Accepted: {self.accept_new_model(won)}",
        f"Training Duration: {duration:.2f} seconds",
    ]

    y_position = 730
    for line in info_text:
        c.drawString(100, y_position, line)
        y_position -= 20  # Abstand zwischen den Zeilen

    # Loss-Plot einfügen (falls vorhanden)
    if os.path.exists(report_image):
        c.drawString(100, y_position - 10, "Training Loss Plot:")
        c.drawImage(report_image, 100, y_position - 250, width=400, height=250)
    else:
        c.drawString(100, y_position - 10, "Training Loss Plot: (Not Found)")

    # PDF speichern
    c.save()
    print(f"📄 PDF gespeichert als {pdf_filename}")


def create_loss_figure(policy_losses, value_losses, epochs, iter_number):

    "creates a loss figure and saves it"


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), policy_losses, label="Policy Loss")
    plt.plot(range(1, epochs + 1), value_losses, label="Value Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Iteration {iter_number} ")
    plt.legend()


    plt.savefig(f"data/losses_plotted/Training_loss_{iter_number}", dpi=300)


def save_loss_data(policy_loss, value_loss, epochs, iter_number):

    path = f"data/losses_plotted/Training_loss_{iter_number}.json"

    data = {"policy_loss": policy_loss, "value_loss": value_loss, "epochs": epochs}

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_game_results(results, generation, edax_level, num_games):
    """
    Saves the game results to a JSON file.
    """
    data = {
        "generation": generation,
        "edax_level": edax_level,
        "num_games": num_games,
        "results": results
    }

    path = f"data/elo/results_gen_{generation}_edax_level_{edax_level}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Game results saved to {path}")