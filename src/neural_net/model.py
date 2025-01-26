import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OthelloZeroModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(OthelloZeroModel, self).__init__()

        self.device = device
        self.board_size = board_size  # Erwartet 8x8
        self.action_size = action_size

        # Hier verwenden wir die 64 Eingabefelder für das 8x8-Board
        self.fc1 = nn.Linear(
            in_features=self.board_size * self.board_size, out_features=64
        )
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Zwei Ausgabeköpfe: einer für Aktionen und einer für den Wert
        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)

        self.to(device)

    def forward(self, x):
        # Wenn Eingabe ein 2D-Board ist, sicherstellen, dass es 3D wird (Batch-Dimension hinzufügen)
        if len(x.shape) == 2:  # Einzelnes Board
            x = x.view(1, -1)  # Hinzufügen einer Batch-Dimension (1, 64)
        else:  # Batch von Boards
            x = x.view(x.size(0), -1)  # Batch-Größe bleibt gleich, flach machen

        # Forward-Pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        # Softmax für Aktionen, Tanh für den Wert
        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)


    def predict(self, board):
        """
        Macht Vorhersagen für ein einzelnes Board oder ein Batch von Boards.

        Args:
            board (np.ndarray): 8x8 Board (einzeln) oder Batch von Boards (N, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Aktionswahrscheinlichkeiten und Werte.
        """
        # Konvertiere zu Tensor und prüfe Batch-Dimension
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)

        if len(board.shape) == 2:  # Einzelzustand (8x8)
            board = board.unsqueeze(0)  # Hinzufügen der Batch-Dimension (1, 8, 8)

        # Forward-Pass im Auswertungsmodus
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()



if __name__ == "__main__":
    # Beispiel: Vorhersage eines Modells für das Othello-Spiel
    model = OthelloZeroModel(
        board_size=8, action_size=64, device="cuda"
    )  # Beispielgerät: 'cuda'
    board = np.zeros((8, 8))  # Ein leeres 8x8-Board als Beispiel
    board[3, 3] = 1  # Ein paar Zellen befüllen, z.B. für Othello

    # Vorhersage des Modells
    pi, v = model.predict(board)

    print("Aktionswahrscheinlichkeiten:", pi)
    print("Wert des Spiels:", v)
