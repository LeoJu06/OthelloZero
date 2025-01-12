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
        self.fc1 = nn.Linear(in_features=self.board_size * self.board_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Zwei Ausgabeköpfe: einer für Aktionen und einer für den Wert
        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)

        self.to(device)

    def forward(self, x):
        # Eingabe 'x' ist 8x8, aber wir müssen es flach machen (64 Elemente)
        x = x.view(-1, self.board_size * self.board_size)  # Umwandlung von 8x8 zu 64
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    def predict(self, board):
        # board ist nun ein 8x8 2D-Array, daher müssen wir es zuerst in ein 1D-Array umwandeln
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.board_size * self.board_size)  # Umwandlung zu 1D (64)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


if __name__ == "__main__":

    # Beispiel: Vorhersage eines Modells für das Othello-Spiel
    model = OthelloZeroModel(board_size=8, action_size=64, device='cuda')  # Beispielgerät: 'cuda'
    board = np.zeros((8, 8))  # Ein leeres 8x8-Board als Beispiel
    board[3, 3] = 1  # Ein paar Zellen befüllen, z.B. für Othello

    # Vorhersage des Modells
    pi, v = model.predict(board)

    print("Aktionswahrscheinlichkeiten:", pi)
    print("Wert des Spiels:", v)
