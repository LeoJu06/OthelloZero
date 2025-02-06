import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OthelloZeroModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(OthelloZeroModel, self).__init__()

        self.device = device
        self.board_size = board_size  # Expected 8x8
        self.action_size = action_size

        # Using 64 input fields for the 8x8 board
        self.fc1 = nn.Linear(
            in_features=self.board_size * self.board_size, out_features=64
        )
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Two output heads: one for actions and one for the value
        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)

        self.to(device)

    def forward(self, x):
        # Flatten the board(s) for the fully connected layers
        x = x.view(x.size(0), -1)  # Batch size stays the same, flatten board

        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        # Softmax for actions, Tanh for the value
        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    def predict(self, board):
        """
        Makes predictions for a single board.

        Args:
            board (np.ndarray): 8x8 Board (single).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and value.
        """
        # Convert to tensor and ensure batch dimension
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device).unsqueeze(0)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()

    def predict_batch(self, boards):
        """
        Makes predictions for a batch of boards.

        Args:
            boards (np.ndarray): Batch of boards (N, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and values for the batch.
        """
        # Convert to tensor
        boards = boards.to(self.device)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(boards)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()


if __name__ == "__main__":
    # Example: Prediction for Othello game model
    model = OthelloZeroModel(
        board_size=8, action_size=64, device="cuda"
    )  # Example device: 'cuda'

    # Single board example
    single_board = np.zeros((8, 8))  # An empty 8x8 board
    single_board[3, 3] = 1  # Example piece placement

    # Batch of boards example
    batch_boards = torch.tensor(np.stack([single_board, single_board * -1]), dtype=torch.float32)  # Two boards in a batch

    # Single board prediction
    pi_single, v_single = model.predict(single_board)
    print("Single Board - Action Probabilities:", pi_single)
    print("Single Board - Value:", v_single)

    # Batch board prediction
    pi_batch, v_batch = model.predict_batch(batch_boards)
    print("Batch Boards - Action Probabilities:", pi_batch)
    print("Batch Boards - Values:", v_batch)
