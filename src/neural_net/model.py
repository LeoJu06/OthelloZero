import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.neural_net.residual_block import ResidualBlock




# OthelloZeroModel with Residual Blocks
class OthelloZeroModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(OthelloZeroModel, self).__init__()

        self.device = device
        self.board_size = board_size  # Expected 8x8
        self.action_size = action_size

        # Initial fully connected layers
        self.fc1 = nn.Linear(in_features=self.board_size * self.board_size, out_features=256)
        
        # Stack of Residual Blocks (9 blocks)
        self.residual_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(10)])
        
        # Output heads: one for actions and one for the value
        self.action_head = nn.Linear(in_features=256, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=256, out_features=1)

        self.to(device)

    def forward(self, x):
        # Flatten the board(s) for the fully connected layers
        x = x.view(x.size(0), -1)  # Batch size stays the same, flatten board

        # Initial fully connected layer
        x = F.relu(self.fc1(x))

        # Pass through all the residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Action and value heads
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
    





