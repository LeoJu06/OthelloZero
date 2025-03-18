import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.neural_net.residual_block import ResidualBlock

# OthelloZeroModel with Residual Blocks and Convolutional Layers
class OthelloZeroModel(nn.Module):
    def __init__(self, board_size, action_size, device):
        super(OthelloZeroModel, self).__init__()

        self.device = device
        self.board_size = board_size  # Expected 8x8
        self.action_size = action_size

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Stack of Residual Blocks (10 blocks)
        self.residual_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(10)])
        
        # Output heads: one for actions (policy) and one for the value
        self.action_head = nn.Linear(in_features=256 * board_size * board_size, out_features=action_size)
        self.value_head = nn.Linear(in_features=256 * board_size * board_size, out_features=1)

        self.to(device)

    def forward(self, x):
        # Add a channel dimension
        x = x.unsqueeze(1)

        # Initial convolutional layer
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through all the residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Action and value heads
        action_logits = self.action_head(x)  # Rohe Logits für die Policy
        value_logit = self.value_head(x)     # Roher Wert

        return action_logits, torch.tanh(value_logit)  # Tanh für den Wert beibehalten

    def predict(self, board):
        """
        Makes predictions for a single board.

        Args:
            board (np.ndarray): 8x8 Board (single).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities (Softmax) and value.
        """
        # Convert to tensor and ensure batch and channel dimensions
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device).unsqueeze(0).unsqueeze(0)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            action_logits, v = self.forward(board)

        # Apply Softmax to action_logits
        action_probs = F.softmax(action_logits, dim=1)

        return action_probs.data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()

    def predict_batch(self, boards):
        """
        Makes predictions for a batch of boards.

        Args:
            boards (torch.Tensor): Batch of boards (N, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities (Softmax) and values for the batch.
        """
        # Ensure channel dimension
        boards = boards.unsqueeze(1).to(self.device)

        # Forward pass in evaluation mode
        self.eval()
        with torch.no_grad():
            action_logits, v = self.forward(boards)

        # Apply Softmax to action_logits
        action_probs = F.softmax(action_logits, dim=1)

        return action_probs.data.cpu().numpy(), v.data.cpu().numpy()