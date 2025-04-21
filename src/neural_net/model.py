import torch
import torch.nn as nn
import torch.nn.functional as F
from src.neural_net.preprocess_board import preprocess_board
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class OthelloZeroModel(nn.Module):
    def __init__(self, board_size=8, action_size=64, num_res_blocks=10, device="cpu"):
        super().__init__()
        self.device = device
        self.board_size = board_size
        
        # Input: 3 Kanäle (Spieler, Gegner, legale Züge)
        self.conv1 = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(192)
        
        # Residual Tower
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(192) for _ in range(num_res_blocks)
        ])
        
        # Policy Head (Original unverändert)
        self.policy_conv = nn.Conv2d(192, 2, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)
        
        # OPTIMIERTER Value Head (stabilisiert)
        self.value_conv = nn.Conv2d(192, 4, kernel_size=1)  # Nur 4 Kanäle (statt 32)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc = nn.Linear(4 * board_size * board_size, 1)  # Direkte Regression

        self.to(device)

    def forward(self, x):
        # Initial Conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        
        # Policy Head (Original)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # NEUER Value Forward (stabilisiert)
        v = F.relu(self.value_bn(self.value_conv(x)))  # [B, 4, 8, 8]
        v = v.view(v.size(0), -1)                     # [B, 256]
        v = torch.tanh(self.value_fc(v))               # [B, 1]
        
        return p, v


       
    def predict(self, board):
        """
        Makes predictions for a single board.

        Args:a
            board (np.ndarray): Shape (8, 8), single board.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and value.
        """
        board = preprocess_board(board)  # Convert to [3 8, 8]
        board = torch.FloatTensor(board).to(self.device).unsqueeze(0)  # Add batch dim -> [1, 2, 8, 8]

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)
            pi = F.softmax(pi, dim=1)

        return pi.data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()


    def predict_batch(self, boards):
        """
        Makes predictions for a batch of boards.

        Args:
            boards (np.ndarray): Shape (batch, 8, 8).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Action probabilities and values for the batch.
        """
        boards = np.array([preprocess_board(board) for board in boards])  # Convert all to [batch, 2, 8, 8]
        boards = torch.FloatTensor(boards).to(self.device)

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(boards)
            pi = F.softmax(pi, dim=1)

        return pi.data.cpu().numpy(), v.data.cpu().numpy()


