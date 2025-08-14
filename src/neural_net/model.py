import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.neural_net.preprocess_board import preprocess_board


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
    def __init__(self, board_size=8, action_size=64, num_res_blocks=4, channels=64, device="cpu"):
        super().__init__()
        self.device = device
        
        # Input block
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        self.final_bn = nn.BatchNorm2d(channels)
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_size)

        # Value head (erweitert)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_fc3 = nn.Linear(64, 1)

        self.to(device)

    def forward(self, x):
        # Shared backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)
        x = self.final_bn(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))

        return p, v

    def predict(self, board):
        """
        Predicts action probabilities and value for a single board.
        """
        board = preprocess_board(board)
        board = torch.FloatTensor(board).to(self.device).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)
            pi = F.softmax(pi, dim=1)

        return pi.cpu().numpy().squeeze(), v.cpu().numpy().squeeze()

    def predict_batch(self, boards):
        """
        Predicts action probabilities and values for a batch of boards.
        """
        boards = np.array([preprocess_board(board) for board in boards])
        boards = torch.FloatTensor(boards).to(self.device)

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(boards)
            pi = F.softmax(pi, dim=1)

        return pi.cpu().numpy(), v.cpu().numpy()
    
    def predict_raw(self, board):
        """
        Gibt rohe Policy-Logits (ohne Softmax) und Value zurück – ideal für Auswertungen.
        """
        board = preprocess_board(board)
        board = torch.FloatTensor(board).to(self.device).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)  # <– KEIN Softmax

        return pi.cpu().numpy().squeeze(), v.cpu().numpy().squeeze()



if __name__ == "__main__":
    model = OthelloZeroModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
