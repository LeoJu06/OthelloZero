import random
from src.utils.coordinates_to_index import coordinates_to_index
from src.config.hyperparameters import Hyperparameters
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# Example usage in dummy_model_predict
def dummy_model_predict(leaf_node, model):
    """
    Use a neural network to predict action probabilities and board value, with GPU support.
    """
    # Convert board to tensor format and move to device
    board_tensor = (
        torch.tensor(leaf_node.board.board, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(Hyperparameters.Neural_Network["device"])
    )

    # Forward pass
    action_logits, value = model(board_tensor)

    # Convert logits to probabilities and handle invalid moves
    action_probs = F.softmax(action_logits, dim=1).squeeze().detach().cpu().numpy()
    valid_moves = leaf_node.board.valid_moves()
    valid_indices = [coordinates_to_index(x, y) for x, y in valid_moves]
    if not valid_indices:
        pass_prob = action_probs[64]
    action_probs = [action_probs[i] if i in valid_indices else 0 for i in range(65)]
    if not valid_indices:
        action_probs[64] = pass_prob

    # Normalize probabilities
    total_prob = sum(action_probs)
    if total_prob > 0:
        action_probs = [prob / total_prob for prob in action_probs]
    else:
        # Assign uniform distribution if no valid moves
        action_probs = [
            1 / len(valid_moves) if i in valid_indices else 0 for i in range(65)
        ]
    # action_probs = [1.0 / len(valid_moves)] * len(valid_moves)

    return action_probs, value.item()


# Define a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 65)  # Policy head (65 moves)
        self.value = nn.Linear(16 * 8 * 8, 1)  # Value head

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        policy = self.fc(x)  # Policy logits
        value = self.value(x)  # Scalar value
        return torch.softmax(policy, dim=-1), torch.tanh(value)


def neural_network_evaluate(
    batch, model, device=Hyperparameters.Neural_Network["device"]
):
    """
    Evaluates a batch of board states using the neural network.
    """
    with torch.no_grad():
        states = (
            torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1).to(device)
        )  # Add channel dim
        policies, values = model(states)
        return policies.cpu().numpy(), values.cpu().numpy()
