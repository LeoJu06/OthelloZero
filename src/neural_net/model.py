
import random
from src.utils.coordinates_to_index import coordinates_to_index
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm


def dummy_model_predict(board):
    """
    A dummy prediction model that returns random action probabilities and a random value.
    """
    value_head = random.choice([x / 10 for x in range(1, 11)])
    action_probs = [0 for _ in range(64)]
    for x in range(8):
        for y in range(8):
            action_probs[coordinates_to_index(x, y)] = random.choice([x / 10 for x in range(1, 11)])
    return action_probs, value_head


# Define a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 64)  # Policy head (64 moves)
        self.value = nn.Linear(16 * 8 * 8, 1)  # Value head

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        policy = self.fc(x)  # Policy logits
        value = self.value(x)  # Scalar value
        return torch.softmax(policy, dim=-1), torch.tanh(value)

def neural_network_evaluate(batch, model, device):
    """
    Evaluates a batch of board states using the neural network.
    """
    with torch.no_grad():
        states = torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dim
        policies, values = model(states) 
        return policies.cpu().numpy(), values.cpu().numpy()