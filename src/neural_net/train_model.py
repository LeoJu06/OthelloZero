import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from random import shuffle
import matplotlib.pyplot as plt

from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager


def train(model, data, epochs=10, batch_size=2048, lr=0.001, save_training_plot=False):
    """
    Trains the AlphaZero model with the given data.

    Args:
        model (OthelloZeroModel): The neural network.
        data (list): Training data [(board, policy, value), ...]
        epochs (int): Number of epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.

    Returns:
        None
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
    policy_loss_fn = nn.CrossEntropyLoss()  # Cross-Entropy für die Policy
    value_loss_fn = nn.MSELoss()           # MSE für den Value

    # Convert data to tensors
    boards = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(model.device)
    policies = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(model.device)  # Wahrscheinlichkeitsverteilungen
    values = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(model.device)

    dataset = torch.utils.data.TensorDataset(boards, policies, values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch_boards, batch_policies, batch_values in dataloader:
            optimizer.zero_grad()
            
            # Forward pass: Model gibt Logits zurück
            pred_policies, pred_values = model(batch_boards)

            # Policy Loss: CrossEntropyLoss erwartet Logits und Ziel-Wahrscheinlichkeiten
            policy_loss = policy_loss_fn(pred_policies, batch_policies)

            # Value loss (MSE für den Wert)
            value_loss = value_loss_fn(pred_values.squeeze(), batch_values)

            # Total loss
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        print(f"Epoch {epoch+1}/{epochs} - Policy Loss: {avg_policy_loss:.4f} - Value Loss: {avg_value_loss:.4f}")

    return model, policy_losses, value_losses


if __name__ == "__main__":
    # Load data
    data_manager = DataManager()
    examples = data_manager.load_examples()[:10000]
    print(len(examples))

    # Initialize the model
    model = OthelloZeroModel(8, 65, "cuda")  # 65 mögliche Züge (einschließlich "Pass")
    
    # Train the model
    train(model, examples, lr=0.004, batch_size=1024, epochs=100)