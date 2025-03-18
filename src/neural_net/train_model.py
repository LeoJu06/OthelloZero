import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager


def train(model, data, epochs=10, batch_size=2048, lr=0.001, accumulation_steps=4, save_training_plot=False):
    """
    Trains the AlphaZero model with the given data.

    Args:
        model (OthelloZeroModel): The neural network.
        data (list): Training data [(board, policy, value), ...]
        epochs (int): Number of epochs.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        accumulation_steps (int): Number of steps for gradient accumulation.
        save_training_plot (bool): Whether to save the training loss plot.
    """
    # Initialize optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
    policy_loss_fn = nn.CrossEntropyLoss()  # Cross-Entropy for Policy
    value_loss_fn = nn.MSELoss()           # MSE for Value

    # Convert data to tensors
    boards = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(model.device)
    policies = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(model.device)  # Probability distributions
    values = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32).to(model.device)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        optimizer.zero_grad()

        for i, (batch_boards, batch_policies, batch_values) in enumerate(dataloader):
            # Forward pass
            pred_policies, pred_values = model(batch_boards)

            # Policy and Value Loss
            policy_loss = policy_loss_fn(pred_policies, batch_policies)
            value_loss = value_loss_fn(pred_values.squeeze(), batch_values)
            loss = (policy_loss + value_loss) / accumulation_steps  # Normalize loss for accumulation

            # Backward pass
            loss.backward()

            # Gradient Accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        # Calculate average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        print(f"Epoch {epoch+1}/{epochs} - Policy Loss: {avg_policy_loss:.4f} - Value Loss: {avg_value_loss:.4f}")

    # Save training plot if requested
    if save_training_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(policy_losses, label="Policy Loss")
        plt.plot(value_losses, label="Value Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig("training_loss.png")
        plt.close()

    return model, policy_losses, value_losses


if __name__ == "__main__":
    # Load data
    data_manager = DataManager()
    examples = data_manager.load_examples()[:10000]  # Use a subset of the data for testing
    print(f"Loaded {len(examples)} training examples.")

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloZeroModel(8, 65, device)  # 65 possible moves (including "Pass")
    print(f"Model initialized on {device}.")

    # Train the model
    trained_model, policy_losses, value_losses = train(
        model, examples, lr=0.004, batch_size=1024, epochs=100, accumulation_steps=4, save_training_plot=True
    )
    print("Training completed.")