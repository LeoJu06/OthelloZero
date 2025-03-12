import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from random import shuffle
import matplotlib.pyplot as plt

from src.neural_net.model import OthelloZeroModel
from src.data_manager.data_manager import DataManager
if __name__ == "__main__":
# Load data
    data_manager = DataManager()
    examples = data_manager.load_examples(0)

    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OthelloZeroModel(8, 65, device)

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
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")  # KL divergence for probabilities
    value_loss_fn = nn.MSELoss()

    # Convert data to tensors
    boards = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32).to(model.device)
    policies = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32).to(model.device)
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
            
            pred_policies, pred_values = model(batch_boards)

            # **Policy Loss Fix:**
            pred_policies = torch.log(pred_policies + 1e-8)  # Log probabilities for KLDivLoss
            policy_loss = policy_loss_fn(pred_policies, batch_policies)

            # Value loss (MSE for the value)
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

        print(f"Epoch {epoch+1}/{epochs} - Policy Loss: {total_policy_loss:.4f} - Value Loss: {total_value_loss:.4f}")
    
    
  

    return model, policy_losses, value_losses

if __name__ == "__main__":
    print(f"Number of training examples: {len(examples)}")
    train(model, examples)
    data_manager.save_model(model)
    print("saved")
    model = data_manager.load_model(None)
    train(model, examples)
    model = data_manager.load_model(None)
    train(model, examples)

