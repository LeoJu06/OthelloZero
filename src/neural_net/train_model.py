import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from random import shuffle
from src.neural_net.preprocess_board import preprocess_board
from src.data_manager.data_manager import DataManager
from src.data_manager.replay_buffer import ReplayBuffer


import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Für Fortschrittsbalken


def calculate_epochs(buffer_size, batch_size, samples_per_iteration=2):
    return ( buffer_size * samples_per_iteration) // batch_size  # 4× Coverage

def calculate_training_steps(buffer_size: int, batch_size: int, coverage: int = 5) -> int:
    """
    Berechnet die Anzahl SGD-Schritte, um jeden Datenpunkt im Buffer 'coverage'-mal zu sehen.
    Gilt bei 'on-the-fly' Augmentation (z. B. 8 symmetrische Varianten pro Sample).
    
    - buffer_size: Anzahl originaler Samples im ReplayBuffer
    - batch_size: z. B. 128 oder 2048
    - coverage: gewünschte Coverage (z. B. 2–4)
    """
    total_samples = buffer_size * coverage
    return total_samples // batch_size



def train(model, replay_buffer, batch_size=2048, lr=0.01, max_epochs=100, samples_per_iteration=10):


    device = model.device
    buffer_size = len(replay_buffer)
    epochs = max_epochs
     # Dynamische Anpassung:
    alpha = 0.75
    model.train()
    
    # Optimizer + OneCycle Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,          # Spitze während der mittleren Phase
    epochs=epochs,
    steps_per_epoch=1,
    pct_start=0.1,        # 10% der Epochen für Warmup
    div_factor=100,       # Start-LR = max_lr/300 = 1e-5
    final_div_factor=1000, # End-LR = max_lr/3000 = 1e-6
    anneal_strategy='cos' # Glatte Abnahme
    )

    value_losses, policy_losses = [], []

    for epoch in range(epochs):

       
        batch = replay_buffer.sample(batch_size)

        # Preprocessing
        states_np = np.array([preprocess_board(x[0]) for x in batch])
        policies_np = np.array([x[1] for x in batch])
        values_np = np.array([x[2] for x in batch])

        states = torch.tensor(states_np, dtype=torch.float32).to(device)
        policy_targets = torch.tensor(policies_np, dtype=torch.float32).to(device)
        value_targets = torch.tensor(values_np, dtype=torch.float32).to(device)

        # Forward
        policy_pred, value_pred = model(states)

        # Loss Functions
        policy_log_probs = F.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_targets * policy_log_probs).sum(dim=1).mean()

        value_loss = F.mse_loss(value_pred.squeeze(), value_targets)

        total_loss = policy_loss + value_loss  # Kombinierter Loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Total Loss: {total_loss.item():.4f} (Policy: {policy_loss.item():.4f}, "
              f"Value: {value_loss.item():.4f}) alpha: {alpha}")

        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())

    return model, policy_losses, value_losses


