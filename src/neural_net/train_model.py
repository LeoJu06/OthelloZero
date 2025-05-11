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


def calculate_epochs(buffer_size, batch_size, samples_per_iteration=3):
    return ( buffer_size * samples_per_iteration) // batch_size  # 3× Coverage



def train(model, replay_buffer, batch_size=512, lr=0.1, epochs=10_000):

    epochs = calculate_epochs(len(replay_buffer), batch_size, 3)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    

    value_losses  =[]
    policy_losses = []
    
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size)

        # Effiziente Konvertierung mit numpy
        states_np = np.array([preprocess_board(x[0]) for x in batch])
        policies_np = np.array([x[1] for x in batch])
        values_np = np.array([x[2] for x in batch])

        # Konvertierung zu Tensoren
        states = torch.tensor(states_np, dtype=torch.float32).to(model.device)
        policy_targets = torch.tensor(policies_np, dtype=torch.float32).to(model.device)
        value_targets = torch.tensor(values_np, dtype=torch.float32).to(model.device)
        
        # Forward pass
        policy_pred, value_pred = model(states)

        policy_log_probs = F.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_targets * policy_log_probs).sum(dim=1).mean()

      
        value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.5f} | Loss: {total_loss.item():.4f}"
               f"(Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")
        
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())

    return model, policy_losses, value_losses

if __name__ == "__main__":
    # Load data
    data_manager = DataManager()
    examples = data_manager.load_examples()
    
    # Initialize the model
    model = data_manager.load_model()
    
    # Create replay buffer and add examples
    replay_buffer = ReplayBuffer()
    replay_buffer.add(examples)
    print(f"Loaded {len(examples)} training examples")
    
    # Train with more stable parameters
    train(
        model,
        replay_buffer=replay_buffer,
        lr=0.0001,  # Reduzierte Lernrate für mehr Stabilität
        batch_size=1024,  # Optimierte Batch-Größe
        epochs=None # epochen werden berechnet. 
    )

