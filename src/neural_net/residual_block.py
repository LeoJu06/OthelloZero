import torch.nn as nn
import torch.nn.functional as F


# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        # Save the input to add later
        identity = x
        
        # First layer
        x = F.relu(self.fc1(x))
        
        # Second layer
        x = self.fc2(x)
        
        # Skip connection (add the original input to the output)
        x += identity
        
        # Apply ReLU after adding
        x = F.relu(x)
        
        return x