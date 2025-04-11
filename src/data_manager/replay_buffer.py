import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=500_000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples):
        self.buffer.extend(examples)
    
    def sample(self, num_samples):
        # Gibt alle Beispiele zurück, wenn weniger als num_samples vorhanden sind
        return random.sample(self.buffer, min(num_samples, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)