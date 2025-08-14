import random
from collections import deque
from typing import List, Tuple
import numpy as np
from src.utils.data_augmentation import random_augment_data

SampleType = Tuple[np.ndarray, np.ndarray, float]  # (board, policy, value)

class ReplayBuffer:
    def __init__(self, max_size: int = 40_000):
        self.buffer: deque[SampleType] = deque(maxlen=max_size)
    
    def add(self, examples: List[SampleType]):
        self.buffer.extend(examples)
    
    def sample(self, num_samples: int, augment_prob: float = 0.75) -> List[SampleType]:
        batch = random.sample(self.buffer, min(num_samples, len(self.buffer)))
        batch = random_augment_data(batch, augment_prob)
        random.shuffle(batch)
        return batch
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()