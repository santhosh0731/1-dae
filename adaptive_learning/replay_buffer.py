"""
Adaptive Learning - Replay Buffer
=================================
FIFO buffer storing physical inputs and trajectories for incremental training.
"""

import random
import numpy as np

class ReplayBuffer:
    """FIFO Replay Buffer for storing online telemetry samples."""
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.buffer = []

    def push(self, state: np.ndarray, target: np.ndarray):
        """Adds a new sample pair (state, target) to the buffer."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, target))

    def sample(self, batch_size: int):
        """Samples a random batch of data."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, targets = zip(*batch)
        return np.array(states, dtype=np.float32), np.array(targets, dtype=np.float32)

    def size(self) -> int:
        return len(self.buffer)
