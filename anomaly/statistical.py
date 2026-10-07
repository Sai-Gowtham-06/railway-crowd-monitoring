from collections import deque
from typing import Optional
import numpy as np

class RollingBaseline:
    """
    Maintains a rolling temporal window to compute statistical baselines
    (mean, standard deviation, and Z-score) for dynamic anomaly detection.
    """
    def __init__(self, window_size: int = 60, min_samples: int = 15, epsilon: float = 1e-4):
        self.window_size = window_size
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.history: deque = deque(maxlen=window_size)
        self.mean: float = 0.0
        self.std: float = 0.0

    def update(self, value: float) -> Optional[float]:
        """
        Appends new observation, updates stats, and returns Z-score.
        Returns None if not enough samples have been gathered.
        """
        self.history.append(float(value))
        
        if len(self.history) < self.min_samples:
            self.mean = float(np.mean(self.history))
            self.std = float(np.std(self.history))
            return None
            
        # Compute baseline on history excluding current point to avoid biasing
        past_values = list(self.history)[:-1]
        self.mean = float(np.mean(past_values))
        self.std = float(np.std(past_values))
        
        # Calculate Z-score
        z_score = (value - self.mean) / (self.std + self.epsilon)
        return float(z_score)

    def is_warmed_up(self) -> bool:
        return len(self.history) >= self.min_samples
