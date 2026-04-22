import torch


class DrQv2Noise:
    """
    DrQv2-style decaying Gaussian action noise.

    Noise standard deviation linearly decays from initial_std to final_std
    over decay_steps samples.
    """

    def __init__(self,
                 action_dim: int,
                 device: torch.device,
                 initial_std: float = 0.3,
                 final_std: float = 0.05,
                 decay_steps: int = 200000):
        self.action_dim = int(action_dim)
        self.device = device
        self.initial_std = float(initial_std)
        self.final_std = float(final_std)
        self.decay_steps = max(1, int(decay_steps))
        self._step = 0

    def _current_std(self) -> float:
        t = min(1.0, self._step / float(self.decay_steps))
        return (1.0 - t) * self.initial_std + t * self.final_std

    def sample(self, size):
        """
        skrl-compatible noise API: sample by tensor shape.
        """
        std = self._current_std()
        self._step += 1
        return torch.randn(size, device=self.device) * std


class FixedGaussianNoise:
    """
    Fixed-std Gaussian noise (utility for TD3 target policy smoothing).
    """

    def __init__(self, device: torch.device, std: float = 0.2, mean: float = 0.0):
        self.device = device
        self.std = float(std)
        self.mean = float(mean)

    def sample(self, size):
        return torch.randn(size, device=self.device) * self.std + self.mean
