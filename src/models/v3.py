import torch.nn as nn
from .base import QNetModel
import torch


class QNetModelV3(QNetModel):
    def __init__(self, n_actions: int, initialization: str, stack_frames: int = 1):
        super().__init__(n_actions, initialization, stack_frames)
        self.legal_actions = list(range(n_actions))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._get_network(n_actions, stack_frames=stack_frames)
        self.q_network.apply(self._get_initializer(initialization))

    def _get_network(self, n_actions: int, stack_frames: int) -> nn.Sequential:
        return nn.Sequential(
            # Input: (B, 4, 84, 84)
            nn.Conv2d(stack_frames, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(0.1),
            # Shape: (B, 32, 20, 20)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.1),
            # Shape: (B, 64, 9, 9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1),
            # Shape: (B, 64, 7, 7)
            nn.Flatten(),
            # 64 * 7 * 7 = 3136
            nn.Linear(2304, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, n_actions),
        ).to(self.device)
