import typing as t
import torch
from .abstract import Action
from .dqn_decay import DQNDecayAgent

class DQNDecayDeepMindAgent(DQNDecayAgent):
    def __init__(
        self,
        cfg: dict,
        legal_actions: t.List[Action],
        total_steps: int,
        input_shape: tuple = (210, 160, 3),
    ):
        super().__init__(input_shape=input_shape,
                            cfg=cfg, legal_actions=legal_actions, total_steps=total_steps)

        # Conv1: (80-5)/2 + 1 = 38
        # Conv2: (38-3)/2 + 1 = 18
        # Conv3: (18-3)/2 + 1 = 8
        # Conv4: (8-3)/1 + 1 = 6
        # Flattened: 128 * 6 * 6 = 4608
        self.q_network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(legal_actions)),
        ).to(self.device)
