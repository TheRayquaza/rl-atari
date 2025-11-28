import torch
from torch import nn
import math
from torchviz import make_dot

class QNetModel(nn.Module):
    def __init__(self, n_actions: int, initialization: str, stack_frames: int = 1):
        super().__init__()
        self.legal_actions = list(range(n_actions))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._get_network(n_actions, stack_frames=stack_frames)
        self.q_network.apply(self._get_initializer(initialization))

    def _get_initializer(self, method: str):
        def init_layer(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif method == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return init_layer

    def _get_network(self, n_actions: int, stack_frames: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(stack_frames, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_network(x)

    def visualize(self) -> None:
        state = torch.randn(1, 1, 84, 84).to(self.device)

        if state.max() > 10.0:
            state = state / 255.0
        
        if state.dim() == 2:
            state = state.unsqueeze(0).unsqueeze(0)

        elif state.dim() == 3:
            if state.shape[-1] <= 4:
                state = state.permute(2, 0, 1).unsqueeze(0)
            else:
                state = state.unsqueeze(0)

        elif state.dim() == 4:
            if state.shape[-1] <= 4:
                state = state.permute(0, 3, 1, 2)

        if state.shape[2] == 210:
            state = state[:, :, 34:194, :]

        if state.shape[-1] != 80:
            state = torch.nn.functional.interpolate(state, size=(80, 80), mode="area")

        if state.shape[-1] >= 40: 
            state = torch.nn.functional.max_pool2d(
                state, kernel_size=2, stride=1, padding=1
            )
            state = state[:, :, :80, :80]

        yhat = self.q_network(state)
        make_dot(yhat, params=dict(self.q_network.named_parameters())).render("net", format="png")
