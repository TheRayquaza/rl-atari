import typing as t
import numpy as np
import torch
from models.abstract import AbstractAtariModel, Action, State
from torchinfo import summary


class DQNAgent(AbstractAtariModel):
    """Base DQN Agent with standard architecture and preprocessing"""
    
    def __init__(
        self,
        input_shape,
        n_actions,
        cfg: dict,
        legal_actions: t.List[Action],
        total_steps: int,
    ):
        super().__init__(input_shape=input_shape, n_actions=n_actions, cfg=cfg, total_steps=total_steps)
        self.legal_actions = legal_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core hyperparameters
        self.gamma = cfg['gamma']
        self.epsilon = cfg.get('epsilon', 0.1)
        
        # Initialize network
        self._build_network(len(legal_actions))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=cfg['learning_rate']
        )

    def _build_network(self, n_actions: int):
        """Build the Q-network architecture"""
        # DQN architecture for 80x80 input
        # Conv1: (80-8)/4 + 1 = 19
        # Conv2: (19-4)/2 + 1 = 8
        # Flattened: 32 * 8 * 8 = 2048
        self.q_network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        ).to(self.device)

    def __str__(self):
        return summary(
            self.q_network, input_size=(1, 1, 80, 80)
        ).__str__()

    def _preprocess_state(self, state):
        """Convert RGB Atari frame to 80x80 grayscale tensor"""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            
            # Convert RGB to grayscale
            if state.ndim == 3 and state.shape[2] == 3:
                state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
            
            # Crop to remove score area (160x160)
            state = state[32:192, :]
            
            # Add batch and channel dimensions
            state = state.unsqueeze(0).unsqueeze(0)
            
            # Resize to 80x80
            state = torch.nn.functional.interpolate(
                state, 
                size=(80, 80), 
                mode='area'
            )
            
            # Normalize to [0, 1]
            state = state / 255.0
            
            return state.to(self.device)

    def get_qvalue(self, state: State, action: Action) -> float:
        """Get Q-value for a specific state-action pair"""
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            q_values = self.q_network(state_tensor)
            action_idx = self.legal_actions.index(action)
            return q_values[0, action_idx].item()

    def get_value(self, state: State) -> float:
        """Get the maximum Q-value for a state"""
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            q_values = self.q_network(state_tensor)
            return q_values.max().item()

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool = False):
        """Update Q-network using single transition"""
        self.q_network.train()
        
        state_tensor = self._preprocess_state(state)
        next_state_tensor = self._preprocess_state(next_state)
        action_idx = self.legal_actions.index(action)
        
        # Compute current Q-value
        current_q = self.q_network(state_tensor)[0, action_idx]

        # Compute target Q-value
        with torch.no_grad():
            next_q = self.q_network(next_state_tensor).max()
            # If episode is done, there's no future reward
            target_q = torch.tensor(reward, device=self.device) + (0.0 if done else self.gamma * next_q)

        # Compute loss and optimize
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, state: State) -> torch.Tensor:
        """Forward pass through Q-network"""
        state_tensor = self._preprocess_state(state)
        return self.q_network(state_tensor)

    def get_action(self, state: State, epsilon: float | None = None) -> Action:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.rand() < epsilon:
            return int(np.random.choice(self.legal_actions))
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.forward(state)
                action_idx = int(torch.argmax(q_values).item())
                return self.legal_actions[action_idx]

    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.q_network.to(self.device)
