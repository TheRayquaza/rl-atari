import typing as t
import torch
from models.abstract import Action, State
from models.dqn import DQNAgent


class DQNDecayAgent(DQNAgent):
    """DQN Agent with linear epsilon decay"""
    
    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        cfg: dict,
        legal_actions: t.List[Action],
        total_steps: int,
    ):
        # Initialize base class
        super().__init__(input_shape, n_actions, cfg, legal_actions, total_steps)
        
        # Epsilon decay parameters
        self.epsilon = cfg['epsilon_start']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.steps = 0
        self.total_steps = total_steps if total_steps else cfg.get('total_steps', 1000000)

    def get_action(self, state: State, epsilon: float | None = None) -> Action:
        """Select action with linear epsilon decay"""
        if epsilon is None:
            # Linear epsilon decay
            decay_steps = min(self.steps, self.epsilon_decay)
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (decay_steps / self.epsilon_decay)
            epsilon = self.epsilon

        self.steps += 1
        
        # Call parent's get_action with computed epsilon
        return super().get_action(state, epsilon)

    def save_model(self, filepath: str):
        """Save model with epsilon decay state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
        }, filepath)

    def load_model(self, filepath: str):
        """Load model with epsilon decay state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.steps = checkpoint.get('steps', 0)
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
        self.q_network.to(self.device)
