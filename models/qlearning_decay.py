import random
import torch
from .qlearning import QLearningAgent, State, Action


class QLearningAgentEpsScheduling(QLearningAgent):
    def __init__(
        self,
        *args,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        **kwargs,
    ):
        """
        Q-Learning Agent with epsilon scheduling

        You should not use directly self._qvalues, but instead its getter/setter.
        """
        super().__init__(*args, **kwargs)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.timestep = 0

    def reset(self):
        """
        Reset epsilon to the start value.
        """
        self.epsilon = self.epsilon_start
        self.timestep = 0

    def get_action(self, state: State, epsilon: float = None) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration follows epsilon-greedy policy with epsilon decaying over time.
        """
        if epsilon is None:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                * (self.timestep / self.epsilon_decay_steps),
            )

        if random.random() < self.epsilon:
            action = random.choice(self.legal_actions)  # explore
        else:
            action = self.get_best_action(state)  # exploit

        self.timestep += 1

        return action

    def forward(self, state: State) -> torch.Tensor:
        """
        Forward pass: Returns a tensor of Q-values for all legal actions in the given state.

        Args:
            state (State): The current state.

        Returns:
            torch.Tensor: A tensor of Q-values for all legal actions.
        """
        q_values = [self.get_qvalue(state, action) for action in self.legal_actions]
        return torch.tensor(q_values, dtype=torch.float32)
