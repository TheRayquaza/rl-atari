from collections import defaultdict
import random
import typing as t
import numpy as np
import torch
from models.abstract import AbstractAtariModel


Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class QLearningAgent(AbstractAtariModel):
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        Q-Learning Agent

        You should not use directly self._qvalues, but instead its getter/setter.
        """
        super().__init__(input_shape=(), n_actions=len(legal_actions))
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state, action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Q-value for [state, action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute V(s) = max_a Q(s, a) over possible actions.
        """
        if state not in self._qvalues or not self._qvalues[state]:
            return 0.0

        return max(self.get_qvalue(state, a) for a in self.legal_actions)

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here:

           TD_target(s, a, r, s') = r + gamma * V(s')
           TD_error(s, a, r, s') = TD_target(s, a, r, s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s, a, R(s, a), s')
        """
        td_target = reward + self.gamma * self.get_value(next_state)
        td_error = td_target - self.get_qvalue(state, action)
        new_value = self.get_qvalue(state, action) + self.learning_rate * td_error
        self.set_qvalue(state, action, new_value)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current Q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probability, generate uniform number in [0, 1]
              and compare it with your probability.
        """
        if random.random() < self.epsilon:
            return random.choice(self.legal_actions)

        return self.get_best_action(state)

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
