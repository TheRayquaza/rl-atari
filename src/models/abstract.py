from abc import ABC, abstractmethod
import numpy as np
import torch
from logging import getLogger

import torch.nn as nn
import typing as t

Action = int # discrete action space
State = np.ndarray # 3D state (210, 160, 3)
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = torch.Tensor

class AbstractAtariModel(ABC, nn.Module):
    """Abstract base class for Atari Q-learning models."""

    def __init__(self, input_shape, n_actions, **kwargs):
        super(AbstractAtariModel, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.logger = getLogger(self.__class__.__name__)

    @abstractmethod
    def forward(self, x: State) -> QValues:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action (batch_size, n_actions)
        """
        pass
    
    @abstractmethod
    def get_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            epsilon: Exploration rate
            
        Returns:
            Selected action index
        """
        pass
    
    def preprocess_state(self, state):
        """Preprocess state for model input.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Preprocessed state tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        return state
    
    @abstractmethod
    def save_model(self, filepath):
        """Save model parameters."""
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        """Load model parameters."""
        pass
