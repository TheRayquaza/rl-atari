import typing as t
import torch
from collections import deque
import random
from models.abstract import Action, State
from models.dqn import DQNAgent


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity: int, batch_size: int):
        self.buffer: deque = deque(maxlen=capacity)
        self.batch_size = batch_size
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self):
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= self.batch_size


class DQNReplayAgent(DQNAgent):
    """DQN Agent with Experience Replay and Target Network"""
    
    def __init__(
        self,
        input_shape,
        n_actions,
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
        
        # Replay buffer parameters
        self.batch_size = cfg.get('batch_size', 32)
        self.update_frequency = cfg.get('update_frequency', 4)
        self.target_update_frequency = cfg.get('target_update_frequency', 10000)
        
        # Experience replay
        replay_capacity = cfg.get('replay_capacity', 100000)
        self.replay_buffer = ReplayBuffer(replay_capacity, self.batch_size)
        
        # Target network
        self.target_network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(legal_actions)),
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Track updates
        self.update_counter = 0

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool = False):
        """Store transition in replay buffer and train if ready"""
        # Store transition
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only train if buffer is ready and it's time to update
        if not self.replay_buffer.is_ready():
            return
        
        if self.update_counter % self.update_frequency != 0:
            self.update_counter += 1
            return
        
        self.update_counter += 1
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Preprocess batch
        state_batch = torch.cat([self._preprocess_state(s) for s in states])
        next_state_batch = torch.cat([self._preprocess_state(s) for s in next_states])
        action_indices = torch.tensor([self.legal_actions.index(int(a)) for a in actions], 
                                     dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute current Q values
        self.q_network.train()
        current_q_values = self.q_network(state_batch)
        current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)
        
        # Compute loss and optimize
        loss = torch.nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
        """Save model with replay agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'update_counter': self.update_counter,
        }, filepath)

    def load_model(self, filepath: str):
        """Load model with replay agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.steps = checkpoint.get('steps', 0)
        self.update_counter = checkpoint.get('update_counter', 0)
        self.target_network.to(self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_network.to(self.device)
