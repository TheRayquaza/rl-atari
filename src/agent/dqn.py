import typing as t
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .buffer import ReplayBuffer
from models.base import QNetModel

from torchinfo import summary
from omegaconf import DictConfig

Action = int # discrete action space
State = torch.Tensor # 3D state (210, 160, 3)
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = torch.Tensor

class DQNAgent:
    def __init__(
        self,
        cfg: DictConfig,
        legal_actions: t.List[Action]
    ):
        self.cfg = cfg
        self.legal_actions = legal_actions
        self.n_actions = len(legal_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = cfg.agent.gamma
        self.batch_size = cfg.agent.get('batch_size', 32)
        self.target_update_freq = cfg.agent.get('target_update_freq', 1000)
        self.epsilon = cfg.agent.epsilon
        
        # 1. Initialize Networks
        self.policy_net = QNetModel(self.n_actions, cfg.agent.initialization).to(self.device)
        self.target_net = QNetModel(self.n_actions, cfg.agent.initialization).to(self.device)
        
        # 2. Sync Target Net initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 3. Optimizer
        self.optimizer = self._get_optimizer(cfg.optimizer)
        self.loss_fn = self._get_loss(cfg.agent.loss)
        
        # 4. Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=cfg.agent.get('buffer_size', 10000), 
            state_shape=(1, 80, 80), 
            device=self.device
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir=cfg.experiment.paths.tensorboard)
        self.writer.add_text('Agent/Type', 'DQN Agent')
        self.writer.add_text('Agent/Network', str(self.policy_net))
        
        hyperparams = {
            'Epsilon': self.epsilon,
            'Gamma': self.gamma,
            'Batch Size': self.batch_size,
            'Target Update Freq': self.target_update_freq,
            'Learning Rate': cfg.optimizer.learning_rate,
            'Optimizer': cfg.optimizer.optimizer,
            'Loss Function': cfg.agent.loss,
        }
        self.writer.add_hparams(hyperparams, {})

        self.step_count = 0
        self.episode_rewards: t.List[float] = []
        self.episode_count = 0

    def __str__(self):
        return summary(self.policy_net, input_size=(1, 1, 80, 80)).__str__()

    def __del__(self):
        self.writer.close()
        
    ############### Action Selection ###############

    def get_action(self, state: State, epsilon: float | None = None) -> Action:
            if epsilon is None:
                epsilon = self.epsilon

            if torch.rand(1).item() < epsilon:
                return self.legal_actions[torch.randint(0, self.n_actions, (1,)).item()]
            else:
                state_tensor = self._preprocess(state)

                if state_tensor.ndim == 3:
                    state_tensor = state_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    self.policy_net.eval()
                    q_values = self.policy_net(state_tensor)
                    self.policy_net.train()

                    action_idx = q_values.argmax().item()
                    return self.legal_actions[action_idx]
    
    ############# Preprocessing ###############

    def _preprocess(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if state.dim() == 4 and state.shape[1] == 1 and state.shape[2] == 80 and state.shape[3] == 80:
            return state
        if state.dim() == 4 and state.shape[0] == 1 and state.shape[1] == 1 and state.shape[2] == 80 and state.shape[3] == 80:
            return state
        original_dim = state.dim()
        if original_dim == 3:
            state = state.unsqueeze(0)  # (210, 160, 3) -> (1, 210, 160, 3)
        
        state = state[:, :, :, 0] * 0.299 + state[:, :, :, 1] * 0.587 + state[:, :, :, 2] * 0.114
        state = state[:, 32:192, :]
        state = state.unsqueeze(1)
        state = torch.nn.functional.interpolate(
            state, 
            size=(80, 80), 
            mode='area'
        )
        state = state / 255.0
        state = (state - 0.5) / 0.5  # [-1, 1]

        return state
    
    ############# Optimizer & Loss ###############
    
    def _get_optimizer(self, cfg: DictConfig) -> torch.optim.Optimizer:
        optimizer_type = cfg.optimizer.lower()
        lr = cfg.learning_rate

        if optimizer_type == 'adam':
            return torch.optim.Adam(
                self.policy_net.parameters(),
                lr=lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.policy_net.parameters(),
                lr=lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.policy_net.parameters(),
                lr=lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov
            )
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(
                self.policy_net.parameters(),
                lr=lr,
                alpha=cfg.alpha,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _get_loss(self, loss_name: str) -> nn.Module:
        match loss_name.lower():
            case 'mse':
                return nn.MSELoss()
            case 'huber' | 'smooth_l1':
                return nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Unknown loss function: {loss_name}")

    ############### Training ###############

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool = False):
        proc_state = self._preprocess(state)
        proc_next = self._preprocess(next_state)

        if proc_state.dim() == 4:
            proc_state = proc_state.squeeze(0)
        if proc_next.dim() == 4:
            proc_next = proc_next.squeeze(0)

        self.buffer.push(proc_state, self.legal_actions.index(action), reward, proc_next, done)

        if len(self.buffer) > self.batch_size:
            self._train_step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_count += 1

    def _train_step(self):
        # Sample Batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Policy Net
        current_q = self.policy_net(states).gather(1, actions)

        # Target Net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute Loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.step_count % 100 == 0:
            self.writer.add_scalar('Training/Loss', loss.item(), self.step_count)
            
            self.writer.add_scalar('Training/Q_Value_Mean', current_q.mean().item(), self.step_count)
            self.writer.add_scalar('Training/Target_Q_Mean', target_q.mean().item(), self.step_count)
            
            td_error = (current_q - target_q).abs().mean().item()
            self.writer.add_scalar('Training/TD_Error_Mean', td_error, self.step_count)
            
            self.writer.add_scalar('Training/Gradient_Norm', grad_norm.item(), self.step_count)


    ############### Model Persistence ###############

    def save_model(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.policy_net.to(self.device)

    ############### Logging ###############
    def log_episode(self, total_reward: float, episode_length: int, epsilon: float = None):
        self.episode_rewards.append(total_reward)
        self.episode_count += 1
        
        self.writer.add_scalar('Episode/Reward', total_reward, self.episode_count)
        self.writer.add_scalar('Episode/Length', episode_length, self.episode_count)
        
        if epsilon is not None:
            self.writer.add_scalar('Episode/Epsilon', epsilon, self.episode_count)
        
        if len(self.episode_rewards) >= 100:
            avg_reward = sum(self.episode_rewards[-100:]) / 100
            self.writer.add_scalar('Episode/Avg_Reward_100', avg_reward, self.episode_count)

    def log_network_stats(self):
        for name, param in self.policy_net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data, self.step_count)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, self.step_count)
