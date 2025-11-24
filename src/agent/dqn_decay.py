from omegaconf import DictConfig
from .dqn import DQNAgent
import typing as t

from .dqn import Action, State

class DQNDecayAgent(DQNAgent):    
    def __init__(self, cfg: DictConfig, legal_actions: t.List[Action]):
        super().__init__(cfg, legal_actions)
        
        self.epsilon_start = cfg.agent.epsilon_start
        self.epsilon_end = cfg.agent.epsilon_end
        self.epsilon_decay = cfg.agent.epsilon_decay
        self.epsilon = self.epsilon_start
        
        # add hyperparams to tensorboard
        self.writer.add_text('Agent/Decay_Type', 'Linear')
        self.writer.add_scalar('Agent/Epsilon_Start', self.epsilon_start)
        self.writer.add_scalar('Agent/Epsilon_End', self.epsilon_end)
        self.writer.add_scalar('Agent/Epsilon_Decay_Steps', self.epsilon_decay)
    
    def get_action(self, state: State, epsilon: float | None = None) -> Action:
        if epsilon is None:
            progress = min(self.step_count / self.epsilon_decay, 1.0)
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
            
            if self.step_count % 100 == 0:
                self.writer.add_scalar('Agent/Epsilon', self.epsilon, self.step_count)
                
            epsilon = self.epsilon

        return super().get_action(state, epsilon)
