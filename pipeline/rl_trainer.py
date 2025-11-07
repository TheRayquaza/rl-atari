import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from models import QLearningAgent, QLearningAgentEpsScheduling, SarsaAgent

log = logging.getLogger(__name__)

class RLTrainer:
    """Handles training, evaluation, and logging of RL agents"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.video_dir = self.output_dir / "videos"
        self.plot_dir = self.output_dir / "plots"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        np.random.seed(cfg.experiment.seed)
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Training metrics
        self.metrics = {
            'episode': [],
            'reward': [],
            'length': [],
            'avg_reward_100': [],
            'epsilon': []
        }
        
    def _create_agent(self):
        """Create agent based on configuration"""
        print(self.cfg.agent)
        agent_type = self.cfg.agent.type
        n_actions = self.cfg.env.n_actions
        
        agent_classes = {
            'qlearning': QLearningAgent,
            'qlearning_eps_scheduling': QLearningAgentEpsScheduling,
            'sarsa': SarsaAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = agent_classes[agent_type]
        agent = agent_class(
            learning_rate=self.cfg.agent.learning_rate,
            epsilon=self.cfg.agent.epsilon,
            gamma=self.cfg.agent.gamma,
            legal_actions=list(range(n_actions))
        )
        
        log.info(f"Created {agent_type} agent with params: "
                 f"lr={self.cfg.agent.learning_rate}, "
                 f"eps={self.cfg.agent.epsilon}, "
                 f"gamma={self.cfg.agent.gamma}")
        
        return agent
    
    def _create_env(self, mode='train'):
        """Create environment with appropriate wrappers"""
        env_id = self.cfg.env.env_id
        
        if mode == 'train':
            env = gym.make(env_id, render_mode=None)
            env = RecordEpisodeStatistics(
                env, 
                buffer_length=self.cfg.training.episodes
            )
        else:  # evaluation mode
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(
                env,
                video_folder=str(self.video_dir),
                name_prefix=f"eval_{mode}",
                episode_trigger=lambda ep: ep % self.cfg.evaluation.video_frequency == 0,
            )
            env = RecordEpisodeStatistics(env, buffer_length=100)
        
        return env
    
    def play_episode(self, env: gym.Env, train=True, max_steps=None):
        """Play a single episode"""
        if max_steps is None:
            max_steps = self.cfg.training.max_steps_per_episode
        
        total_reward = 0.0
        steps = 0
        s, _ = env.reset()
        
        for _ in range(max_steps):
            if train:
                a = self.agent.get_action(s)
            else:
                # Greedy action for evaluation
                a = self.agent.get_best_action(s) if hasattr(self.agent, 'get_best_action') else self.agent.get_action(s)
            
            next_s, r, done, truncated, _ = env.step(a)
            
            if train:
                self.agent.update(s, a, r, next_s)
            
            total_reward += r
            steps += 1
            s = next_s
            
            if done or truncated:
                break
        
        return total_reward, steps
    
    def train(self):
        """Main training loop"""
        log.info(f"Starting training for {self.cfg.training.episodes} episodes")
        
        env = self._create_env(mode='train')
        
        for episode in tqdm(range(self.cfg.training.episodes), desc="Training"):
            reward, length = self.play_episode(env, train=True)
            
            # Store metrics
            self.metrics['episode'].append(episode)
            self.metrics['reward'].append(reward)
            self.metrics['length'].append(length)
            
            # Calculate running average
            if episode >= 99:
                avg_reward = np.mean(self.metrics['reward'][-100:])
            else:
                avg_reward = np.mean(self.metrics['reward'])
            
            self.metrics['avg_reward_100'].append(avg_reward)
            
            # Store epsilon if available
            if hasattr(self.agent, 'epsilon'):
                self.metrics['epsilon'].append(self.agent.epsilon)
            else:
                self.metrics['epsilon'].append(0.0)
            
            # Logging
            if (episode + 1) % self.cfg.training.log_frequency == 0:
                log.info(
                    f"Episode {episode+1}/{self.cfg.training.episodes} | "
                    f"Reward: {reward:.2f} | "
                    f"Avg(100): {avg_reward:.2f} | "
                    f"Epsilon: {self.metrics['epsilon'][-1]:.3f}"
                )
            
            # Save checkpoint
            if (episode + 1) % self.cfg.training.checkpoint_frequency == 0:
                self.save_checkpoint(episode + 1)
        
        env.close()
        log.info("Training completed!")
        
        # Save final checkpoint and metrics
        self.save_checkpoint('final')
        self.save_metrics()
        
    def evaluate(self, n_episodes=None):
        """Evaluate the agent"""
        if n_episodes is None:
            n_episodes = self.cfg.evaluation.episodes
        
        log.info(f"Evaluating agent for {n_episodes} episodes")
        
        env = self._create_env(mode='eval')
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in tqdm(range(n_episodes), desc="Evaluating"):
            reward, length = self.play_episode(env, train=False)
            eval_rewards.append(reward)
            eval_lengths.append(length)
        
        env.close()
        
        # Calculate statistics
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths)
        }
        
        log.info(f"Evaluation results: {eval_stats}")
        
        # Save evaluation results
        eval_df = pd.DataFrame({
            'episode': range(n_episodes),
            'reward': eval_rewards,
            'length': eval_lengths
        })
        eval_df.to_csv(self.output_dir / 'evaluation_results.csv', index=False)
        
        with open(self.output_dir / 'evaluation_stats.json', 'w') as f:
            json.dump(eval_stats, f, indent=2)
        
        return eval_stats
    
    def save_checkpoint(self, episode):
        """Save agent checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pkl"
        
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.__dict__,
            'metrics': self.metrics,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        log.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load agent checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore agent state
        self.agent.__dict__.update(checkpoint['agent_state'])
        self.metrics = checkpoint['metrics']
        
        log.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['episode']
    
    def save_metrics(self):
        """Save training metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        metrics_path = self.output_dir / 'training_metrics.csv'
        df.to_csv(metrics_path, index=False)
        log.info(f"Saved metrics to {metrics_path}")
    
    def create_plots(self):
        """Create and save training plots"""
        log.info("Creating plots...")
        
        df = pd.DataFrame(self.metrics)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Rewards plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
        ax.plot(df['episode'], df['avg_reward_100'], linewidth=2, label='Avg Reward (100 eps)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'rewards.png', dpi=150)
        plt.close()
        
        # 2. Episode length plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['episode'], df['length'], alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'episode_lengths.png', dpi=150)
        plt.close()
        
        # 3. Epsilon decay (if applicable)
        if np.any(df['epsilon']):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['episode'], df['epsilon'])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.set_title('Epsilon Decay')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'epsilon_decay.png', dpi=150)
            plt.close()
        
        # 4. Combined plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3)
        axes[0, 0].plot(df['episode'], df['avg_reward_100'], linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(df['episode'], df['length'], alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon
        if np.any(df['epsilon']):
            axes[1, 0].plot(df['episode'], df['epsilon'])
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Epsilon Decay')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 1].hist(df['reward'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'training_summary.png', dpi=150)
        plt.close()
        
        log.info(f"Saved plots to {self.plot_dir}")

