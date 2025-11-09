import logging
from pathlib import Path
import shimmy # do not remove
import ale_py
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
from models import DQNAgent, AbstractAtariModel, DQNDecayAgent, DQNDecayDeepMindAgent, DQNReplayAgent

log = logging.getLogger(__name__)

class RLTrainer:
    """Handles training, evaluation, and logging of RL agents"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.video_dir = self.output_dir / "videos"
        self.training_video_dir = self.output_dir / "videos" / "training"
        self.eval_video_dir = self.output_dir / "videos" / "evaluation"
        self.plot_dir = self.output_dir / "plots"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.training_video_dir.mkdir(parents=True, exist_ok=True)
        self.eval_video_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        np.random.seed(cfg.experiment.seed)
        
        # Training metrics
        self.metrics: dict = {
            'episode': [],
            'reward': [],
            'length': [],
            'avg_reward_100': [],
            'epsilon': []
        }
        
    def _create_agent(self, steps) -> AbstractAtariModel:
        """Create agent based on configuration"""
        agent_type = self.cfg.agent.type
        n_actions = self.cfg.env.n_actions

        agent_classes: dict[str, type[AbstractAtariModel]] = {
            'deep_qlearning': DQNAgent,
            'deep_qlearning_decay': DQNDecayAgent,
            'deep_qlearning_decay_v2': DQNDecayDeepMindAgent,
            'deep_qlearning_replay': DQNReplayAgent
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = agent_classes[agent_type]
        agent = agent_class(
            cfg=self.cfg.agent,
            n_actions=n_actions,
            input_shape=(210, 160, 3),
            total_steps=steps,
            legal_actions=list(range(n_actions))
        )

        log.info(str(agent))
        
        return agent
    
    def _create_env(self, mode='train'):
        """Create environment with appropriate wrappers"""
        env_id = self.cfg.env.env_id
        
        if mode == 'train':
            # Check if video recording is enabled for training
            if self.cfg.training.get('record_video', False):
                env = gym.make(env_id, render_mode="rgb_array")
                video_freq = self.cfg.training.get('video_frequency', 100)
                env = RecordVideo(
                    env,
                    video_folder=str(self.training_video_dir),
                    name_prefix="train",
                    episode_trigger=lambda ep: ep % video_freq == 0,
                    disable_logger=True
                )
            else:
                env = gym.make(env_id, render_mode=None)
            
            env = RecordEpisodeStatistics(
                env, 
                buffer_length=self.cfg.training.episodes
            )
        else:  # evaluation mode
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(
                env,
                video_folder=str(self.eval_video_dir),
                name_prefix=f"eval_{mode}",
                episode_trigger=lambda ep: ep % self.cfg.evaluation.get('video_frequency', 1) == 0,
                disable_logger=True
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
                a = self.agent.get_action(s, epsilon=0.0)

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
        log.info(f"Starting training for {self.cfg.training.episodes} episodes")
        
        if self.cfg.training.get('record_video', False):
            log.info(f"Video recording enabled - saving every {self.cfg.training.get('video_frequency', 100)} episodes")

        env = self._create_env(mode='train')
        self.agent = self._create_agent(steps=self.cfg.training.episodes * self.cfg.training.max_steps_per_episode)
        
        pbar = tqdm(range(self.cfg.training.episodes), desc="Training")
        for episode in pbar:
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
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'reward': f'{reward:.1f}',
                'avg_100': f'{avg_reward:.1f}',
                'eps': f'{self.metrics["epsilon"][-1]:.3f}'
            })
            
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
                # Create plots at checkpoint time
                self.create_plots()
        
        env.close()
        log.info("Training completed!")
        
        # Save final checkpoint and metrics
        self.save_checkpoint('final')
        self.save_metrics()
        self.create_plots()
        
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
            'mean_length': np.mean(eval_lengths),
            'median_reward': np.median(eval_rewards)
        }
        
        log.info(
            f"Evaluation Results:\n"
            f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n"
            f"  Min/Max: {eval_stats['min_reward']:.2f} / {eval_stats['max_reward']:.2f}\n"
            f"  Median: {eval_stats['median_reward']:.2f}\n"
            f"  Mean Length: {eval_stats['mean_length']:.2f}"
        )
        
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
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
        
        # Save model using the agent's save_model method
        self.agent.save_model(str(checkpoint_path))
        
        # Save additional training state
        training_state = {
            'episode': episode,
            'metrics': self.metrics,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        training_state_path = self.checkpoint_dir / f"training_state_{episode}.pkl"
        with open(training_state_path, 'wb') as f:
            pickle.dump(training_state, f)
        
        log.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, training_state_path=None):
        """Load agent checkpoint"""
        # Load model
        self.agent.load_model(checkpoint_path)
        
        # Load training state if provided
        if training_state_path and Path(training_state_path).exists():
            with open(training_state_path, 'rb') as f:
                training_state = pickle.load(f)
            self.metrics = training_state['metrics']
            episode = training_state['episode']
            log.info(f"Loaded checkpoint from episode {episode}")
            return episode
        else:
            log.info(f"Loaded model from {checkpoint_path}")
            return 0
    
    def save_metrics(self):
        """Save training metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        metrics_path = self.output_dir / 'training_metrics.csv'
        df.to_csv(metrics_path, index=False)
        log.info(f"Saved metrics to {metrics_path}")
    
    def create_plots(self):
        """Create and save training plots"""
        if not self.metrics['episode']:
            log.warning("No metrics to plot")
            return
            
        log.info("Creating plots...")
        
        df = pd.DataFrame(self.metrics)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Rewards plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward', color='tab:blue')
        ax.plot(df['episode'], df['avg_reward_100'], linewidth=2, label='Avg Reward (100 eps)', color='tab:orange')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Rewards Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'rewards.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Episode length plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['episode'], df['length'], alpha=0.5, color='tab:green')
        # Add rolling average
        if len(df) > 50:
            rolling_avg = df['length'].rolling(window=50, min_periods=1).mean()
            ax.plot(df['episode'], rolling_avg, linewidth=2, color='darkgreen', label='Avg (50 eps)')
            ax.legend(fontsize=10)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length (steps)', fontsize=12)
        ax.set_title('Episode Lengths Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'episode_lengths.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Epsilon decay (if applicable)
        if np.any(df['epsilon']):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['episode'], df['epsilon'], color='tab:red', linewidth=2)
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Epsilon', fontsize=12)
            ax.set_title('Epsilon Decay Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'epsilon_decay.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Combined plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Rewards
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, color='tab:blue')
        axes[0, 0].plot(df['episode'], df['avg_reward_100'], linewidth=2, color='tab:orange')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Rewards', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['Episode', 'Avg (100)'], fontsize=8)
        
        # Episode lengths
        axes[0, 1].plot(df['episode'], df['length'], alpha=0.5, color='tab:green')
        if len(df) > 50:
            rolling_avg = df['length'].rolling(window=50, min_periods=1).mean()
            axes[0, 1].plot(df['episode'], rolling_avg, linewidth=2, color='darkgreen')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon
        if np.any(df['epsilon']):
            axes[1, 0].plot(df['episode'], df['epsilon'], color='tab:red', linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Epsilon Decay', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No epsilon tracking', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Epsilon Decay', fontweight='bold')
        
        # Reward distribution
        axes[1, 1].hist(df['reward'], bins=50, alpha=0.7, edgecolor='black', color='tab:purple')
        axes[1, 1].axvline(df['reward'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {df['reward'].mean():.2f}")
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].legend()
        
        plt.suptitle('Training Summary', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'training_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved plots to {self.plot_dir}")
