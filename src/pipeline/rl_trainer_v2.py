import logging
from pathlib import Path
import hydra
import ale_py  # noqa: F401
import shimmy  # noqa: F401
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
    AtariPreprocessing,
    FrameStackObservation,
    NumpyToTorch,
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
from agent import DQNAgent, DQNDecayAgent, DQNStackingAgent


class RLTrainerV2:
    """Handles training, evaluation, and logging of RL agents"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.video_dir = self.output_dir / "videos"
        self.training_video_dir = self.output_dir / "videos" / "training"
        self.eval_video_dir = self.output_dir / "videos" / "evaluation"
        self.plot_dir = self.output_dir / "plots"
        self.tensorboard_dir = Path(cfg.experiment.paths.tensorboard)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log = logging.getLogger(__name__)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.training_video_dir.mkdir(parents=True, exist_ok=True)
        self.eval_video_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(cfg.experiment.seed)

        self.metrics: dict = {
            "episode": [],
            "reward": [],
            "length": [],
            "avg_reward_100": [],
            "epsilon": [],
        }

    def _create_agent(self) -> DQNAgent:
        agent_type = self.cfg.agent.type
        n_actions = self.cfg.env.n_actions

        agent_classes: dict[str, type[DQNAgent]] = {
            "deep_qlearning": DQNAgent,
            "deep_qlearning_decay": DQNDecayAgent,
            "deep_qlearning_stacking": DQNStackingAgent,
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = agent_classes[agent_type]
        agent = agent_class(
            cfg=self.cfg, legal_actions=list(range(n_actions)), use_preprocessing=False
        )  # Use atari preprocessor for raw states

        self.log.info(str(agent))

        return agent

    def _create_env(self, mode="train"):
        env_id = self.cfg.env.env_id

        if mode == "train":
            if self.cfg.training.get("record_video", False):
                env = gym.make(
                    env_id, render_mode="rgb_array", repeat_action_probability=0.0
                )
                video_freq = self.cfg.training.get("video_frequency", 100)
                env = RecordVideo(
                    env,
                    video_folder=str(self.training_video_dir),
                    name_prefix="train",
                    episode_trigger=lambda ep: ep % video_freq == 0,
                    disable_logger=True,
                )
            else:
                env = gym.make(env_id, render_mode=None, repeat_action_probability=0.0)

            env = RecordEpisodeStatistics(env, buffer_length=self.cfg.training.episodes)
        else:
            env = gym.make(
                env_id, render_mode="rgb_array", repeat_action_probability=0.0
            )
            env = RecordVideo(
                env,
                video_folder=str(self.eval_video_dir),
                name_prefix=f"eval_{mode}",
                episode_trigger=lambda ep: ep
                % self.cfg.evaluation.get("video_frequency", 1)
                == 0,
                disable_logger=True,
            )
            env = RecordEpisodeStatistics(env, buffer_length=100)

        env = AtariPreprocessing(
            env,
            screen_size=80,
            frame_skip=1,
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=False,
        )
        env = FrameStackObservation(env, stack_size=self.cfg.env.stack_frames)
        env = NumpyToTorch(env, device=torch.device("cpu"))
        return env

    def play_episode(self, env: gym.Env, train=True, max_steps=None):
        if max_steps is None:
            max_steps = self.cfg.training.max_steps_per_episode

        device = self.device

        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, device=device, dtype=dtype)

        total_reward = 0.0
        steps = 0

        s, _ = env.reset()
        state = to_tensor(s)

        for _ in range(max_steps):
            if train:
                a = self.agent.get_action(state)
            else:
                a = self.agent.get_action(state, epsilon=0.0)

            next_s, r, done, truncated, _ = env.step(a)

            next_state = to_tensor(next_s)
            reward = to_tensor(r)
            is_done = done or truncated
            done_tensor = to_tensor(is_done, dtype=torch.uint8)

            if train:
                self.agent.update(state, a, reward, next_state, done_tensor)

            total_reward += float(r)
            steps += 1

            state = next_state

            if is_done:
                break

        if train:
            self.agent.log_episode(total_reward, steps, self.agent.epsilon)
            if self.agent.episode_count % 25 == 0:
                self.agent.log_grad_cam(state)
                #self.agent.log_input_state(state)
            if (self.agent.step_count % 1000) == 0:
                self.agent.log_network_stats()

        return total_reward, steps

    def train(self):
        self.log.info(f"Starting training for {self.cfg.training.episodes} episodes")

        if self.cfg.training.get("record_video", False):
            self.log.info(
                f"Video recording enabled - saving every {self.cfg.training.get('video_frequency', 100)} episodes"
            )

        env = self._create_env(mode="train")
        self.agent = self._create_agent()

        pbar = tqdm(range(self.cfg.training.episodes), desc="Training")
        for episode in pbar:
            reward, length = self.play_episode(env, train=True)

            self.metrics["episode"].append(episode)
            self.metrics["reward"].append(reward)
            self.metrics["length"].append(length)

            if episode >= 99:
                avg_reward = np.mean(self.metrics["reward"][-100:])
            else:
                avg_reward = np.mean(self.metrics["reward"])

            self.metrics["avg_reward_100"].append(avg_reward)

            if hasattr(self.agent, "epsilon"):
                self.metrics["epsilon"].append(self.agent.epsilon)
            else:
                self.metrics["epsilon"].append(0.0)

            pbar.set_postfix(
                {
                    "reward": f"{reward:.1f}",
                    "avg_100": f"{avg_reward:.1f}",
                    "eps": f'{self.metrics["epsilon"][-1]:.3f}',
                }
            )

            if (episode + 1) % self.cfg.training.log_frequency == 0:
                self.log.info(
                    f"Episode {episode+1}/{self.cfg.training.episodes} | "
                    f"Reward: {reward:.2f} | "
                    f"Avg(100): {avg_reward:.2f} | "
                    f"Epsilon: {self.metrics['epsilon'][-1]:.3f}"
                )

            if (episode + 1) % self.cfg.training.checkpoint_frequency == 0:
                self.save_checkpoint(episode + 1)

        env.close()
        self.log.info("Training completed!")

        self.save_checkpoint("final")

    def evaluate(self, n_episodes=None):
        if n_episodes is None:
            n_episodes = self.cfg.evaluation.episodes

        self.log.info(f"Evaluating agent for {n_episodes} episodes")

        env = self._create_env(mode="eval")

        eval_rewards = []
        eval_lengths = []

        for episode in tqdm(range(n_episodes), desc="Evaluating"):
            reward, length = self.play_episode(env, train=False)
            eval_rewards.append(reward)
            eval_lengths.append(length)

        env.close()

        # Calculate statistics
        eval_stats = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "median_reward": np.median(eval_rewards),
        }

        self.log.info(
            f"Evaluation Results:\n"
            f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n"
            f"  Min/Max: {eval_stats['min_reward']:.2f} / {eval_stats['max_reward']:.2f}\n"
            f"  Median: {eval_stats['median_reward']:.2f}\n"
            f"  Mean Length: {eval_stats['mean_length']:.2f}"
        )

        # Save evaluation results
        eval_df = pd.DataFrame(
            {
                "episode": range(n_episodes),
                "reward": eval_rewards,
                "length": eval_lengths,
            }
        )
        eval_df.to_csv(self.output_dir / "evaluation_results.csv", index=False)

        with open(self.output_dir / "evaluation_stats.json", "w") as f:
            json.dump(eval_stats, f, indent=2)

        return eval_stats

    def save_checkpoint(self, episode):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pt"

        self.agent.save_model(str(checkpoint_path))

        training_state = {
            "episode": episode,
            "metrics": self.metrics,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        training_state_path = self.checkpoint_dir / f"training_state_{episode}.pkl"
        with open(training_state_path, "wb") as f:
            pickle.dump(training_state, f)

        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, training_state_path=None):
        self.agent.load_model(checkpoint_path)

        if training_state_path and Path(training_state_path).exists():
            with open(training_state_path, "rb") as f:
                training_state = pickle.load(f)
            self.metrics = training_state["metrics"]
            episode = training_state["episode"]
            self.log.info(f"Loaded checkpoint from episode {episode}")
            return episode
        else:
            self.log.info(f"Loaded model from {checkpoint_path}")
            return 0
