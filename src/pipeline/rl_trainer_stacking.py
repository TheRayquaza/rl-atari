import logging
import ale_py  # noqa: F401
import shimmy  # noqa: F401
from omegaconf import DictConfig
from gymnasium.wrappers import FrameStackObservation, MaxAndSkipObservation

from pipeline import RLTrainer
from agent import DQNStackingAgent, DQNDecayAgent, DQNAgent


class RLTrainerStacking(RLTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.log = logging.getLogger(__name__)

        self.stack_frames = cfg.env.get("stack_frames", 1)
        self.skip_frames = cfg.env.get("skip_frames", 1)

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
        if self.stack_frames < 1:
            raise ValueError(f"stack_frames must be >= 1, got {self.stack_frames}")
        if self.stack_frames > 1 and agent_type != "deep_qlearning_stacking":
            raise ValueError(
                f"stack_frames > 1 requires agent type 'deep_qlearning_stacking', got '{agent_type}'"
            )

        agent_class = agent_classes[agent_type]
        agent = agent_class(cfg=self.cfg, legal_actions=list(range(n_actions)))

        self.log.info(str(agent))

        return agent

    def _create_env(self, mode="train"):
        env = super()._create_env(mode)
        env = FrameStackObservation(env, stack_size=self.stack_frames)
        if self.skip_frames > 1:
            env = MaxAndSkipObservation(env, skip=self.skip_frames)
        return env
