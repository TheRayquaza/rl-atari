import typing as t
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm
from torchinfo import summary
from omegaconf import DictConfig

from .buffer import ReplayBuffer
from models import QNetModel, QNetModelV2, QNetModelV3

Action = int  # discrete action space
State = torch.Tensor  # 3D state (210, 160, 3)
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = torch.Tensor


class DQNAgent:
    def __init__(
        self,
        cfg: DictConfig,
        legal_actions: t.List[Action],
        use_preprocessing: bool = True,
    ):
        self.cfg = cfg
        self.legal_actions = legal_actions
        self.n_actions = len(legal_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = cfg.agent.gamma
        self.batch_size = cfg.agent.get("batch_size", 32)
        self.target_update_freq = cfg.agent.get("target_update_freq", 1000)
        self.epsilon = cfg.agent.epsilon
        self.buffer_size = cfg.agent.get("buffer_size", 10000)
        self.use_preprocessing = use_preprocessing
        self.reward_clipping = cfg.agent.get("reward_clipping", False)

        # Initialize Networks
        if cfg.agent.model == "v2":
            self.policy_net = QNetModelV2(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1
            self.target_net = QNetModelV2(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1
        elif cfg.agent.model == "v3":
            self.policy_net = QNetModelV3(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1
            self.target_net = QNetModelV3(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1
        else:
            self.policy_net = QNetModel(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1
            self.target_net = QNetModel(self.n_actions, cfg.agent.initialization).to(
                self.device
            )  # stack_frames=1

        # Sync Target Net initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = self._get_optimizer(cfg.optimizer)
        self.loss_fn = self._get_loss(cfg.agent.loss)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=self.buffer_size, state_shape=(1, 80, 80), device=self.device
        )

        # Logging
        self.writer = SummaryWriter(log_dir=cfg.experiment.paths.tensorboard)
        self.metrics: dict[str, list[float]] = {
            "loss": [],
            "q_values": [],
            "episode_rewards": [],
            "episode_lengths": [],
        }
        self.writer.add_text("Agent/Type", "DQN Agent")
        self.writer.add_text("Agent/Network", str(self.policy_net))

        hyperparams = {
            "Epsilon": self.epsilon,
            "Gamma": self.gamma,
            "Optimizer": cfg.optimizer.optimizer,
            "Loss Function": cfg.agent.loss,
            "Initialization": cfg.agent.initialization,
            "Learning Rate": cfg.optimizer.learning_rate,
            "Stacked Frames": 1,
            "Batch Size": self.batch_size,
            "Buffer Size": self.buffer_size,
            "Target Update Freq": self.target_update_freq,
            "Reward Clipping": self.reward_clipping,
        }
        self.writer.add_hparams(hyperparams, {})

        self.step_count = 0
        self.episode_rewards: t.List[float] = []
        self.episode_count = 0

    def __str__(self):
        return summary(self.policy_net, input_size=(1, 1, 80, 80), verbose=0).__str__()

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

    def _preprocess(self, state: State) -> torch.Tensor:
        if not self.use_preprocessing:
            return state

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        else:
            state = state.to(self.device, dtype=torch.float32)

        if state.max() > 10.0:
            state = state / 255.0
        
        if state.dim() == 2:
            state = state.unsqueeze(0).unsqueeze(0)

        elif state.dim() == 3:
            if state.shape[-1] <= 4:
                state = state.permute(2, 0, 1).unsqueeze(0)
            else:
                state = state.unsqueeze(0)

        elif state.dim() == 4:
            if state.shape[-1] <= 4:
                state = state.permute(0, 3, 1, 2)

        if state.shape[2] == 210:
            state = state[:, :, 34:194, :]

        if state.shape[-1] != 80:
            state = torch.nn.functional.interpolate(state, size=(80, 80), mode="area")

        if state.shape[-1] >= 40: 
            state = torch.nn.functional.max_pool2d(
                state, kernel_size=2, stride=1, padding=1
            )
            state = state[:, :, :80, :80]

        return state

    ############# Optimizer & Loss ###############

    def _get_optimizer(self, cfg: DictConfig) -> torch.optim.Optimizer:
        optimizer_type = cfg.optimizer.lower()
        lr = cfg.learning_rate

        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.policy_net.parameters(),
                lr=lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.policy_net.parameters(),
                lr=lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.policy_net.parameters(),
                lr=lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov,
            )
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                self.policy_net.parameters(),
                lr=lr,
                alpha=cfg.alpha,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _get_loss(self, loss_name: str) -> nn.Module:
        match loss_name.lower():
            case "mse":
                return nn.MSELoss()
            case "huber" | "smooth_l1":
                return nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Unknown loss function: {loss_name}")

    ############### Training ###############

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool = False,
    ):
        proc_state = self._preprocess(state)
        proc_next = self._preprocess(next_state)

        if self.reward_clipping:
            reward = max(-1.0, min(1.0, reward))

        self.buffer.push(
            proc_state, self.legal_actions.index(action), reward, proc_next, done
        )

        if len(self.buffer) > self.batch_size:
            self._train_step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_count += 1

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0
        )
        self.optimizer.step()

        self.metrics["loss"].append(loss.item())
        self.metrics["q_values"].append(current_q.mean().item())

        if self.step_count % 100 == 0:
            self.writer.add_scalar("Training/Loss", loss.item(), self.step_count)
            self.writer.add_scalar(
                "Training/Q_Value_Mean", current_q.mean().item(), self.step_count
            )
            self.writer.add_scalar(
                "Training/Target_Q_Mean", target_q.mean().item(), self.step_count
            )
            td_error = (current_q - target_q).abs().mean().item()
            self.writer.add_scalar("Training/TD_Error_Mean", td_error, self.step_count)
            self.writer.add_scalar(
                "Training/Gradient_Norm", grad_norm.item(), self.step_count
            )

    ############### Model Persistence ###############

    def save_model(self, filepath: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
                "metrics": self.metrics,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(
            checkpoint.get("target_net", checkpoint["policy_net"])
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.step_count = checkpoint.get("step_count", 0)
        self.metrics = checkpoint.get("metrics", self.metrics)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    ############### Logging ###############
    def log_episode(
        self, total_reward: float, episode_length: int, epsilon: float = None
    ):
        self.metrics["episode_rewards"].append(total_reward)
        self.metrics["episode_lengths"].append(episode_length)
        self.episode_count += 1

        self.writer.add_scalar("Episode/Reward", total_reward, self.episode_count)
        self.writer.add_scalar("Episode/Length", episode_length, self.episode_count)

        if epsilon is not None:
            self.writer.add_scalar("Episode/Epsilon", epsilon, self.episode_count)

        if len(self.metrics["episode_rewards"]) >= 100:
            avg_reward = sum(self.metrics["episode_rewards"][-100:]) / 100
            self.writer.add_scalar(
                "Episode/Avg_Reward_100", avg_reward, self.episode_count
            )

    def log_network_stats(self):
        for name, param in self.policy_net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"Weights/{name}", param.data, self.step_count
                )
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Gradients/{name}", param.grad, self.step_count
                    )

    def log_grad_cam(self, state: State, tag: str = "Agent/GradCAM"):

        self.policy_net.eval()
        input_tensor = self._preprocess(state)

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad = True

        target_layer = None
        for module in reversed(list(self.policy_net.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            print("Error: No Conv2d layer found for Grad-CAM.")
            self.policy_net.train()
            return

        feature_maps = []
        gradients = []

        def save_fmaps(module, input, output):
            feature_maps.append(output)

        def save_grads(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = target_layer.register_forward_hook(save_fmaps)
        handle_b = target_layer.register_full_backward_hook(save_grads)

        q_values = self.policy_net(input_tensor)
        best_action_idx = q_values.argmax().item()

        self.policy_net.zero_grad()
        target_score = q_values[0, best_action_idx]
        target_score.backward()

        grads = gradients[0].cpu().data.numpy()[0]  # (C, H, W)
        fmaps = feature_maps[0].cpu().data.numpy()[0]  # (C, H, W)

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmaps[i]

        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_tensor = torch.nn.functional.interpolate(
            cam_tensor, size=(80, 80), mode="bilinear", align_corners=False
        )
        cam_img = cam_tensor.squeeze().numpy()

        orig_img = input_tensor.detach().cpu().numpy()[0]

        if orig_img.ndim == 3:
            orig_img = orig_img[-1]
        elif orig_img.ndim == 2:
            pass
        else:
            print(f"Unexpected orig_img shape: {orig_img.shape}")
            handle_f.remove()
            handle_b.remove()
            self.policy_net.train()
            return

        if orig_img.min() < 0 or orig_img.max() > 1:
            orig_img = (orig_img * 0.5) + 0.5
        orig_img = np.clip(orig_img, 0, 1)

        heatmap_colored = cm.jet(cam_img)[:, :, :3]  # (80, 80, 3) RGB
        orig_img_rgb = np.stack([orig_img] * 3, axis=-1)  # (80, 80, 3)
        superimposed = 0.6 * orig_img_rgb + 0.4 * heatmap_colored
        img_for_tb = np.transpose(superimposed, (2, 0, 1))  # (3, 80, 80)

        self.writer.add_image(tag, img_for_tb, self.step_count)

        handle_f.remove()
        handle_b.remove()
        self.policy_net.train()

    def log_input_state(self, state: State):
        tensor = self._preprocess(state)

        if tensor.shape[1] > 1:
            num_frames = tensor.shape[1]
            frames = [tensor[0, i, :, :] for i in range(num_frames)]
            img_vis = torch.cat(frames, dim=1)
            img_vis = img_vis.unsqueeze(0)
        else:
            img_vis = tensor[0]

        img_vis = torch.clamp(img_vis, 0, 1)
        self.writer.add_image("Debug/Preprocessed_Input", img_vis, self.step_count)
