from omegaconf import DictConfig
from torchinfo import summary
import torch.nn as nn
import numpy as np
import torch
import matplotlib.cm as cm
import typing as t

from .dqn import Action, State
from .dqn_decay import DQNDecayAgent

from models import QNetModelV2, QNetModelV3, QNetModel
from .buffer import ReplayBuffer


class DQNStackingAgent(DQNDecayAgent):
    def __init__(
        self,
        cfg: DictConfig,
        legal_actions: t.List[Action],
        use_preprocessing: bool = True,
    ):
        super().__init__(cfg, legal_actions, use_preprocessing)

        self.stack_frames: int = int(cfg.env.get("stack_frames", 1))

        hyperparams = {
            "Stacked Frames": self.stack_frames,
        }
        self.writer.add_hparams(hyperparams, {})

        # Initialize Networks
        if cfg.agent.model == "v2":
            self.policy_net = QNetModelV2(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)
            self.target_net = QNetModelV2(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)
        elif cfg.agent.model == "v3":
            self.policy_net = QNetModelV3(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)
            self.target_net = QNetModelV3(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)
        else:
            self.policy_net = QNetModel(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)
            self.target_net = QNetModel(
                self.n_actions, cfg.agent.initialization, self.stack_frames
            ).to(self.device)

        self.buffer = ReplayBuffer(
            capacity=self.buffer_size,
            state_shape=(self.stack_frames, 80, 80),
            device=self.device,
        )

    def get_action(self, state: State, epsilon: float | None = None) -> Action:
        if epsilon is None:
            progress = min(self.step_count / self.epsilon_decay, 1.0)
            self.epsilon = (
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
            )

            if self.step_count % 100 == 0:
                self.writer.add_scalar("Agent/Epsilon", self.epsilon, self.step_count)

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

    def __str__(self):
        return summary(
            self.policy_net, input_size=(1, self.stack_frames, 80, 80), verbose=0
        ).__str__()

    def log_grad_cam(self, state, tag="Agent/GradCAM_Video"):
        self.policy_net.eval()
        input_tensor = self._preprocess(state)

        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad = True
        
        batch_size, num_frames, img_h, img_w = input_tensor.shape

        target_layer = None
        for module in reversed(list(self.policy_net.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
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

        grads = gradients[0].cpu().data.numpy()[0]  # (C, H_feat, W_feat)
        fmaps = feature_maps[0].cpu().data.numpy()[0]  # (C, H_feat, W_feat)

        weights = np.mean(grads, axis=(1, 2))  # (C,)
        
        cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmaps[i]

        cam = np.maximum(cam, 0) # ReLU

        if cam.max() > 0:
            cam = cam / cam.max() # Normalize

        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_tensor = torch.nn.functional.interpolate(
            cam_tensor, size=(img_h, img_w), mode="bilinear", align_corners=False
        )
        cam_img = cam_tensor.squeeze().numpy() # (H, W)

        heatmap_colored = cm.jet(cam_img)[:, :, :3]  # (H, W, 3)
        video_frames = []

        for i in range(num_frames):
            raw_frame = input_tensor[0, i].detach().cpu().numpy() # (H, W)
            raw_frame = np.clip(raw_frame, 0, 1)
            
            raw_frame_rgb = np.stack([raw_frame] * 3, axis=-1)

            superimposed = 0.6 * raw_frame_rgb + 0.4 * heatmap_colored
            video_frames.append(superimposed)

        video_array = np.stack(video_frames) # (T, H, W, C)
        video_array = np.transpose(video_array, (0, 3, 1, 2)) # (T, C, H, W)
        video_array = np.expand_dims(video_array, axis=0) # (1, T, C, H, W)

        self.writer.add_video(tag, video_array, self.step_count, fps=4)

        handle_f.remove()
        handle_b.remove()
        self.policy_net.train()

    def log_input_state(self, state: State):
        tensor = self._preprocess(state)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        batch_size, num_frames, height, width = tensor.shape

        if num_frames > 1:
            frames = [tensor[0, i, :, :] for i in range(num_frames)]
            img_vis = torch.cat(frames, dim=1)
            img_vis = img_vis.unsqueeze(0)

            if self.step_count % 100 == 0:
                frame_diffs = []
                for i in range(num_frames - 1):
                    diff = torch.abs(tensor[0, i + 1, :, :] - tensor[0, i, :, :])
                    frame_diffs.append(diff)

                if frame_diffs:
                    diff_vis = torch.cat(frame_diffs, dim=1)
                    diff_vis = diff_vis.unsqueeze(0)
                    diff_vis = torch.clamp(diff_vis * 5.0, 0, 1)
                    self.writer.add_image(
                        "Debug/Frame_Differences", diff_vis, self.step_count
                    )
        else:
            img_vis = tensor[0]

        img_vis = torch.clamp(img_vis, 0, 1)
        self.writer.add_image("Debug/Preprocessed_Input", img_vis, self.step_count)
