import os
import argparse
import time
import pdb
import numpy as np
import yaml

import torch
import torch.nn as nn

from vint_train.training.train_utils import get_delta, normalize_data
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

class NaviBridge(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net,
                       states_pred_net):
        super(NaviBridge, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        self.states_pred_net = states_pred_net
        self.prior_model = None
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        elif func_name == "states_pred_net":
            output = self.states_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output


class StatesPredNet(nn.Module):
    def __init__(self, embedding_dim, class_num, len_traj_pred):
        super(StatesPredNet, self).__init__()
        self.output_length = class_num + len_traj_pred * 2
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.len_traj_pred = len_traj_pred

        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, self.output_length)
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)

        class_logits = output[:, :self.class_num]
        coords = output[:, self.class_num:]

        class_probs = nn.functional.softmax(class_logits, dim=-1)

        coords = coords.reshape((-1, self.len_traj_pred, 2))

        return class_probs, coords


class PriorModel:
    def __init__(self, prior=None, action_dim=2, len_traj_pred=8):
        self.prior = prior
        self.action_dim = action_dim
        self.len_traj_pred = len_traj_pred

    def sample(self, cond, device, num_samples=1):
        states = cond

        if self.prior is None:
            if type(states) is np.array:
                states_torch = torch.as_tensor(states)
            else:
                states_torch = states

            if isinstance(states_torch, tuple):
                prior_sample = torch.randn((states_torch[1].shape[0], self.len_traj_pred, self.action_dim), device=device)
            else:
                prior_sample = torch.randn((states_torch.shape[0], self.len_traj_pred, self.action_dim), device=device)
        else:
            if type(states) is np.array:
                states_torch = torch.as_tensor(states)
            else:
                states_torch = states

            prior_sample = self.prior.sample_prior(states_torch, num_samples=num_samples, device=device)

            prior_sample = prior_sample.to(device)

        return prior_sample

class Prior_HandCraft:
    def __init__(self,
                class_num: int = 5,
                angle_ranges: list = None, # [(0, 67.5), (67.5, 112.5), (112.5, 180), (180, 270), (270, 360)]
                min_std_angle: float = 5.0,
                max_std_angle: float = 20.0,
                min_std_length: float = 1.0,
                max_std_length: float = 5.0,
                num_samples: int = 1,
                len_traj_pred=8,
    ):
        if angle_ranges is None:
            self._set_angle_ranges()
        else:
            self.angle_ranges = angle_ranges
        self.class_num = class_num
        self.min_std_angle = min_std_angle
        self.max_std_angle = max_std_angle
        self.min_std_length = min_std_length
        self.max_std_length = max_std_length
        self.num_samples = num_samples
        self.len_traj_pred = len_traj_pred

    def _set_angle_ranges(self):
        self.angle_ranges = [(0, 67.5),
                             (67.5, 112.5),
                             (112.5, 180),
                             (180, 270),
                             (270, 360)]

    def _extract_length(self, coords):
        x_end, y_end = coords[-1]
        length = np.sqrt(x_end**2 + y_end**2)

        return length

    def _exact_states(self, states_pred):
        class_probs, coords_pred = states_pred
        return class_probs.cpu(), coords_pred.cpu()

    def _convert_confidence_to_std(self, confidence, min_std=0.5, max_std=5.0):
        return min_std + (max_std - min_std) * (1 - confidence)

    def _preprocess(self, actions):
        actions = np.squeeze(actions)
        if actions.ndim == 2:
            actions = np.expand_dims(actions, axis=0)
        deltas = get_delta(actions)
        ndeltas = normalize_data(deltas, ACTION_STATS)
        naction = from_numpy(ndeltas)
        return naction

    def sample_prior(self, states, num_samples=None, device=None):
        if num_samples is not None:
            self.num_samples = num_samples

        class_probs, coords_pred = self._exact_states(states)
        batch_size = class_probs.shape[0]
        sampled_trajectories_batch = []

        for batch_idx in range(batch_size):
            class_probs_single = class_probs[batch_idx]
            coords_pred_single = coords_pred[batch_idx]

            self.path_length_mean = self._extract_length(coords_pred_single)
            length_confidence = class_probs_single

            sampled_trajectories = []

            for _ in range(self.num_samples):
                category = torch.multinomial(class_probs_single, num_samples=1).item()

                angle_range = self.angle_ranges[category]

                angle_std = self._convert_confidence_to_std(
                    class_probs_single[category],
                    min_std=self.min_std_angle,
                    max_std=self.max_std_angle
                )

                length_std = self._convert_confidence_to_std(
                    length_confidence[category],
                    min_std=self.min_std_length,
                    max_std=self.max_std_length
                )

                angle_mean = (angle_range[0] + angle_range[1]) / 2
                angle_sample = np.random.normal(loc=angle_mean, scale=angle_std)

                path_length_sample = np.random.normal(loc=self.path_length_mean, scale=length_std)

                x_end = path_length_sample * np.cos(np.radians(angle_sample))
                y_end = path_length_sample * np.sin(np.radians(angle_sample))

                h_mid = (0 + x_end) / 2
                h_range = (x_end - 0) / 2
                if y_end >= 0:
                    h = np.random.uniform(x_end, x_end * 2)
                else:
                    h = np.random.uniform(h_mid - h_range / 2, h_mid - h_range / 4)

                a = (y_end - 0) / ((x_end - 0) * (x_end + 0 - 2 * h))
                k = -a * h**2

                x_vals = np.linspace(0, x_end, self.len_traj_pred)
                y_vals = a * (x_vals - h) ** 2 + k
                combined = np.stack((y_vals, x_vals), axis=1)
                sampled_trajectories.append(combined)

            sampled_trajectories = np.stack(sampled_trajectories, axis=0)

            sampled_trajectories_batch.append(sampled_trajectories)

        actions = np.stack(sampled_trajectories_batch, axis=0)
        naction = self._preprocess(actions)
        return naction
