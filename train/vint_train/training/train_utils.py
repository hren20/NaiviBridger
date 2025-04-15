import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from vint_train.models.navibridge.ddbm.karras_diffusion import KarrasDenoiser
from vint_train.models.navibridge.ddbm.resample import *
from vint_train.models.navibridge.ddbm.karras_diffusion import karras_sample
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )

def _compute_losses_navibridge(
    ema_model,
    diffusion,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    naction: torch.Tensor,
    action_category: torch.Tensor,
    device: torch.device,
    prior_policy: str,
    action_mask: torch.Tensor,
    model_kwargs=None,
    # ddbm params
    steps = 10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output_bridge(
        ema_model,
        diffusion,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        naction,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
        prior_policy=prior_policy,
        model_kwargs=batch_action_label,
        # ddbm params
        steps=steps,
        clip_denoised=clip_denoised,
        sampler=sampler,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        churn_step_ratio=churn_step_ratio,
        rho=rho,
        guidance=guidance,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']
    states_pred = model_output_dict['states_pred']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    action_category = action_category.to(device)
    class_probs, coords_pred = states_pred
    log_probs = torch.log(class_probs + 1e-8)

    nll_loss_fn = torch.nn.NLLLoss()
    class_loss = nll_loss_fn(log_probs, action_category)
    coords_loss = F.mse_loss(coords_pred, naction)

    class_loss_weight = 1.0
    coords_loss_weight = 1.0

    states_loss = class_loss_weight * class_loss + coords_loss_weight * coords_loss


    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
        "class_loss": class_loss,
        "coords_loss": coords_loss,
        "states_loss": states_loss,
    }

    return results

def forward_backward(
    noise_scheduler,
    diffusion,
    model,
    naction,
    obsgoal_cond,
    t,
    B,
    init_cond=None,
    train=True
):

    compute_losses = functools.partial(
        diffusion.training_bridge_losses,
        model,
        naction,
        t,
        global_cond=obsgoal_cond,
        model_kwargs=init_cond
    )
    losses_diffusion, denoised = compute_losses()

    if train:
        return losses_diffusion, denoised
    else:
        return denoised

def train_navibridge(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    diffusion: KarrasDenoiser,
    noise_scheduler: UniformSampler,
    prior_policy: str,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    # ddbm params
    steps = 10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    class_loss_logger = Logger("class_loss", "train", window_size=print_log_freq)
    coords_loss_logger = Logger("coords_loss", "train", window_size=print_log_freq)
    states_loss_logger = Logger("states_loss", "train", window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
        "class_loss": class_loss_logger,
        "coords_loss": coords_loss_logger,
        "states_loss": states_loss_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
                action_category,
            ) = data
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            if prior_policy == "handcraft":
                action_category = action_category.to(device)
                # Predict aciton states
                states_pred = model("states_pred_net", obsgoal_cond=obsgoal_cond)
                class_probs, coords_pred = states_pred
                log_probs = torch.log(class_probs + 1e-8)

                nll_loss_fn = torch.nn.NLLLoss()
                class_loss = nll_loss_fn(log_probs, action_category)
                coords_loss = F.mse_loss(coords_pred, naction)

                class_loss_weight = 1.0
                coords_loss_weight = 1.0

                states_loss = class_loss_weight * class_loss + coords_loss_weight * coords_loss

            if prior_policy == "cvae":
                prior_cond = batch_obs_images
            elif prior_policy == "handcraft":
                prior_cond = states_pred
            else:
                prior_cond = None
            if model.prior_model.prior is None:
                initial_samples = torch.randn(naction.shape, device=device)
            else:
                with torch.no_grad():
                    initial_samples = model.prior_model.sample(cond=prior_cond, device=device)
                assert initial_samples.shape[-1] == 2, "action dim must be 2"
            init_cond = {'xT': initial_samples}

            t, weights = noise_scheduler.sample(B, device)

            losses_diffusion, denoised = forward_backward(
    noise_scheduler, diffusion, model, naction, obsgoal_cond, t, B, init_cond=init_cond, train=True
)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            diffusion_loss = action_reduce(losses_diffusion["loss"] * weights)

            if prior_policy == "handcraft":
                loss = alpha * dist_loss + (1-alpha) * diffusion_loss + states_loss
            else:
                loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})
            if prior_policy == "handcraft":
                wandb.log({"states_loss": states_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_navibridge(
                            ema_model.averaged_model,
                            diffusion,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            naction,
                            action_category,
                            device,
                            prior_policy,
                            action_mask.to(device),
                            model_kwargs=init_cond,
                            # ddbm params
                            steps=steps,
                            clip_denoised=clip_denoised,
                            sampler=sampler,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            churn_step_ratio=churn_step_ratio,
                            rho=rho,
                            guidance=guidance,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_bridge_action_distribution(
                    ema_model.averaged_model,
                    diffusion,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    prior_policy,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                    # ddbm params
                    steps=steps,
                    clip_denoised=clip_denoised,
                    sampler=sampler,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    churn_step_ratio=churn_step_ratio,
                    rho=rho,
                    guidance=guidance,
                )


def evaluate_navibridge(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    prior_policy: str,
    diffusion: KarrasDenoiser,
    noise_scheduler: UniformSampler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
    # ddbm params
    steps = 10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    class_loss_logger = Logger("class_loss", "train", window_size=print_log_freq)
    coords_loss_logger = Logger("coords_loss", "train", window_size=print_log_freq)
    states_loss_logger = Logger("states_loss", "train", window_size=print_log_freq)
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
        "class_loss": class_loss_logger,
        "coords_loss": coords_loss_logger,
        "states_loss": states_loss_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
                action_category,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            # Predict aciton states
            states_pred = ema_model("states_pred_net", obsgoal_cond=obsgoal_cond)
            class_probs, coords_pred = states_pred

            if prior_policy == "cvae":
                prior_cond = batch_obs_images
            elif prior_policy == "handcraft":
                prior_cond = states_pred
            else:
                prior_cond = None

            if ema_model.prior_model.prior is None:
                initial_samples = torch.randn(naction.shape, device=device)
            else:
                with torch.no_grad():
                    initial_samples = ema_model.prior_model.sample(cond=prior_cond, device=device)
                    # initial_samples = initial_samples.squeeze(1)
            init_cond = {'xT': initial_samples}
            t, weights = noise_scheduler.sample(B, device)
            with torch.no_grad():
                rand_mask_noise_pred = forward_backward(
                noise_scheduler, diffusion, ema_model, naction, rand_mask_cond, t, B, init_cond=init_cond, train=False
        )
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, initial_samples)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            with torch.no_grad():
                no_mask_noise_pred = forward_backward(
                noise_scheduler, diffusion, ema_model, naction, obsgoal_cond, t, B, init_cond=init_cond, train=False
        )
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, initial_samples)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            with torch.no_grad():
                goal_mask_noise_pred = forward_backward(
                noise_scheduler, diffusion, ema_model, naction, goal_mask_cond, t, B, init_cond=init_cond, train=False
        )
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, initial_samples)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_navibridge(
                            ema_model,
                            diffusion,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            naction,
                            action_category,
                            device,
                            prior_policy,
                            action_mask.to(device),
                            model_kwargs=init_cond,
                            # ddbm params
                            steps=steps,
                            clip_denoised=clip_denoised,
                            sampler=sampler,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            churn_step_ratio=churn_step_ratio,
                            rho=rho,
                            guidance=guidance,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_bridge_action_distribution(
                    ema_model,
                    diffusion,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    prior_policy,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                    # ddbm params
                    steps=steps,
                    clip_denoised=clip_denoised,
                    sampler=sampler,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    churn_step_ratio=churn_step_ratio,
                    rho=rho,
                    guidance=guidance,
                )

def train_cvae(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    use_wandb: bool = True,
):
    model.net.train()
    num_batches = len(dataloader)

    loss_logger = Logger("loss", "train", window_size=print_log_freq)
    rec_loss_logger = Logger("rec_loss", "train", window_size=print_log_freq)
    kl_loss_logger = Logger("kl_loss", "train", window_size=print_log_freq)
    loggers = {
        "loss": loss_logger,
        "rec_loss": rec_loss_logger,
        "kl_loss": kl_loss_logger,
    }

    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
                action_category,
            ) = data
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            loss, loss_info = model.get_loss(batch_obs_images, naction, device)
            # Backpropagate
            loss.backward()

            # Update the model parameters
            optimizer.step()
            optimizer.zero_grad()

            # Update Exponential Moving Average (EMA)
            model.ema.update()

            if i % print_log_freq == 0:
                for key, value in loss_info.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def model_output_bridge(
    model: nn.Module,
    diffusion: KarrasDenoiser,
    noise_scheduler: UniformSampler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    naction: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    prior_policy: str,
    model_kwargs=None,
    # ddbm params
    steps = 10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
    naction = naction.repeat_interleave(num_samples, dim=0)
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # Predict aciton states
    states_pred = model("states_pred_net", obsgoal_cond=obs_cond)

    if prior_policy == "cvae":
        prior_cond = batch_obs_images.repeat_interleave(num_samples, dim=0)
    else:
        prior_cond = states_pred

    # initialize action from Gaussian noise
    if model.prior_model is None:
        initial_samples = torch.randn(naction.shape, device=device)
    else:
        with torch.no_grad():
            initial_samples = model.prior_model.sample(cond=prior_cond, device=device)
        assert initial_samples.shape[-1] == 2, "action dim must be 2"
    diffusion_output = {'xT': initial_samples}
    B = initial_samples.shape[0]
    t, weights = noise_scheduler.sample(B, device)
    with torch.no_grad():
        diffusion_output = forward_backward(
        noise_scheduler, diffusion, model, naction, obs_cond, t, B, init_cond=diffusion_output, train=False
        )

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    states_pred = model("states_pred_net", obsgoal_cond=obsgoal_cond)
    # initialize action from Gaussian noise

    if prior_policy == "cvae":
        prior_cond = batch_obs_images.repeat_interleave(num_samples, dim=0)
    else:
        prior_cond = states_pred


    if model.prior_model is None:
        initial_samples = torch.randn(naction.shape, device=device)
    else:
        with torch.no_grad():
            initial_samples = model.prior_model.sample(cond=prior_cond, device=device)
        assert initial_samples.shape[-1] == 2, "action dim must be 2"
    diffusion_output = {'xT': initial_samples}

    B = initial_samples.shape[0]
    t, weights = noise_scheduler.sample(B, device)
    diffusion_output = forward_backward(
        noise_scheduler, diffusion, model, naction, obsgoal_cond, t, B, init_cond=diffusion_output, train=False
    )

    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
        'states_pred': states_pred,
    }

def visualize_diffusion_bridge_action_distribution(
    ema_model: nn.Module,
    diffusion: KarrasDenoiser,
    noise_scheduler: UniformSampler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    prior_policy: str,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
    # ddbm params
    steps = 10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    deltas = get_delta(batch_action_label)
    ndeltas = normalize_data(deltas, ACTION_STATS)
    naction = from_numpy(ndeltas).to(device)
    assert naction.shape[-1] == 2, "action dim must be 2"

    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output_bridge(
            ema_model,
            diffusion,
            noise_scheduler,
            obs,
            goal,
            naction,
            pred_horizon,
            action_dim,
            num_samples,
            device,
            prior_policy,
            # ddbm params
            steps=steps,
            clip_denoised=clip_denoised,
            sampler=sampler,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            churn_step_ratio=churn_step_ratio,
            rho=rho,
            guidance=guidance,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)
