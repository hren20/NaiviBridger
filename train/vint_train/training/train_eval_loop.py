import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train_navibridge, evaluate_navibridge
from vint_train.training.train_utils import train_cvae
from vint_train.models.navibridge.ddbm.resample import *
from vint_train.models.navibridge.ddbm.karras_diffusion import KarrasDenoiser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

def train_eval_loop_navibridge(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: UniformSampler,
    diffusuon: KarrasDenoiser,
    prior_policy: str,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    # ddbm params
    steps=10,
    clip_denoised: bool = True,
    sampler = "heun",
    sigma_min = 0.002,
    sigma_max = 80,
    churn_step_ratio = 0.,
    rho = 7.0,
    guidance = 1,
):
    """
    Train and evaluate the model for several epochs
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_navibridge(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                diffusion=diffusuon,
                noise_scheduler=noise_scheduler,
                prior_policy=prior_policy,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
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
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_navibridge(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    prior_policy=prior_policy,
                    diffusion=diffusuon,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
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
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})
    print()

def train_eval_loop_cvae(
        train_model: bool,
        model: nn.Module,
        optimizer:Adam,
        lr_scheduler: torch.optim.lr_scheduler.StepLR,
        train_loader: DataLoader,
        transform: transforms,
    #  num_itr: int,
        epochs: int,
        prior_policy: str,
    #  model_args,
        device: torch.device,
        project_folder: str,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        current_epoch: int = 0,
        save_freq=1,
        use_wandb: bool = True,
):
    """
    This function handles the training and evaluation loop for a CVAE model.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        sched (torch.optim.lr_scheduler, optional): Learning rate scheduler, can be None.
        dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        model_args (dict): Contains model-specific arguments, including prior_policy.
        opt (argparse.Namespace or similar): Contains other optimization settings like device and log_freq.
        ckpt_path (str): Path where model checkpoints and logs should be saved.
        log_freq (int, optional): Frequency of logging to wandb, defaults to 10 iterations.
        save_freq (int, optional): Frequency of saving the model, defaults to 100 iterations.
    """

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_cvae(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                use_wandb=use_wandb,
            )
            lr_scheduler.step()
        
        # Save model checkpoints and optimizer state
        if (epoch + 1) % save_freq == 0:
            numbered_path = os.path.join(project_folder, f"cvae_{epoch}.pth")
            model.save_model(ckpt_path=numbered_path, epoch=epoch)

            # save optimizer
            latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
            torch.save(optimizer.state_dict(), latest_optimizer_path)

            # save scheduler
            latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if lr_scheduler is not None: 
            lr_scheduler.step()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "navibridge":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params

    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params