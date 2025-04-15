from vint_train.models.navibridge.navibridge import NaviBridge, DenseNetwork, StatesPredNet
from vint_train.models.navibridge.navibridg_utils import NaviBridge_Encoder, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from vint_train.models.navibridge.ddbm.karras_diffusion import KarrasDenoiser
from vint_train.models.navibridge.ddbm.resample import create_named_schedule_sampler
from vint_train.models.navibridge.navibridge import PriorModel, Prior_HandCraft
from vint_train.models.navibridge.vae.vae import VAEModel

import os
def create_model(config, device):
    """
    Create a model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary that includes model type and various parameters.

    Returns:
        model (object): Created model based on the specified configuration.

    Raises:
        ValueError: If the model type or vision encoder is not supported.
    """
    # Create the model based on configuration
    if config["model_type"] == "navibridge":
        # Select and configure the vision encoder
        if config["vision_encoder"] == "navibridge_encoder":
            vision_encoder = NaviBridge_Encoder(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        # Create the noise prediction network and distribution prediction network
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        states_pred_net = StatesPredNet(embedding_dim=config["encoding_size"],
                                        class_num=config["class_num"],
                                        len_traj_pred=config["len_traj_pred"])

        model = NaviBridge(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
            states_pred_net=states_pred_net,
        )
        set_model_prior(model, config, device)
    elif config["model_type"] == "cvae":
        model = VAEModel(config)
    else:
        raise ValueError(f"Model {config['model_type']} not supported")

    return model

def create_noise_scheduler(config):
    if config["model_type"] == "navibridge":
        diffusion = create_diffusion(config)
        noise_scheduler = create_named_schedule_sampler(config["sampler_name"], diffusion)
        return noise_scheduler, diffusion

def create_diffusion(config,
                     ):
    # ddbm params
    sigma_data=config["sigma_data"]
    sigma_min=config["sigma_min"]
    sigma_max=config["sigma_max"]
    pred_mode=config["pred_mode"]
    weight_schedule=config["weight_schedule"]
    beta_d=config["beta_d"]
    beta_min=config["beta_min"]
    cov_xy=config["cov_xy"]
    diffusion = KarrasDenoiser(
        sigma_data=sigma_data,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        beta_d=beta_d,
        beta_min=beta_min,
        cov_xy=cov_xy,
        weight_schedule=weight_schedule,
        pred_mode=pred_mode
    )
    return diffusion

def set_model_prior(model, prior_args, device):
    prior = load_prior_model(prior_args, device)
    if prior_args['prior_policy'] == 'handcraft':
        prior_model = PriorModel(prior=prior, len_traj_pred=prior_args["len_traj_pred"], action_dim=prior_args["action_dim"])
    elif prior_args['prior_policy'] == 'cluster':
        pass
    elif prior_args['prior_policy'] == 'gaussian':
        prior_model = PriorModel(prior=prior, len_traj_pred=prior_args["len_traj_pred"], action_dim=prior_args["action_dim"])
    elif prior_args['prior_policy'] == 'cvae':
        prior_model = PriorModel(prior=prior, len_traj_pred=prior_args["len_traj_pred"], action_dim=prior_args["action_dim"])
    model.prior_model = prior_model
    return model


def load_prior_model(prior_args, device):
    if prior_args['prior_policy'] == 'handcraft':
        prior_model = Prior_HandCraft(
            class_num=prior_args["class_num"],
            angle_ranges=prior_args["angle_ranges"],
            min_std_angle=prior_args["min_std_angle"],
            max_std_angle=prior_args["max_std_angle"],
            min_std_length=prior_args["min_std_length"],
            max_std_length=prior_args["max_std_length"],
            len_traj_pred=prior_args["len_traj_pred"],
        )
    elif prior_args['prior_policy'] == 'cluster':
        pass
    elif prior_args['prior_policy'] == 'gaussian':
        prior_model = None
    elif prior_args['prior_policy'] == 'cvae':
        from vint_train.models.navibridge.vae.vae import VAEModel
        model_spec_names = prior_args['net_type']
        ckpt_path = prior_args["ckpt_path"]
        prior_args['pretrain'] = True
        prior_args['ckpt_path'] = ckpt_path

        prior_model = VAEModel(prior_args)
        prior_model.load_model(model_args=prior_args, device=device)

    else:
        raise NotImplementedError(f"Can not be found prior policy: {prior_args['prior_policy']}")

    return prior_model

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    ReduceLROnPlateau,
    StepLR
)
from warmup_scheduler import GradualWarmupScheduler

def get_optimizer_and_scheduler(config, model):
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()

    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None

    if config["model_type"] == "cvae":
        if config['lr_gamma'] < 1.0:
            scheduler = StepLR(optimizer,
                            step_size=config["lr_step"],
                            gamma=config["lr_gamma"])
        else:
            scheduler = None

    elif config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()

        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = CyclicLR(
                optimizer,
                base_lr=lr / 10.0,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config.get("warmup", False):
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    return optimizer, scheduler
