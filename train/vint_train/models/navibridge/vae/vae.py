# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from torch_ema import ExponentialMovingAverage
from vint_train.models.navibridge.vae.conditional_mlp_1D_vae import *
import os

def kl_divergence_normal(mean1, mean2, std1, std2):
    kl_loss = ((std2 + 1e-9).log() - (std1 + 1e-9).log()
               + (std1.pow(2) + (mean2 - mean1).pow(2))
               / (2 * std2.pow(2) + 1e-9) - 0.5).sum(-1).mean()
    return kl_loss


class VAEModel():
    def __init__(self, model_args):
        self.net = None
        self.ema = None

        self.anneal_factor = 0.0
        self.prior_policy = 'gaussian'

    def sample_prior(self, cond, device, num_samples=None, x_prior=None, diffuse_step=None):
        num_samples = cond.shape[0]
        latent_sample = torch.randn((num_samples, self.net.latent_dim)).to(device)
        action_hat = self.net.decoder(torch.cat([cond.flatten(start_dim=1), latent_sample], dim=-1))
        action_hat = action_hat.reshape(-1, self.net.len_traj_pred, self.net.action_dim)
        return action_hat

    def get_loss(self, obs_img, naction, device):
        nobs = obs_img.to(device).float().flatten(start_dim=1)
        naction = naction.to(device).float()

        latent_post_dist = self.net.encoder(torch.cat([nobs, naction.flatten(1)], dim=-1))
        latent_post_rsample = latent_post_dist.rsample()
        latent_post_mean = latent_post_dist.mean
        latent_post_std = latent_post_dist.stddev

        latent_prior_mean = torch.zeros_like(latent_post_mean).float().to(device)
        latent_prior_std = torch.ones_like(latent_post_std).float().to(device)

        action_rec = self.net.decoder(torch.cat([nobs, latent_post_rsample], dim=-1))
        rec_loss = torch.nn.functional.mse_loss(action_rec, naction.flatten(1)) * 10.0
        kl_loss = self.anneal_factor * kl_divergence_normal(latent_post_mean, latent_prior_mean, latent_post_std, latent_prior_std)

        self.anneal_factor += 0.0001
        self.anneal_factor = 0.1 if self.anneal_factor > 0.1 else self.anneal_factor

        loss = rec_loss + kl_loss
        loss_info = {
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
        }
        return loss, loss_info

    def log_info(self, writer, log, loss_info, optimizer, itr, num_itr):
        writer.add_scalar(itr, 'loss', loss_info['loss'].detach())

        log.info("train_it {}/{} | lr:{} | loss:{}".format(
            1 + itr,
            num_itr,
            "{:.2e}".format(optimizer.param_groups[0]['lr']),
            "{:+.2f}".format(loss_info['loss'].item()),
        ))

    def load_model(self, model_args, device):

        if model_args['net_type'] == 'vae_mlp':
            self.net = VAEConditionalMLP(
                action_dim=model_args['action_dim'],
                len_traj_pred=model_args['len_traj_pred'],
                global_cond_dim=3 * model_args['image_size'][0] * model_args['image_size'][1] * (model_args['context_size'] + 1),
                latent_dim=model_args['latent_dim'],
                layer=model_args['layer']
            )
        else:
            raise NotImplementedError

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.99)
        if model_args['pretrain']:
            checkpoint = torch.load(model_args['ckpt_path'], map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            self.ema.load_state_dict(checkpoint["ema"])

        self.net.to(device)
        self.ema.to(device)

    def save_model(self, ckpt_path, epoch):
        torch.save({
            "net": self.net.state_dict(),
            "ema": self.ema.state_dict(),
        }, ckpt_path)
        print(f"Saved model to {ckpt_path}")


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]