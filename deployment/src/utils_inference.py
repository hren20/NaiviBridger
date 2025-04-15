import os
import sys
import io
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
import cv2
from typing import List, Tuple, Dict, Optional
import importlib.resources as pkg_resources
from tqdm import tqdm

# models
from vint_train.models.navibridge.navibridge import NaviBridge, DenseNetwork, StatesPredNet
from vint_train.models.navibridge.navibridg_utils import NaviBridge_Encoder, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

from vint_train.models.navibridge.ddbm.karras_diffusion import KarrasDenoiser
from vint_train.models.navibridge.ddbm.resample import create_named_schedule_sampler
from vint_train.models.navibridge.navibridge import PriorModel, Prior_HandCraft
from vint_train.models.navibridge.vae.vae import VAEModel
from vint_train.models.model_utils import set_model_prior

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    model_type = config["model_type"]
    
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
        raise ValueError(f"Invalid model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)

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
    model.to(device)
    return model

def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def ensure_pil_image(item):
    if isinstance(item, PILImage.Image):
        return item
    elif isinstance(item, np.ndarray):
        return PILImage.fromarray(item)
    else:
        raise ValueError(f"Unsupported data type: {type(item)}")

def alternate_merge(list1, list2):
    list1 = [ensure_pil_image(item) for item in list1]
    list2 = [ensure_pil_image(item) for item in list2]

    merged_list = [None] * (len(list1) + len(list2))
    merged_list[::2] = list1
    merged_list[1::2] = list2

    return merged_list

# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def find_images(directory):
    files = os.listdir(directory)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return [os.path.join(directory, file) for file in image_files]

def project_and_draw(image, trajectories, extrinsic_matrix, intrinsic_matrix):
    """
    Project trajectories from world coordinates to pixel coordinates and draw them on the image using matplotlib.
    Note that the projection is not accurate, only for visualization.
    Args:
        image (np.array): The image on which to draw the trajectories.
        trajectories (list of list of tuples): Each trajectory is a list of (x, y, z) tuples.
        extrinsic_matrix (np.array): The 4x4 extrinsic matrix (world to camera).
        intrinsic_matrix (np.array): The 3x4 intrinsic matrix (camera to image).

    Returns:
        None: Displays the image with trajectories drawn using matplotlib.
    """
    # Convert image to numpy array
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]

    # Set up matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)
    ax.imshow(image_np)  # Show the image

    # Plot each trajectory
    for trajectory in trajectories:
        points = []
        for (x, y) in trajectory:
            # Convert from world to camera coordinates
            camera_coords = world_to_camera(np.array([x, y, -1.5]), extrinsic_matrix)
            # Project to pixel coordinates
            pixel_coords = camera_to_pixel(camera_coords, intrinsic_matrix)
            u, v = int(pixel_coords[0]), int(pixel_coords[1])
            u += (image.size[0] // 2)
            # v += (image.size[1] // 2)
            points.append((u, v))

        # Separate u and v coordinates for plotting
        if points:
            u_coords, v_coords = zip(*points)  # Unpack list of tuples into separate lists
            ax.plot(u_coords, v_coords, color='yellow', linewidth=8)  # Plot trajectory

    # Limit the view to the image's dimensions only
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Reverse the y-axis to match image coordinates
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Display the final image with trajectories
    plt.axis('off')  # Turn off axes for clarity
    return fig

def world_to_camera(world_coords, extrinsic_matrix):
    """
    Convert world coordinates to camera coordinates using the extrinsic matrix.

    Args:
        world_coords (np.array): Coordinates in the world frame.
        extrinsic_matrix (np.array): The 4x4 extrinsic matrix (world to camera).

    Returns:
        np.array: Coordinates in the camera frame.
    """
    # Convert to homogeneous coordinates
    world_coords_homogeneous = np.append(world_coords, 1)  
    # Transform to camera coordinates
    camera_coords_homogeneous = np.linalg.inv(extrinsic_matrix) @ world_coords_homogeneous
    # Convert back to 3D and normalize
    return camera_coords_homogeneous[:3] / camera_coords_homogeneous[3]  

def camera_to_pixel(camera_coords, intrinsic_matrix):
    """
    Project camera coordinates to pixel coordinates using the intrinsic matrix.

    Args:
        camera_coords (np.array): Coordinates in the camera frame.
        intrinsic_matrix (np.array): The 3x4 intrinsic matrix (camera to image).

    Returns:
        np.array: Coordinates in pixel space.
    """
    # Focal lengths and principal point from intrinsic matrix
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    X, Y, Z = camera_coords[0], camera_coords[1], camera_coords[2]
    
    # Project to pixel coordinates
    x = (fx * X / Z) + cx
    y = (fy * Y / Z) + cy

    return np.array([x, y])  # Convert to 2D pixel coordinates