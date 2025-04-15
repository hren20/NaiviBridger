import argparse
import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from vint_train.models.model_utils import create_noise_scheduler
from vint_train.training.train_utils import get_action
from vint_train.visualizing.action_utils import plot_trajs_and_points
from vint_train.models.navibridge.ddbm.karras_diffusion import karras_sample

import torch
import numpy as np
import yaml
from PIL import Image as PILImage
import matplotlib.pyplot as plt

from utils_inference import to_numpy, transform_images, load_model, project_and_draw

PARAMS_PATH = "../config/params.yaml"
with open(PARAMS_PATH, "r") as f:
    params_config = yaml.safe_load(f)
image_path = params_config["image_path"]

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
ACTION_STATS = {}
ACTION_STATS['min'] = np.array([-2.5, -4])
ACTION_STATS['max'] = np.array([5, 4])
# GLOBALS
context_queue = []
context_size = None

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_bottom_folder_name(path):
    folder_path = os.path.dirname(path)
    bottom_folder_name = os.path.basename(folder_path)
    return bottom_folder_name

def ensure_directory_exists(save_path):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def path_project_plot(img, path, args, camera_extrinsics, camera_intrinsics):
    for i, naction in enumerate(path):
        gc_actions = to_numpy(get_action(naction))
        fig = project_and_draw(img, gc_actions, camera_extrinsics, camera_intrinsics)
        dir_basename = get_bottom_folder_name(image_path)
        save_path = os.path.join('../output', dir_basename, f'png_{args.model}_image_with_trajs_{i}.png')
        ensure_directory_exists(save_path)
        fig.savefig(save_path)
        save_path = os.path.join('../output', dir_basename, f'svg_{args.model}_image_with_trajs_{i}.svg')
        ensure_directory_exists(save_path)
        fig.savefig(save_path)
        print(f"output image saved as {save_path}")

def main(args):
    camera_intrinsics = np.array([[470.7520828622471, 0, 16.00531005859375],
                    [0, 470.7520828622471, 403.38909912109375],
                    [0, 0, 1]])
    camera_extrinsics = np.array([[0, 0, 1, -0.600],
                                  [-1, 0, 0, -0.000],
                                  [0, -1, 0, -0.042],
                                  [0, 0, 0, 1]])
    global context_size, image_path
    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    if model_params["model_type"] == "cvae":
        if "train_params" in model_params:
            model_params.update(model_params["train_params"])
        if "diffuse_params" in model_params:
            model_params.update(model_params["diffuse_params"])

    if model_params.get("prior_policy", None) == "cvae":
        if "diffuse_params" in model_params:
            model_params.update(model_params["diffuse_params"])

    context_size = model_params["context_size"]
    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model.eval()

    if model_params["model_type"] == "navibridge":
        noise_scheduler, diffusion = create_noise_scheduler(model_params)

    for i in range(4):
        
        img_path = image_path + str(i + 18) + ".png"
        img = PILImage.open(img_path)
        context_queue.append(img)

    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)

    # infer action
    obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
    obs_images = obs_images.to(device)
    fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
    mask = torch.ones(1).long().to(device)

    # You can change the fake_goal to a goal image, do the same operation as obs_images
    # goal_image = transform_images(goal_image, model_params["image_size"], center_crop=False)
    # goal_image = goal_image.to(device)
    # mask = torch.zeros(1).long().to(device)

    obs_cond_gc = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
    scale_factor=3 * MAX_V / RATE
    with torch.no_grad():
        if len(obs_cond_gc.shape) == 2:
            obs_cond_gc = obs_cond_gc.repeat(args.num_samples, 1)
        else:
            obs_cond_gc = obs_cond_gc.repeat(args.num_samples, 1, 1)

        if model_params["model_type"] == "navibridge":
            if model_params["prior_policy"] == "handcraft":
                # Predict aciton states
                states_pred = model("states_pred_net", obsgoal_cond=obs_cond_gc)

            if model_params["prior_policy"] == "cvae":
                prior_cond = obs_images.repeat_interleave(args.num_samples, dim=0)
            elif model_params["prior_policy"] == "handcraft":
                prior_cond = states_pred
            else:
                prior_cond = None

            # initialize action from Gaussian noise
            if model.prior_model.prior is None:
                initial_samples = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
            else:
                with torch.no_grad():
                    initial_samples = model.prior_model.sample(cond=prior_cond, device=device)
                assert initial_samples.shape[-1] == 2, "action dim must be 2"
            naction, path, nfe = karras_sample(
                diffusion,
                model,
                initial_samples,
                None,
                steps=model_params["num_diffusion_iters"],
                model_kwargs=initial_samples,
                global_cond=obs_cond_gc,
                device=device,
                clip_denoised=model_params["clip_denoised"],
                sampler="heun",
                sigma_min=diffusion.sigma_min,
                sigma_max=diffusion.sigma_max,
                churn_step_ratio=model_params["churn_step_ratio"],
                rho=model_params["rho"],
                guidance=model_params["guidance"]
            )

    gc_actions = to_numpy(get_action(naction))

    gc_actions *= scale_factor
    if args.path_visual:
        path_project_plot(context_queue[-1], path, args, camera_extrinsics, camera_intrinsics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run DIFFUSION NAVIGATION demo")
    parser.add_argument(
        "--model",
        "-m",
        default="navibridge_noise",
        type=str,
        help="model name (hint: check ../config/models.yaml)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=100,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--path-visual",
        default=True,
        type=bool,
        help="visualization",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=0,
        type=int,
        help="device",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
