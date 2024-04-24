if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import torch.utils
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
import timeit
import pandas as pd


from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

from imitation.utils.metrics import compute_variance_waypoints, compute_smoothness_from_vel, compute_smoothness_from_pos

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class EvalDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        try:
            self.load_checkpoint(path=cfg.task.ckpt_path)
            print(f"Resuming from checkpoint {cfg.task.ckpt_path}")
        except Exception as e:
            print(e)
            print("No checkpoint loaded")
            raise 

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()

        # assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset[:cfg.num_episodes], **cfg.dataloader)
        

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)



        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        # for logging the eval metrics
        metrics = pd.DataFrame()

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        step_log = {} # for wandb logging

        delta_traj = []
        times = [] # to store the time taken to generate the action

        seed = 0

        # ========= train for this epoch ==========
        train_losses = list()
        for data in train_dataloader:
            # device transfer
            batch = dict_apply(data, lambda x: x.to(device, non_blocking=True))
            
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run diffusion sampling on a training batch
            with torch.no_grad():
                # sample trajectory from training set, and evaluate difference
                obs_dict = batch['obs']
                gt_action = batch['action'].cpu().numpy()
                
                # generate action multiple times to get multimodality
                mm_traj = []
                for _ in range(int(cfg.num_seeds)):
                    # change seed
                    torch.manual_seed(seed)
                    start_time = timeit.default_timer()
                    action = policy.get_action(obs_dict)
                    end_time = timeit.default_timer()
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    mm_traj.append(action)
                    # compute the difference against the ground truth
                    error = action - gt_action
                    delta_traj.append(error)
                    seed += 1
        
            print(f"Average execution time for 50 : {np.array(times[-cfg.num_seeds:]).mean()}")
            step_log["execution_time"] = np.array(times[-cfg.num_seeds:]).mean()
            
            mm_traj = torch.tensor(mm_traj)
            
            # calculate Waypoint Variance: sum (along the trajectory dimension) of the pairwise L2- distance 
            # variance between waypoints at corresponding time
            waypoint_variance = compute_variance_waypoints(mm_traj)
            print(f"Mean Waypoint Variance: {waypoint_variance/mm_traj.shape[1]}")
            step_log["waypoint_variance"] = waypoint_variance
            step_log["mean_waypoint_variance"] = waypoint_variance/mm_traj.shape[1] # mean over time (trajectory) dimension
            # calculate Smoothness: sum (along the trajectory dimension) of the pairwise L2- distance between
            # consecutive waypoints
            if hasattr(cfg.task, 'control_mode'):
                if cfg.task.control_mode == "JOINT_POSITION":
                    smoothness = compute_smoothness_from_pos(mm_traj)
                elif cfg.task.control_mode == "JOINT_VELOCITY":
                    smoothness = compute_smoothness_from_vel(mm_traj)
            else: # lowdim task - default is velocity
                smoothness = compute_smoothness_from_vel(mm_traj)
            
            # compute average smoothness over trajectories
            smoothness = smoothness.mean()
            step_log["smoothness"] = smoothness
            print(f"Smoothness: {smoothness}")
            wandb.log(step_log)
            step_log = {}
        
            delta_traj_sample = np.array(delta_traj[-cfg.num_seeds:])
            mean_error = np.mean(delta_traj_sample)
            std_error = np.std(np.mean(delta_traj_sample, axis=1), axis=0).mean()
            print(f"Mean error: {mean_error}")
            print(f"Std error: {std_error}")
            wandb.log({"mean_error": mean_error, "std_error": std_error})
            metrics = pd.concat([metrics, pd.DataFrame(
                data=[
                    [waypoint_variance.numpy(), waypoint_variance.numpy()/mm_traj.shape[1], smoothness.numpy(), mean_error, std_error, np.array(times).mean(), np.array(times).std()]
                ],
                columns=["waypoint_variance", "mean_waypoint_variance", "smoothness", "mean_error", "std_error", "mean_execution_time", "std_execution_time"]
            )], ignore_index=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
