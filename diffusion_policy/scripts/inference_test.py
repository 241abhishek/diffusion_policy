"""
Script to test inference on a trained model.

Usage:
python inference_test.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import os
import torch
import numpy as np
import click
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from torch.utils.data import DataLoader as Dataloader
import dill
import argparse
import tqdm
import time

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath(
        'diffusion_policy','config')),
    config_name="train_exo_diffusion_unet_lowdim_workspace"
)
def main(cfg: OmegaConf):

    # path to the checkpoint (replace with the path to the checkpoint you want to test)
    checkpoint_path = "/home/cerebro/diff/diffusion_policy/data/epoch=0020-val_loss=0.013.ckpt"
    
    # resolve immediately so all the ${now:} resolvers
    OmegaConf.resolve(cfg)

    # configure validation dataset
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = Dataloader(val_dataset, **cfg.val_dataloader)
    normalizer = dataset.get_normalizer()

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.model.set_normalizer(normalizer)

    # configure model
    workspace.model.set_normalizer(normalizer)
    # load checkpoint
    workspace.load_checkpoint(checkpoint_path)

    # device transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workspace.model.to(device)

    # set the number of inferences to make 
    num_inf = 10

    # create empty numpy arrays to store the observations, actual actions, and predicted actions
    obs_list = []
    actual_action_list = []
    predicted_action_list = []
    time_list = []

    # sample random indices from the validation dataset
    indices = torch.randperm(len(val_dataloader.dataset))[:num_inf]

    # set the model to evaluation mode
    workspace.model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(indices):
            # fetch the observation-action pair from the validation dataset
            data = val_dataset.__getitem__(i)
            obs_dict = {'obs': data['obs'].unsqueeze(0)}
            gt_action = data['action']
            # only include the last 200 valuse in the ground truth action
            gt_action = gt_action[-200:]
            # calculate the time taken to predict the action
            start = time.time()
            result = workspace.model.predict_action(obs_dict)
            end = time.time()
            time_list.append(end-start)
            pred_action = result['action'].squeeze(0)
            obs_list.append(data['obs'][:100].numpy())
            actual_action_list.append(gt_action.numpy())
            predicted_action_list.append(pred_action.to('cpu').numpy())

    # calculate the mean time taken to predict the action
    mean_time = sum(time_list)/len(time_list)
    print(f"Time taken: {time_list}")
    print(f"Mean time taken to predit the action: {mean_time}")
    
    # convert the observation, actual action, and predicted action lists from radians to degrees
    obs_list = np.array([np.rad2deg(obs) for obs in obs_list])
    actual_action_list = np.array([np.rad2deg(action) for action in actual_action_list])
    predicted_action_list = np.array([np.rad2deg(action) for action in predicted_action_list])

    # plot the actual actions against the predicted actions to visualize the performance of the model
    import matplotlib.pyplot as plt
    prediction_horizon = 200
    observation_horizon = 100
    for k in range(num_inf):
        # Create a figure with 4 subplots for tracking the 4 joint positions
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        fig.suptitle(f"Joint Positions Over Time - {k+1}")
        # Increase vertical space between subplots
        fig.subplots_adjust(hspace=0.6)

        # Initialize lines for each subplot
        lines = []
        titles = ["Left Hip", "Left Knee", "Right Hip", "Right Knee"]
        for i, ax in enumerate(axes):
            ax.set_xticks(np.arange(0, prediction_horizon + observation_horizon + 1, 30))
            ax.set_xticklabels([f'{x/100.0:.1f}' for x in np.arange(0, prediction_horizon + observation_horizon + 1, 30)])
            if i == 0 or i == 1:
                ax.set_ylim(np.min(np.concatenate((actual_action_list[k, :, i + 2], predicted_action_list[k, :, i + 2], obs_list[k, :, i], obs_list[k, :, i + 6]))) - 10, 
                            np.max(np.concatenate((actual_action_list[k, :, i + 2], predicted_action_list[k, :, i + 2], obs_list[k, :, i], obs_list[k, :, i + 6]))) + 10)
            else:
                ax.set_ylim(np.min(np.concatenate((actual_action_list[k, :, i - 2], predicted_action_list[k, :, i - 2], obs_list[k, :, i], obs_list[k, :, i + 2]))) - 10, 
                            np.max(np.concatenate((actual_action_list[k, :, i - 2], predicted_action_list[k, :, i - 2], obs_list[k, :, i], obs_list[k, :, i + 2]))) + 10) 
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Joint Position (degrees)")
            ax.set_title(titles[i])
            line1, = ax.plot([], [], label="Actual Action")
            line2, = ax.plot([], [], label="Predicted Action")
            line3, = ax.plot([], [], label="Observation Data (Patient)")
            line4, = ax.plot([], [], label="Observation Data (Instructor)")
            lines.append((line1, line2, line3, line4))

        # loop over the inferences and plot num_inf number of plots
        for j, (line1, line2, line3, line4) in enumerate(lines):
            if j == 0 or j == 1:
                line1.set_data(np.arange(observation_horizon, prediction_horizon + observation_horizon), actual_action_list[k, :, j + 2])
                line2.set_data(np.arange(observation_horizon, prediction_horizon + observation_horizon), predicted_action_list[k, :, j + 2])
                line3.set_data(np.arange(observation_horizon), obs_list[k, :, j])
                line4.set_data(np.arange(observation_horizon), obs_list[k, :, j + 6])
            else:
                line1.set_data(np.arange(observation_horizon, prediction_horizon + observation_horizon), actual_action_list[k, :, j - 2])
                line2.set_data(np.arange(observation_horizon, prediction_horizon + observation_horizon), predicted_action_list[k, :, j - 2])
                line3.set_data(np.arange(observation_horizon), obs_list[k, :, j])
                line4.set_data(np.arange(observation_horizon), obs_list[k, :, j + 2])

        # Create a single legend for the entire figure
        fig.legend(handles=[line1, line2, line3, line4], 
            labels=["Actual Action", "Predicted Action", "Observation Data (Patient)", "Observation Data (Instructor)"], 
            loc="upper right")

        # save the plot
        plt.savefig(f"images/joint_positions_{k+1}.png")
if __name__ == "__main__": 
    main()