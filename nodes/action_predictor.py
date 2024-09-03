#!/usr/bin/env python
import rospy
import csv
from std_msgs.msg import String, Float32MultiArray
import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv

import torch
import dill
import hydra
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
import numpy as np
import threading
import yaml
from enum import Enum, auto as enum_auto
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

class ActionPredictor:
    """ROS node that predicts actions using Diffusion Policy"""
    def __init__(self):
        rospy.init_node('action_predictor', anonymous=True)

        # Parameters
        self.checkpoint_path = rospy.get_param('~checkpoint_path', '/home/cerebro/diff/src/diffusion_policy/data/epoch=0020-val_loss=0.013.ckpt')

        self.num_inferences = rospy.get_param('~num_inferences', 100)

        self.num_actions_taken = rospy.get_param('~num_actions_taken', 100)

        # Publishers
        self.action_pub = rospy.Publisher('action_topic', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('csv_topic', Float32MultiArray, self.csv_callback)

        # Services
        self.srv_start_inference = rospy.Service('start_inference', std_srvs.srv.Trigger, self.start_inference)
        self.srv_stop_inference = rospy.Service('stop_inference', std_srvs.srv.Trigger, self.stop_inference)
        self.srv_start_action = rospy.Service('start_action', std_srvs.srv.Trigger, self.start_action)
        self.srv_stop_action = rospy.Service('stop_action', std_srvs.srv.Trigger, self.stop_action)
        self.enable_inference = False
        self.enable_action = False

        # Load checkpoint
        self.payload = torch.load(open(self.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']

        # Load workspace
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)

        # Load model
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        # Enable GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.eval().to(self.device)

        # This basically just stops the policy from limiting the number of actions in the output
        # of an inference so more actions can be taken if necessary (i.e. if the model inference
        # takes too long)
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        # log the n_actions_steps
        rospy.loginfo(f"n_action_steps: {self.policy.n_action_steps}")

        if self.num_actions_taken > self.policy.n_action_steps:
            rospy.logwarn(f"num_actions_taken ({self.num_actions_taken}) is greater than n_action_steps ({self.policy.n_action_steps})")

        def shutdown_hook():
            rospy.loginfo("Shutting down action_predictor")

        rospy.on_shutdown(shutdown_hook)

    def csv_callback(self, data):
        """Callback for CSV topic"""

        if self.enable_inference:
            # log the data from the CSV topic
            rospy.loginfo(f"Received data: {data.data}")


    def start_inference(self, req):
        """Starts the inference process"""
        self.enable_inference = True
        return std_srvs.srv.TriggerResponse(success=True, message="Inference started")

    def stop_inference(self, req):
        """Stops the inference process"""
        self.enable_inference = False
        return std_srvs.srv.TriggerResponse(success=True, message="Inference stopped")
    
    def start_action(self, req):
        """Starts the action process"""
        self.enable_action = True
        return std_srvs.srv.TriggerResponse(success=True, message="Action started")
    
    def stop_action(self, req):
        """Stops the action process"""
        self.enable_action = False
        return std_srvs.srv.TriggerResponse(success=True, message="Action stopped")

if __name__ == '__main__':
    try:
        action_predictor = ActionPredictor()
        rospy.spin()  # Keeps the node running and processing callbacks
    except rospy.ROSInterruptException:
        pass