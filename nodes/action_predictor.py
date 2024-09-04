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
from collections import deque

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

class FIFOQueue:
    """FIFO queue with a maximum length for recording obs data"""
    def __init__(self, max_len=100):
        self.queue = deque(maxlen=max_len)

    def push(self, item):
        self.queue.append(item)

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear() 

    def to_numpy(self):
        return np.array(self.queue)

class ActionPredictor:
    """ROS node that predicts actions using Diffusion Policy"""
    def __init__(self):
        rospy.init_node('action_predictor', anonymous=True)

        # Parameters
        self.checkpoint_path = rospy.get_param('checkpoint_path', '/home/cerebro/diff/src/diffusion_policy/data/epoch=0020-val_loss=0.013.ckpt')

        rospy.set_param('num_inferences', 100)
        rospy.set_param('num_actions_taken', 100)

        self.num_inferences = rospy.get_param('num_inferences', 100)

        self.num_actions_taken = rospy.get_param('num_actions_taken', 100)

        # Publishers
        self.patient_obs_pub = rospy.Publisher('patient_obs_topic', Float32MultiArray, queue_size=10)
        self.true_action_pub = rospy.Publisher('true_action_topic', Float32MultiArray, queue_size=10)

        # Subscribers
        rospy.Subscriber('csv_topic', Float32MultiArray, self.true_obs_callback)

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

        # Initialize the FIFO queue for storing obs data
        self.patient_obs_queue = FIFOQueue(max_len=self.policy.n_obs_steps)
        self.action_obs_queue = FIFOQueue(max_len=self.policy.n_obs_steps)

        # Flags
        self.inference_executed = False
        
        def shutdown_hook():
            rospy.loginfo("Shutting down action_predictor")

        rospy.on_shutdown(shutdown_hook)

    def true_obs_callback(self, data):
        """Callback for the true ground truth observation data"""

        # parse the data
        # first four elements are the patient observation
        patient_obs = data.data[:4]
        patient_obs_msg = Float32MultiArray(data=patient_obs)

        # convert the patient observation to a numpy array
        patient_obs_np = np.array(patient_obs)
        # push the patient observation to the queue
        self.patient_obs_queue.push(patient_obs_np)

        # if an inference has not yet been executed, fill the action queue with the patient observation
        # this is done to ensure that the observation set contains therapist action data when making the intial inferences
        # after the initial inferences, the action queue will be filled with the predicted action data
        if not self.inference_executed:
            self.action_obs_queue.push(patient_obs_np)

        # the rest of the elements are the true action
        true_action = data.data[4:]
        true_action_msg = Float32MultiArray(data=true_action)

        # publish the messages
        self.patient_obs_pub.publish(patient_obs_msg)
        self.true_action_pub.publish(true_action_msg)

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