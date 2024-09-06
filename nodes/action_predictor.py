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
    def __init__(self, max_len=None):
        self.queue = deque(maxlen=max_len)

    def push(self, item):
        self.queue.append(item)

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear() 

    def to_numpy(self):
        return np.array(self.queue)

    def from_numpy(self, np_array):
        self.clear() # clear the queue
        if isinstance(np_array, np.ndarray) and np_array.ndim == 2:
            for row in np_array:
                self.push(row)
        else:
            raise ValueError("Input must be a 2D numpy array")

class ActionPredictor:
    """ROS node that predicts actions using Diffusion Policy"""
    def __init__(self):
        rospy.init_node('action_predictor', anonymous=True)

        # Parameters
        self.checkpoint_path = rospy.get_param('checkpoint_path', '/home/cerebro/diff/src/diffusion_policy/data/epoch=0030-val_loss=0.018.ckpt')

        rospy.set_param('num_inferences', 100)
        rospy.set_param('num_actions_taken', 100)

        self.num_inferences = rospy.get_param('num_inferences', 100)

        self.num_actions_taken = rospy.get_param('num_actions_taken', 100)

        # Publishers
        self.patient_obs_pub = rospy.Publisher('patient_obs_topic', Float32MultiArray, queue_size=10)
        self.true_action_pub = rospy.Publisher('true_action_topic', Float32MultiArray, queue_size=10)
        self.predicted_action_pub = rospy.Publisher('predicted_action_topic', Float32MultiArray, queue_size=10)

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
        rospy.loginfo(f"Using device: {self.device}")
        self.policy.eval().to(self.device)

        # This basically just stops the policy from limiting the number of actions in the output
        # of an inference so more actions can be taken if necessary (i.e. if the model inference
        # takes too long)
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        # log the n_actions_steps
        # rospy.loginfo(f"n_action_steps: {self.policy.n_action_steps}")

        if self.num_actions_taken > self.policy.n_action_steps:
            rospy.logwarn(f"num_actions_taken ({self.num_actions_taken}) is greater than n_action_steps ({self.policy.n_action_steps})")

        # Initialize the FIFO queue for storing obs data
        self.patient_obs_queue = FIFOQueue(max_len=self.policy.n_obs_steps)
        # self.action_obs_queue = FIFOQueue(max_len=self.policy.n_obs_steps)

        # Initialize the FIFO queue for storing predicted action data
        self.predicted_action_queue = FIFOQueue()

        # Flags
        self.inference_executed = False
        self.running_inference = False
        
        # Attributes for action prediction
        self.latency_counter = 0 # this counter is used to keep track of the latency of the action prediction
        self.action_prediction_array = np.empty((0,4), dtype=np.float32) # this array is used to store the predicted actions

        # Subscribers
        rospy.Subscriber('csv_topic', Float32MultiArray, self.true_obs_callback)

        def shutdown_hook():
            rospy.loginfo("Shutting down action_predictor")

        rospy.on_shutdown(shutdown_hook)

    def true_obs_callback(self, data):
        """Callback for the true ground truth observation data"""
        # increment the latency counter
        self.latency_counter += 1

        if self.action_prediction_array.size > 0:

            rospy.loginfo(f"Lantency counter: {self.latency_counter}")
            # remove the first latency_counter rows from the action_prediction_array
            # this is done to account for the delay in action prediction
            self.action_prediction_array = self.action_prediction_array[self.latency_counter:]
            rospy.loginfo(f"Action prediction array shape: {self.action_prediction_array.shape}")
            # convert the action_prediction_array to a queue
            self.predicted_action_queue.from_numpy(self.action_prediction_array)
            self.action_prediction_array = np.empty((0,4), dtype=np.float32) # reset the action_prediction_array
            
            # reset the latency counter
            self.latency_counter = 0

            # set the inference_executed flag
            self.inference_executed = True

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
            # self.action_obs_queue.push(patient_obs_np)
            pass
        elif self.inference_executed and self.predicted_action_queue.size() > 0:
            # fetch the predicted action from the predicted action queue
            predicted_action = self.predicted_action_queue.queue.popleft()
            assert predicted_action.size == 4, "Predicted action must have 4 elements"
            # push the predicted action to the action queue
            # self.action_obs_queue.push(predicted_action)
            # convert the predicted action to a numpy array of shape (1,4), dtype=np.float32
            predicted_action = np.array(predicted_action, dtype=np.float32).reshape(1,4).squeeze().tolist()
            print(f"Predicted action: {predicted_action}")
            predicted_action_msg = Float32MultiArray(data=predicted_action)
            # publish the predicted action
            self.predicted_action_pub.publish(predicted_action_msg)

        # the last four elements are the true action
        true_action = data.data[4:]
        # push the true action to the action queue
        # self.action_obs_queue.push(true_action)
        print(f"True action: {true_action}")
        true_action_msg = Float32MultiArray(data=true_action)

        # publish the messages
        self.patient_obs_pub.publish(patient_obs_msg)
        self.true_action_pub.publish(true_action_msg)

        # if the inference process is enabled, run an inference in a separate thread
        if self.enable_inference and not self.running_inference and self.patient_obs_queue.size() == 100:
            inference_thread = threading.Thread(target=self.run_inference)
            inference_thread.start()

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

    def run_inference(self):
        """Runs the inference process"""
        with torch.no_grad():
            # set the running_inference flag to True
            self.running_inference = True
            rospy.loginfo("Running inference")

            # fetch the observation data from the patient_obs_queue and action_obs_queue
            obs_data = self.patient_obs_queue.to_numpy()
            # action_data = self.action_obs_queue.to_numpy()

            # concatenate the observation and action data
            # obs_data = np.concatenate((obs_data, action_data), axis=1)
            assert obs_data.shape == (100,4) # the shape of the observation data must be (100,4)

            # convert the observation data to a tensor and add a batch dimension
            obs_data = torch.tensor(obs_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            obs_dict = {'obs': obs_data}
            # calculate the time taken to predict the action
            start = time.time()
            result = self.policy.predict_action(obs_dict)
            end = time.time()
            self.action_prediction_array = result['action'].squeeze(0).to('cpu').numpy()

            # log the time taken to predict the action
            rospy.loginfo(f"Inference Finished. Time taken: {end-start}")

            # set the running_inference flag to False
            self.running_inference = False

if __name__ == '__main__':
    try:
        action_predictor = ActionPredictor()
        rospy.spin()  # Keeps the node running and processing callbacks
    except rospy.ROSInterruptException:
        pass