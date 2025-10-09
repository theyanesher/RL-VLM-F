#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
import pickle as pkl
import glob
from collections import defaultdict

from logger import Logger
from replay_buffer import ReplayBuffer

from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts

import utils
import hydra
from PIL import Image

import cv2


class Relabel_class(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.reward = self.cfg.reward # what types of reward to use
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/prompt.py {}/".format(current_file_path, self.logger._log_dir))
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            self.env = utils.make_env(cfg)
        
        self.resize_factor = 1
        if "sweep" in cfg.env or 'drawer' in cfg.env or "soccer" in cfg.env:
            image_height = image_width = 300 
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "CartPole" in cfg.env:
            image_height = image_width = 200
        if "Cloth" in cfg.env:
            image_height = image_width = 360
            
        self.image_height = image_height
        self.image_width = image_width

        # instantiating the reward model
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore
        
        self.reward_model = reward_model_class(
            ### original PEBBLE parameters
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            capacity=cfg.max_feedback * 2,
            
            ### vlm parameters
            vlm_label=cfg.vlm_label,
            vlm=cfg.vlm,
            env_name=cfg.env,
            clip_prompt=clip_env_prompts[cfg.env],
            log_dir=self.logger._log_dir,
            flip_vlm_label=cfg.flip_vlm_label,
            cached_label_path=cfg.cached_label_path,

            ### image-based reward model parameters
            image_reward=cfg.image_reward,
            image_height=image_height,
            image_width=image_width,
            resize_factor=self.resize_factor,
            resnet=cfg.resnet,
            conv_kernel_sizes=cfg.conv_kernel_sizes,
            conv_strides=cfg.conv_strides,
            conv_n_channels=cfg.conv_n_channels,
        )
        
        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, 200) 
                
    
    def relabel(self, data):
        if not self.cfg.image_reward:
            batch_size = 200
        else:
            batch_size = 32
        self.idx = len(data['observations'])
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
        if  "rewards_pred" not in data:
            data["rewards_pred"] = np.empty_like(data["rewards"])
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
            
            if not self.cfg.image_reward:
                obses = data["observations"][index*batch_size:last_index]
                actions = data["actions"][index*batch_size:last_index]
                inputs = np.concatenate([obses, actions], axis=-1)
            else:
                inputs = data["next images"][index*batch_size:last_index]
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = inputs.astype(np.float32) / 255.0

            # Convert numpy array to torch tensor and move to the correct device
            inputs = torch.from_numpy(inputs).float().to(self.device)
            pred_reward = self.reward_model.r_hat_batch(inputs)
            data["rewards_pred"][index*batch_size:last_index] = np.squeeze(pred_reward)
        torch.cuda.empty_cache()        
        
        
@hydra.main(config_path='config/train_PEBBLE_offline.yaml', strict=True)
def main(cfg):
    workspace = Relabel_class(cfg)
    
    data_directory = "/share1/RL-VLM-F/test_dummy/soccer/expert" # for soccer
    input_files = glob.glob(os.path.join(data_directory, '*.pkl'))

    if not input_files:
        print(f"Error: No '.pkl' files found in '{data_directory}'. Please check the path.")
        return

    print(f"Found {len(input_files)} files to process:")
    for f_path in input_files:
        print(f"  - {f_path}")

    # 2. Load and combine data from all found pkl files.
    all_data_lists = defaultdict(list)
    for file_path in input_files:
        print(f"Loading data from: {file_path}")
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
            for key, value in data.items():
                all_data_lists[key].append(value)

    print("\nCombining all loaded data...")
    combined_data = {}
    for key, value_list in all_data_lists.items():
        # Concatenate numpy arrays from each file along the first axis (batch dimension)
        combined_data[key] = np.concatenate(value_list, axis=0)
        print(f"  - Combined shape for '{key}': {combined_data[key].shape}")

    # free memory before next operation
    print("\nFreeing up memory before relabeling...")
    del all_data_lists
    import gc
    gc.collect()

    # relabel the entire combined dataset
    print("\nStarting the relabeling process on the combined data...")
    workspace.relabel(combined_data)
    print("Relabeling complete.")

    # save the combined and relabeled data to a single new pkl file
    output_dir = workspace.work_dir
    output_path = os.path.join('/share1', "replayed_combined_data.pkl")
    print(f"\nSaving new dataset to: {output_path}")
    with open(output_path, "wb") as f:
        # using a higher protocol can be more efficient for large objects
        pkl.dump(combined_data, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("\nProcess finished successfully!")
    print(f"Total entries in the final dataset: {len(combined_data['rewards_pred'])}")

if __name__ == '__main__':
    main()

