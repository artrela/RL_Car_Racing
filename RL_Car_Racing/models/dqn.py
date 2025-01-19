# ==============================================================================
# Created by: Alec Trela
# GitHub: https://github.com/artrela
# Description: Classes related to a DQN RL agent
#
# This is included as a part of the MRSD bootcamp, meant to be a primer for students
# entering their first year of the program at Carnegie Mellon University 
# 
# Feel free to use, modify, and share this file. Attribution is appreciated! 
# For more information, visit my GitHub or 
# https://github.com/RoboticsKnowledgebase/mrsd-software-bootcamp.
# ==============================================================================

from collections import namedtuple, deque
from utils import WandBLogger
from typing import List, NamedTuple, Optional
import gc
import gymnasium
import numpy as np
import random
import torch

# ==============================================================================
# DQN Action Space Discretization
#   
# A Deep Q-network helps to approximate a Q-table (for future rewards) for a set 
# of discrete actions, which means a DQN requires a set of finite actions. It was
# found that the given # set of discrete states provided by the Car Racing 
# environment is inadequate for learning. As such it is up to you to create a 
# list of actions that could help the agent learn how to drive. 
# 
# Action (len(3)): [steering, gas, break]
# 1. Steering: A continuous value between [-1, 1] representing the steering 
#    angle of the agent's vehicle.
# 2. Gas: A binary action [0, 1], where 0 represents no acceleration and 1 
#    represents full acceleration.
# 3. Braking: A binary action [0, 1], where 0 represents no braking and 1 
#    represents full braking.
# 
# ==============================================================================
ACTION_SPACE = []

class DQNAgent():
    def __init__(self, env, experiment, log):
        """ A DQN agent for driving the Car Racing environment. 
        """
        
        # ========= your code goes here ===============
        
        
        
        # ========= your code goes here ===============
        
        # Logging related 
        self.logger = WandBLogger(experiment) if log else None
        self.max_tiles: int = 0
        self.current_episode: int = 1
        
        return 
        

    def __call__(self, s0, a0, r0, s1, t):
        """ Given the required information for an experience, take the necessary steps to train the agent. 
        """
        raise NotImplementedError

    
    def fillMemory(self):
        """ A helper function to assist the agent by filling its memory before the first
        episode with random actions. 
        """
        raise NotImplementedError


    def _prepareMinibatch(self, experiences):
        """ Given some experiences, generate a minibatch from them. 
        
        For states and actions, this means loading extraction from the experience and loading 
        to the device. 
        
        Targets must be passed through the target function. 
        """
        raise NotImplementedError
    
    
    def yj(self, tj, rj, sj):
        """ 
        Compute y_j as described in https://arxiv.org/pdf/1312.5602, using the Bellman Equation. 
        If the state is terminal (t_j), then no future rewards can exist, so just return the future reward. 
        
        Otherwise, we must use the Bellman Equation to find the discounted future reward that would be taken
        if you took the best possible action. 
        """
        raise NotImplementedError


    def _epsilonDecay(self):
        """ Linearlly anneal the towards the target epsilon over self.episode_decay episodes. 
        """
        raise NotImplementedError


    def selectAction(self, state):
        """ 
        Select an action. If the state is not provided already, implement an 
        epsilon greedy strategy on the most recent state on the replay buffer. 
        """
        raise NotImplementedError


    def _trackProgress(self, episode_end):
        """ Implement some 'catch all' logic to interface with the wandb logger. 

        Args:
            episode_end (bool): Some statistics should not be sent unless the episode is over, 
            use this as a flag for that. 
        """
        if episode_end:
            self.current_episode += 1
            self.episode_actions = [0 for _ in range(len(self.action_space))]
            
            if self.current_episode > 0 and self.logger:
                
                self.logger.setStatistic("epi_avg_q", self.logger.averageStatistic("q_values"))
                self.logger.setStatistic("epi_avg_rets", self.logger.averageStatistic("returns"))
                self.logger.setStatistic("epi_tot_rets", self.logger.sumStatistic("returns"))
                self.logger.setStatistic("epi_avg_loss", self.logger.averageStatistic("losses"))
                self.logger.setStatistic("tiles_visited", self.env.unwrapped.tile_visited_count)

                self.logger.trackStatistic("returns")            
                self.logger.trackStatistic("q_values")            
                self.logger.trackStatistic("losses")    

        self.total_steps += 1
        
        if self.logger:
            self.logger.setStatistic("tot_steps", step=True)
            self.logger.setStatistic("eps", self.epsilon)

        # Number of tiles visited by the agent is a good proxy for learning
        if self.env.unwrapped.tile_visited_count > self.max_tiles:
            raise NotImplementedError

        return
        

class QNetwork(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ExperienceReplay():
    def __init__(self, memory_length):
        """ A memory object for an RL agent. 
        
        The basis for the Bellman Equation is that observations are independent of each other. 
        In many of the gymnasium settings, this assumption does not hold. To help get closer to this assumption, we can 
        pulling randomly from a set of memories to reduce their correlation. 
        """
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminal"])


    def storeExperience(self, s0, a0, r0, s1, t):
        """ Store an experience of the agent. If the memory is full, throw away the oldest one. 

        Args:
            s0 (torch.Tensor): The previous observation
            a0 (int): The action taken to transition from the previous observation to new observation
            r0 (float): The reward for the given action 
            s1 (torch.Tensor): The observation seen as a result of the action 
            t (bool): Was a terminal state reached. 

        """
        raise NotImplementedError
    

    def getRandomExperiences(self, batch_size):
        """Return a list of experiences. May be a good idea to do some prioritzed memory buffer here. 

        Args:
            batch_size (int): Length of lists to return

        Returns:
            List[NamedTuple]: List of memories. 
        """
        raise NotImplementedError

    
    def getCurrentExperience(self):
        raise NotImplementedError
    
    def getCapacity(self):
        raise NotImplementedError
    
    def isFull(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

