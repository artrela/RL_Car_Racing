from dqn import DQNAgent, QNetwork
import gymnasium as gym
import torch
import numpy as np


class DDQNAgent(DQNAgent):
    def __init__(self, env, experiment, log):
        super().__init__(env, experiment, log)
        
        # ====== Your code goes here ========  
        
        
        
        # ====== Your code goes here ========  

    def yj(self, tj, rj, sj):
        """ 
        Compute y_j as described in https://arxiv.org/pdf/1312.5602, using the Bellman Equation. 
        If the state is terminal (t_j), then no future rewards can exist, so just return the future reward. 
        
        Otherwise, we must use the Bellman Equation to find the discounted future reward that would be taken
        if you took the best possible action. 
        
        Different than the DQN update, now we use target net to predict Q-Values! https://arxiv.org/pdf/1509.06461
        """
        raise NotImplementedError

    def _trackProgress(self, episode_end)->None:
        """ Implement some 'catch all' logic to interface with the wandb logger. 
        
        As opposed to the DQN base class, this model also updates the target model
        every so often. 
        """
        
        raise NotImplementedError
        