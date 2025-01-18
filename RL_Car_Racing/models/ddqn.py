from dqn import DQNAgent, QNetwork
import gymnasium as gym
import torch
import numpy as np


class DDQNAgent(DQNAgent):
    def __init__(self, env: gym.Env, experiment: dict, log: bool):
        super().__init__(env, experiment, log)
        
        self.target_update: int  = experiment['params']['target_update']
        
        self.target_net = QNetwork(action_space=len(self.action_space))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()    

    def yj(self, tj: bool, rj: float, sj: torch.Tensor)->torch.Tensor:
        """ 
        Compute y_j as described in https://arxiv.org/pdf/1312.5602, using the Bellman Equation. 
        If the state is terminal (t_j), then no future rewards can exist, so just return the future reward. 
        
        Otherwise, we must use the Bellman Equation to find the discounted future reward that would be taken
        if you took the best possible action. 
        
        Different than the DQN update, now we use target net to predict Q-Values! https://arxiv.org/pdf/1509.06461


        Args:
            tj (bool): Is the state terminal
            rj (float): Current reward for the experience
            sj (torch.Tensor): The current state for which the reward was observerd. 

        Returns:
            torch.Tensor: The Q value for the given experience context
        """
        if tj:
            return torch.tensor(rj)
        else:
            with torch.no_grad():
                Qs = self.target_net(sj.unsqueeze(0).to(self.device)).squeeze().cpu().numpy()
                
            return torch.tensor(rj + self.gamma * np.max(Qs)).float()

    def _trackProgress(self, episode_end:bool)->None:
        """ Implement some 'catch all' logic to interface with the wandb logger. 
        
        As opposed to the DQN base class, this model also updates the target model
        every so often. 

        Args:
            episode_end (bool): Some statistics should not be sent unless the episode is over, 
            use this as a flag for that. 
        """
        
        super()._trackProgress(episode_end)
        
        if episode_end:    
            if self.current_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return
        