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
ACTION_SPACE: List[np.ndarray] = [
        np.array([    0,   1,    0]), # all gas, no break
        np.array([    0, 0.8,    0]), # all gas, no break
        np.array([   -1,   0,    0]), # hard left
        np.array([    1,   0,    0]), # hard right
        np.array([-0.67,   0,    0]), # soft left
        np.array([ 0.67,   0,    0]), # soft left
        np.array([-0.33,   0,    0]), # soft right
        np.array([ 0.33,   0,    0]), # soft right
        np.array([    0,   0,  1.0]), # break left
        np.array([    0,   0, 0.67]), # break left
        np.array([    0,   0,  0.3]), # break right
        np.array([    0,   0,    0])  # do nothing
    ]

class DQNAgent():
    def __init__(self, env: gymnasium.Env, experiment: dict, log: bool):
        """ A DQN agent for driving the Car Racing environment. 

        Args:
            env (gymnasium.Env): Car racing environment
            experiment (dict): A set of hyper parameters used to define the agent's characteristics
            log (bool): Whether or not to log the agent's results in WandB
        """
        self.current_episode:int = 0
        self.total_steps: int = experiment['params']['start_skip'] # start after skipping
        self.start_skip: int = experiment['params']['start_skip']
        self.episode_decay: int = experiment['params']['episode_decay']
        self.network_update: int = experiment['params']['step_update']
        self.target_update: int  = experiment['params']['target_update']
        self.epsilon_final: float = experiment['params']['epsilon']
        self.epsilon: float = 1.
        self.gamma: float = experiment['params']['gamma']
        self.lr: float = experiment['params']['learning_rate']
        self.exp_replay = ExperienceReplay(experiment['params']['mem_len'])
        self.batch_size = experiment['params']['batch_size']
        self.seed = experiment['params']['random_seed']
        self.env = env
        self.action_space = ACTION_SPACE
        self.episode_actions = [0 for _ in range(len(self.action_space))]
        
        # Set up net (for training), target net (for stability), and best net (for testing)
        self.q_net = QNetwork(action_space=len(self.action_space))
        self.target_net = QNetwork(action_space=len(self.action_space))
        self.best_net = QNetwork(action_space=len(self.action_space))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.best_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()    
        self.best_net.eval()    
        self.device = self.q_net.device
        self.optim = torch.optim.AdamW(params=self.q_net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

        # Logging related 
        self.logger = WandBLogger(experiment) if log else None
        self.max_tiles = 0
        
        return 
        

    def __call__(self, s0: torch.Tensor, a0: int, r0: float, s1: torch.Tensor, t: bool)->int:
        """ Given the required information for an experience, take the necessary steps to train the agent. 
        
        1. Store the new experience
        2. Train the model
        3. Select the next action you should take
        4. Reduce exploration via epsilon decay

        Args:
            s0 (torch.Tensor): The previous observation
            a0 (int): The action taken to transition from the previous observation to new observation
            r0 (float): The reward for the given action 
            s1 (torch.Tensor): The observation seen as a result of the action 
            t (bool): Was a terminal state reached. 

        Returns:
            int: What is the next action we should take, given our epsilon greedy strategy?
        """
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)
        
        if self.total_steps % self.network_update == 0:
            
            experiences = self.exp_replay.getRandomExperiences(self.batch_size)
            targets, actions, states = self._prepareMinibatch(experiences) 
            
            rows = torch.arange(self.batch_size).to(self.device)
            q_pred = self.q_net(states)[rows, actions]
            
            losses = self.loss_fn(targets, q_pred)
            
            print("Updating:", losses.item(), q_pred.detach().cpu().numpy().mean())
            if self.logger:
                self.logger.trackStatistic("q_values", q_pred.detach().cpu().numpy().mean())
                self.logger.trackStatistic("losses", losses.item())

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

            del q_pred, targets, states
            gc.collect()
            torch.cuda.empty_cache()
        
        if self.logger:
            self.logger.trackStatistic('returns', r0)
        
        next_action = self.selectAction()
        self._epsilonDecay()
        self._trackProgress(t) 
        
        return next_action

    
    def fillMemory(self):
        """ A helper function to assist the agent by filling its memory before the first
        episode with random actions. 
        """
        s0, _ = self.env.reset(seed=self.seed)

        step, a0 = 0, 0
        while len(self.exp_replay) < self.batch_size:
            
            step += 1
            s1, r0, ter, trunc, _ = self.env.step(self.action_space[a0])
            if step < self.start_skip:
                continue
            
            a0 = random.randrange(0, len(self.action_space))
            self.exp_replay.storeExperience(s0, a0, r0, s1, ter or trunc)
        
        self.env.reset(seed=self.seed)

        return
    

    def _prepareMinibatch(self, experiences: List[NamedTuple]):
        """ Given some experiences, generate a minibatch from them. 
        
        For states and actions, this means loading extraction from the experience and loading 
        to the device. 
        
        Targets must be passed through the target function. 

        Args:
            experiences (List[namedtuple]): Includes entries from 
                ["state", "action", "reward", "next_state", "terminal"]

        Returns:
            tuple: a return of targets, actions, and states
        """
        targets = torch.stack([self.yj(memory.terminal, memory.reward, memory.next_state)
                            for memory in experiences], dim=0).squeeze().to(self.device)
        
        states  = torch.stack([memory.state for memory in experiences], dim=0).to(self.device)
        actions  = torch.tensor(np.array([memory.action for memory in experiences])).to(self.device)

        return targets, actions, states
    
    
    def yj(self, tj: bool, rj: float, sj: torch.Tensor)->torch.Tensor:
        """ 
        Compute y_j as described in https://arxiv.org/pdf/1312.5602, using the Bellman Equation. 
        If the state is terminal (t_j), then no future rewards can exist, so just return the future reward. 
        
        Otherwise, we must use the Bellman Equation to find the discounted future reward that would be taken
        if you took the best possible action. 

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


    def _epsilonDecay(self)->None:
        """ Linearlly anneal the towards the target epsilon over self.episode_decay episodes. 
        """
        self.epsilon = max((self.episode_decay - self.current_episode) / self.episode_decay, self.epsilon_final)
        return


    def selectAction(self, state: Optional[torch.Tensor]=None)->int:
        """ 
        Select an action. If the state is not provided already, implement an 
        epsilon greedy strategy on the most recent state on the replay buffer. 

        Args:
            state (Optional[torch.Tensor], optional): A given state to 
            run the best policy on. Defaults to None.

        Returns:
            int: The chosen action
        """
        
        if state is not None: 
            action = self.q_net(state.unsqueeze(0).to(self.device))
            return torch.argmax(action).detach().cpu().int().item()
        
        P = random.random()
        if P > self.epsilon:
            with torch.no_grad(): 
                state = self.exp_replay.getCurrentExperience().state
                action = self.q_net(state.unsqueeze(0).to(self.device))
                action = torch.argmax(action).detach().cpu().int().item()
        else:
            action = random.randrange(0, len(self.action_space))
            
        self.episode_actions[action] += 1
            
        return action


    def _trackProgress(self, episode_end:bool)->None:
        """ Implement some 'catch all' logic to interface with the wandb logger. 

        Args:
            episode_end (bool): Some statistics should not be sent unless the episode is over, 
            use this as a flag for that. 
        """
        if episode_end:
            self.current_episode += 1
            print("[TRAIN] | Actions Taken:", self.episode_actions)
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
                
            if self.current_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        self.total_steps += 1
        
        if self.logger:
            self.logger.setStatistic("tot_steps", step=True)
            self.logger.setStatistic("eps", self.epsilon)

        if self.env.unwrapped.tile_visited_count > self.max_tiles:
            self.env.unwrapped.tile_visited_count = self.max_tiles
            self.best_net.load_state_dict(self.q_net.state_dict())

        return
        

class QNetwork(torch.nn.Module):
    def __init__(self, action_space: int):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =  torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2592, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=action_space)
        ]).to(self.device)

    def forward(self, x):
        return self.model(x)


class ExperienceReplay():
    def __init__(self, memory_length):
        """ A memory object for an RL agent. 
        
        The basis for the Bellman Equation is that observations are independent of each other. 
        In many of the gymnasium settings, this assumption does not hold. To help get closer to this assumption, we can 
        pulling randomly from a set of memories to reduce their correlation. 

        Args:
            memory_length (int): size of the memory
        """
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminal"])
        self._replay_memory = deque(maxlen=memory_length)
        self.memory_length = memory_length


    def storeExperience(self, s0: torch.Tensor, a0: int, r0: float, s1: torch.Tensor, t: bool)->None:
        """ Store an experience of the agent. If the memory is full, throw away the oldest one. 

        Args:
            s0 (torch.Tensor): The previous observation
            a0 (int): The action taken to transition from the previous observation to new observation
            r0 (float): The reward for the given action 
            s1 (torch.Tensor): The observation seen as a result of the action 
            t (bool): Was a terminal state reached. 

        """
        if len(self._replay_memory) == self._replay_memory.maxlen:
            self._replay_memory.popleft()

        new_experience = self.Experience(s0, a0, r0, s1, t)
        self._replay_memory.append(new_experience)
        
        return
    

    def getRandomExperiences(self, batch_size: int)->List[NamedTuple]:
        """Return a list of experiences. May be a good idea to do some prioritzed memory buffer here. 

        Args:
            batch_size (int): Length of lists to return

        Returns:
            List[NamedTuple]: List of memories. 
        """
        return random.choices(self._replay_memory, 
                            weights=[i+1 for i in range(len(self._replay_memory))], 
                            k=batch_size)
    
    def getCurrentExperience(self):
        return self._replay_memory[-1]
    
    def getCapacity(self):
        return len(self._replay_memory) / self.memory_length * 100
    
    def isFull(self):
        return len(self._replay_memory) == self.memory_length

    def __len__(self):
        return len(self._replay_memory)

