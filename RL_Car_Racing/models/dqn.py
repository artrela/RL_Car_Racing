import torch
from collections import namedtuple, deque
import random
import numpy as np


class DQNAgent():
    def __init__(self, env, memory_size, batch_size):

        self.total_steps = 0 
        self.current_episode = 0
        self.network_update = 1

        self.epsilon, self.end_exploration = 0.1, 1e6
        
        self.exp_replay = ExperienceReplay(memory_size)
        self._batch_size = batch_size

        self.env = env   # experience replay parameters

        self.q_net = QNetwork(action_space=env.action_space.shape[0])
        self.optim = torch.optim.AdamW(params=self.q_net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, s0, a0, r0, s1, t): 
        
        # store environment interactions into the experience buffer
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)

        self.epsilon = self._epsilonDecay()

        # select a next action with an epsilon greedy strategy
        next_action = self._selectAction()
        
        if self.current_episode % self.network_update == 0:

            experiences = self.exp_replay.getRandomExperiences(self._batch_size)
            
            targets, states = self._prepareMinibatch(experiences)
            
            q_pred = self.q_net(states)
            losses = self.loss_fn(targets, q_pred)

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()
        
        self._trackProgress(t)
        
        return next_action
    
    def _prepareMinibatch(self, experiences):
        
        breakpoint()
        targets = torch.concat(experiences.actions, dim=0)
    

    def _epsilonDecay(self):
        return (1 - self.epsilon) / self.end_exploration * self.current_episode


    def _selectAction(self):
        
        P = random.random()
        if P < 1 - self.epsilon:
            with torch.no_grad(): 
                state = self.exp_replay.getCurrentExperience().state
                actions = self.q_net(state.to(self.q_net.device)) 
                action = torch.argmax(actions)
        else:
            action = torch.tensor(self.env.action_space.sample()).float()

        return action
    

    def _trackProgress(self, episode_end):

        if episode_end:
            self.current_episode += 1 
        
        self.total_steps += 1

        return
        

class QNetwork(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =  torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=13824, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_space)
        ]).to(self.device)

    def forward(self, x):
        return self.model(x)


class ExperienceReplay():
    def __init__(self, memory_length):

        self.Experience = namedtuple("Experience", ["state", "reward", "action", "next_state", "terminal"])
        self._replay_memory = deque(maxlen=memory_length)

    def storeExperience(self, s0, a0, r0, s1, t):

        if len(self._replay_memory) == self._replay_memory.maxlen:
            self._replay_memory.popleft()

        new_experience = self.Experience(s0, a0, r0, s1, t)
        self._replay_memory.append(new_experience)

        return

    def getRandomExperiences(self, batch_size):
        return random.sample(self._replay_memory, k=batch_size)
    
    def getCurrentExperience(self):
        return self._replay_memory[-1]

