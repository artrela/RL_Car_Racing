import torch
from collections import namedtuple, deque
import random
import numpy as np


class DQNAgent():
    def __init__(self, env, memory_length, exp_size):

        self.total_steps = 0 
        self.current_episode = 0
        self.network_update = 1

        self.epsilon, self.end_exploration = 0.1, 1e6
        
        self.exp_replay = ExperienceReplay(memory_length, exp_size)
        
        self.env = env   # experience replay parameters

        self.QNetwork = QNetwork(action_space=env.action_space.shape)
        self.optim = torch.optim.AdamW(params=self.QNetwork.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, s0, a0, r0, s1, t): 
        
        # store environment interactions into the experience buffer
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)

        # select a next action with an epsilon greedy strategy
        next_action = self._selectAction(self.epsilon)
        
        if self.current_episode % self.network_update == 0:

            experiences = self.exp_replay.getRandomExperiences()
            
            targets, states = self._prepareMinibatch(experiences)
            
            q_pred = self.QNetwork(states)
            losses = self.loss_fn(targets, q_pred)

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()
        
        self._trackProgress(t)
        
        return next_action
    
    def _prepareMinibatch(self, experiences):
        pass
    

    def _epsilonDecay(self):
        
        new_eps = (1 - self.eps) / self.end_exploration * self.current_episode
        return new_eps


    def _selectAction(self, eps):
        
        P = random.random()
        if P < 1 - eps:
            with torch.no_grad(): 
                state = self.exp_replay.getCurrentExperience()
                actions = self.QNetwork(state.to(self.QNetwork.device)) 
                action = torch.argmax(actions)
        else:
            action = self.env.action_space.sample()

        return action
    

    def _trackProgress(self, episode_end):

        if episode_end:
            self.current_episode += 1 
        
        self.total_steps += 1

        return
        

class QNetwork(torch.nn.Module):
    def __init__(self, action_space):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=13824, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_space)
        ).to(self.device)

    def forward(self, x):
        return self.model(x)


class ExperienceReplay():
    def __init__(self, batch_size, memory_length, state_length):

        self.Experience = namedtuple("Experience", ["state", "reward", "action", "next_state", "terminal"])
        self._replay_memory = deque(maxlen=memory_length)

        self._current_state = deque(maxlen=state_length)
        self._next_state = deque(maxlen=state_length)
        self.state_length = state_length

        self._batch_size = batch_size

    def _createStates(self, ss0, ss1):

        self._current_state.append(ss0)
        self._next_state.appendleft(ss1)

        while self._current_state < self.state_length:
            self._current_state.appendleft(torch.zeros_like(ss0))
        
        while self._next_state < self.state_length:
            self._current_state.append(torch.zeros_like(ss1))

        return torch.stack(self._current_state, dim=-1), torch.stack(self._next_state, dim=-1)

    def storeExperience(self, ss0, a0, r0, ss1, t):

        s_in, s_out = self._createStates(ss0, ss1)

        if len(self._replay_memory) == self._replay_memory.maxlen:
            self._replay_memory.popleft()

        new_experience = self.Experience(s_in, a0, r0, s_out, t)
        self._replay_memory.append(new_experience)

        return

    def getRandomExperiences(self):
        return random.sample(self._replay_memory, k=self._batch_size)
    
    def getCurrentExperience(self):
        return self._replay_memory[-1]

