import torch
from collections import namedtuple, deque
import random
import numpy as np
import time
from utils import WandBLogger


class DQNAgent():
    def __init__(self, env, memory_size, batch_size):

        self.total_steps = 0 
        self.current_episode = 1
        self.network_update = 100

        self.epsilon = 0.1
        self.gamma = 0.9
        
        self.exp_replay = ExperienceReplay(memory_size)
        self._batch_size = batch_size

        self.env = env   # experience replay parameters

        self.q_net = QNetwork(action_space=env.action_space.shape[0])
        self.device = self.q_net.device
        self.optim = torch.optim.AdamW(params=self.q_net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

        self.rets = []
        self.epi_start = time.time()
        self.wb = WandBLogger()    

    def __call__(self, s0, a0, r0, s1, t): 

        self.rets.append(r0)
        
        # store environment interactions into the experience buffer
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)

        self.epsilon = self._epsilonDecay()

        # select a next action with an epsilon greedy strategy
        next_action = self._selectAction()
        
        if self.total_steps % self.network_update == 0:

            experiences = self.exp_replay.getRandomExperiences(self._batch_size)
            
            targets, states = self._prepareMinibatch(experiences) 
            q_pred = self.q_net(states)
            losses = self.loss_fn(targets, q_pred)

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

            del q_pred, targets
            torch.cuda.empty_cache()
        
        self._trackProgress(t) 
        
        return next_action

    
    def fillMemory(self):

        s0, _ = self.env.reset()

        while len(self.exp_replay) < self._batch_size:

            a0 = torch.tensor(self.env.action_space.sample()).float()
            s1, r0, ter, trunc, _ = self.env.step(a0)
            self.exp_replay.storeExperience(s0, a0, r0, s1, ter or trunc)
        
        self.env.reset()

        return

    def _prepareMinibatch(self, experiences):
        
        targets = torch.stack([self.yj(memory.terminal, memory.reward, memory.next_state) 
                            for memory in experiences], dim=0).squeeze()
        
        states  = torch.stack([memory.state for memory in experiences], dim=0)

        return targets, states
    
    def yj(self, tj, rj, sj):
        
        if tj:
            return rj.to(self.device)
        else:
            return rj.to(self.device) + self.gamma * self.q_net(sj.unsqueeze(0))


    def _epsilonDecay(self):
        return max((1 - self.epsilon) / self.total_steps, self.epsilon)


    def _selectAction(self):
        
        P = random.random()
        if P < 1 - self.epsilon:
            with torch.no_grad(): 
                state = self.exp_replay.getCurrentExperience().state
                action = self.q_net(state.unsqueeze(0)) 
                action = self._clipAction(action).squeeze()
        else:
            action = torch.tensor(self.env.action_space.sample()).float()

        return action
    

    def _clipAction(self, action):
        return torch.clip(action,
                            torch.tensor(self.env.action_space.low).to(self.device),
                            torch.tensor(self.env.action_space.high).to(self.device))
    

    def _trackProgress(self, episode_end):

        if episode_end:

            self.current_episode += 1 

            stats = {
                "eps": self.epsilon,
                "epi_tot_rets": sum(self.rets),
                "epi_avg_rets": sum(self.rets)/len(self.rets),
                "epi_dur_seconds": round(time.time() - self.epi_start, 2)
            }

            self.epi_start = time.time()
            self.rets = []
        
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
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=3200, out_features=256),
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

    def __len__(self):
        return len(self._replay_memory)

