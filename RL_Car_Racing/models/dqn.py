import torch
from collections import namedtuple, deque
import random
import numpy as np
import time
import gc
from utils import WandBLogger

#TODO 
'''
- need to push items to the gpu only when they need to go there
- the experience buffer should be on cpu not gpu
'''


class DQNAgent():
    def __init__(self, env, memory_size, batch_size):
        
        self.episode_decay = 350
        self.total_steps = 1 
        self.episode_steps = 1
        self.current_episode = 1
        self.network_update = 1
        self.target_update  = 5

        self.epsilon = 0.1
        self.gamma = 0.96
        
        self.exp_replay = ExperienceReplay(memory_size)
        self._batch_size = batch_size

        self.env = env   # experience replay parameters
        
        self.q_net = QNetwork(action_space=env.action_space.n)
        self.target_net = QNetwork(action_space=env.action_space.n)

        self.device = self.q_net.device
        self.optim = torch.optim.AdamW(params=self.q_net.parameters(), lr=0.0005)
        self.loss_fn = torch.nn.MSELoss()

        self.rets = []
        self.epi_qs = []
        self.epi_losses = []
        self.epi_selected_actions = [0 for _ in range(5)]
        self.epi_start = time.time()
        self.wb = WandBLogger()    

    def __call__(self, s0, a0, r0, s1, t): 

        self.rets.append(r0)
        
        # store environment interactions into the experience buffer
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)

        epsilon = self._epsilonDecay()

        # select a next action with an epsilon greedy strategy
        next_action = self._selectAction(epsilon)
        
        if self.total_steps % self.network_update == 0:

            experiences = self.exp_replay.getRandomExperiences(self._batch_size)
            
            targets, states = self._prepareMinibatch(experiences) 
            
            q_pred = torch.amax(self.q_net(states), dim=1)
            
            losses = self.loss_fn(targets, q_pred)
            
            self.epi_qs.append(q_pred.detach().cpu().numpy().mean())
            self.epi_losses.append(losses.item())

            self.optim.zero_grad()
            losses.backward()
            self.optim.step()

            del q_pred, targets, states
            gc.collect()
            torch.cuda.empty_cache()
        
        
        self._trackProgress(t, epsilon) 
        
        return next_action

    
    def fillMemory(self):

        s0, _ = self.env.reset()

        while len(self.exp_replay) < self._batch_size:

            a0 = self.env.action_space.sample()
            # assert self.env.action_space.contains(a0)
            # print(a0, type(a0))
            s1, r0, ter, trunc, _ = self.env.step(a0)
            
            if a0 == 3:
                r0 += 10
            elif a0 == 0:
                r0 -= 10
                
            self.epi_selected_actions[a0] += 1
                
            self.exp_replay.storeExperience(s0, a0, r0, s1, ter or trunc)
        
        self.env.reset()

        return

    def _prepareMinibatch(self, experiences):
        
        targets = torch.stack([self.yj(memory.terminal, memory.reward, memory.next_state)
                            for memory in experiences], dim=0).squeeze()#.to(self.device)
        
        states  = torch.stack([memory.state for memory in experiences], dim=0).to(self.device)

        return targets, states
    
    def yj(self, tj, rj, sj):
        
        # breakpoint()
        if tj:
            return torch.tensor(rj).to(self.device)
        else:
            # return (rj.to(self.device) + self.gamma * self.q_net(sj.unsqueeze(0).to(self.device))).squeeze()
            with torch.no_grad():
                Qs = self.target_net(sj.unsqueeze(0).to(self.device)).squeeze().cpu().numpy()
                
            # breakpoint()
                
            return torch.tensor(rj + self.gamma * np.max(Qs)).float().to(self.device)

    def _epsilonDecay(self):
        return max((self.episode_decay - self.current_episode) / self.episode_decay, self.epsilon)

    def _selectAction(self, epsilon):
        
        P = random.random()
        if P > epsilon:
            with torch.no_grad(): 
                state = self.exp_replay.getCurrentExperience().state
                action = self.q_net(state.unsqueeze(0).to(self.device))
                action = torch.argmax(action).detach().cpu().int().numpy()
                # action = self._clipAction(action).squeeze().detach()
        else:
            # action = torch.tensor(self.env.action_space.sample()).int()
            action = self.env.action_space.sample()

        self.epi_selected_actions[action] += 1

        return action
    

    def _clipAction(self, action):
        return torch.clip(action,
                            torch.tensor(self.env.action_space.low).to(self.device),
                            torch.tensor(self.env.action_space.high).to(self.device))
    

    def _trackProgress(self, episode_end, epsilon):

        if episode_end:

            if self.current_episode > 1:

                stats = {
                    "eps": epsilon,
                    "epi_tot_rets": sum(self.rets),
                    "epi_avg_rets": sum(self.rets)/len(self.rets),
                    "epi_dur_seconds": round(time.time() - self.epi_start, 2),
                    "tot_steps": self.total_steps,
                    "epi_avg_q": sum(self.epi_qs)/len(self.epi_qs),
                    "epi_avg_loss": sum(self.epi_losses)/len(self.epi_losses)
                }

                print(f"Episode Times: {stats['epi_dur_seconds']}")
                self.wb.send_log(stats)

            print(f"Steps in Episode: {self.episode_steps}")
            print(f"Actions in Episode: {self.epi_selected_actions}")
            
            self.current_episode += 1 
            self.episode_steps = 0 

            self.epi_start = time.time()
            self.rets = []
            self.epi_selected_actions = [0 for _ in range(5)]

            if self.current_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.total_steps += 1
        self.episode_steps += 1

        if self.total_steps % 1000 == 0 and self.exp_replay.memory_capacity < 100:
            print(f"Memory Capacity: {self.exp_replay.memory_capacity:.2f}%")

        return
        

class QNetwork(torch.nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =  torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
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
        self.memory_length = memory_length

    def storeExperience(self, s0, a0, r0, s1, t):

        if len(self._replay_memory) == self._replay_memory.maxlen:
            self._replay_memory.popleft()

        new_experience = self.Experience(s0, a0, r0, s1, t)
        self._replay_memory.append(new_experience)
        
        self.memory_capacity = len(self._replay_memory) / self.memory_length * 100

        return

    def getRandomExperiences(self, batch_size):
        return random.sample(self._replay_memory, k=batch_size)
    
    def getCurrentExperience(self):
        return self._replay_memory[-1]

    def __len__(self):
        return len(self._replay_memory)

