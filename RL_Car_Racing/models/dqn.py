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

# 0: steering [-1, 1]
# 1: gas [0, 1]
# 2: braking [0, 1]
action_space = [
        np.array([   0,   1,   0]), # all gas, no break
        np.array([  -1,   1,   0]), # hard left
        np.array([   1,   1,   0]), # hard right
        np.array([-0.5,   1,   0]), # soft left
        np.array([ 0.5,   1,   0]), # soft right
        np.array([   1,   1, 0.3]), # drift left
        np.array([  -1,   1, 0.3]), # drift right
        np.array([   1,   0, 0.3]), # break left
        np.array([  -1,   0, 0.3]), # break right
        np.array([   0,   0,   0])  # do nothing
    ]

class DQNAgent():
    def __init__(self, env, memory_size, batch_size):
        
        self.episode_decay = 200
        self.total_steps = 29 # offset with starting frame start skip 
        self.episode_steps = 1
        self.current_episode = 1
        self.network_update = 10
        self.target_update  = 2

        self.epsilon = 0.1
        self.gamma = 0.95
        
        self.exp_replay = ExperienceReplay(memory_size)
        self._batch_size = batch_size

        self.env = env   # experience replay parameters
        
        self.action_space = action_space
        
        self.q_net = QNetwork(action_space=len(self.action_space))
        self.target_net = QNetwork(action_space=len(self.action_space))
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.device = self.q_net.device
        self.optim = torch.optim.AdamW(params=self.q_net.parameters(), lr=0.0005)
        self.loss_fn = torch.nn.MSELoss()

        self.rets = []
        self.epi_qs = []
        self.epi_losses = []
        self.epi_selected_actions = [0 for _ in range(len(self.action_space))]
        self.epi_start = time.time()
        self.wb = WandBLogger()    

    def __call__(self, s0, a0, r0, s1, t): 

        self.rets.append(r0)
        
        # store environment interactions into the experience buffer
        # print("Object Call:", s0.shape, a0, r0, s1.shape, t)
        self.exp_replay.storeExperience(s0, a0, r0, s1, t)

        epsilon = self._epsilonDecay()

        # select a next action with an epsilon greedy strategy
        next_action = self._selectAction(epsilon)
        
        if self.total_steps % self.network_update == 0:

            experiences = self.exp_replay.getRandomExperiences(self._batch_size)
            
            targets, actions, states = self._prepareMinibatch(experiences) 
            
            # q_pred = torch.amax(self.q_net(states), dim=1)
            # breakpoint()
            rows = torch.arange(self._batch_size).to(self.device)
            q_pred = self.q_net(states)[rows, actions]
            
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

        s0, _ = self.env.reset(seed=1)

        step, a0 = 0, 0
        while len(self.exp_replay) < self._batch_size:
            
            step += 1
            
            s1, r0, ter, trunc, _ = self.env.step(self.action_space[a0])
            
            if step < 30:
                continue
            
            a0 = random.randint(0, len(self.action_space)-1)#self.env.action_space.sample()
                            
            self.epi_selected_actions[a0] += 1
            
            # print("Fill Mem:", s0.shape, a0, r0, s1.shape, ter or trunc)
            self.exp_replay.storeExperience(s0, a0, r0, s1, ter or trunc)
        
        self.env.reset(seed=1)

        return

    def _prepareMinibatch(self, experiences):
        
        # for e in experiences:
        # l = [self.yj(memory.terminal, memory.reward, memory.next_state)
        #                     for memory in experiences]
        #     print(f"{e.reward=}, {e.next_state.shape=}")
        # breakpoint()
        targets = torch.stack([self.yj(memory.terminal, memory.reward, memory.next_state)
                            for memory in experiences], dim=0).squeeze()#.to(self.device)
        
        states  = torch.stack([memory.state for memory in experiences], dim=0).to(self.device)
        actions  = torch.tensor(np.array([memory.action for memory in experiences])).to(self.device)
        # breakpoint()

        return targets, actions, states
    
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
                # print(P, self.episode_steps, torch.argmax(action).detach().cpu().int().numpy(), action)
                action = torch.argmax(action).detach().cpu().int().numpy()
                # action = self._clipAction(action).squeeze().detach()
        else:
            # action = torch.tensor(self.env.action_space.sample()).int()
            action = random.randint(0, len(self.action_space)-1)
            

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
                    "epi_avg_loss": sum(self.epi_losses)/len(self.epi_losses),
                    "tiles_visited": self.env.unwrapped.tile_visited_count,
                    "percent_do_nothing": self.epi_selected_actions[-1]/sum(self.epi_selected_actions) * 100
                }

                print(f"Episode Times: {stats['epi_dur_seconds']}")
                self.wb.send_log(stats)

            print(f"Steps in Episode: {self.episode_steps}")
            print(f"Actions in Episode: {self.epi_selected_actions}")
            print(f"Tiles Visited: {self.env.unwrapped.tile_visited_count}")
            
            self.current_episode += 1 
            self.episode_steps = 29

            self.epi_start = time.time()
            self.rets = []
            self.epi_selected_actions = [0 for _ in range(len(self.action_space))]

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
            # torch.nn.Linear(in_features=3200, out_features=256),
            torch.nn.Linear(in_features=2592, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_space)
        ]).to(self.device)

    def forward(self, x):
        return self.model(x)


class ExperienceReplay():
    def __init__(self, memory_length):

        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "terminal"])
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

