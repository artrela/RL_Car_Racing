import gymnasium as gym
from models.dqn import DQNAgent
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, \
                            TransformObservation, ClipAction, numpy_to_torch, RecordVideo
import torch
import numpy as np
from scipy.spatial import KDTree
import math

NUM_EPISODES = int(5000)
MEM_LEN = int(1e5)
SEED = 1
BATCH_SIZE = 64
np.random.seed(1)

'''
Todo:
Write an evaluation function that tracks the target/best network over time
'''

def train(config: str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    env = GrayscaleObservation(env, keep_dim=False) #True)
    env = FrameStackObservation(env, stack_size=4)
    env = TransformObservation(env, lambda x: torch.tensor(x).float(), env.observation_space)
    # env = numpy_to_torch.NumpyToTorch(env, device='cpu')
    # env = TransformObservation(env, lambda x: torch.tensor(x).to('cpu'), env.observation_space)
    # env = TransformObservation(env, lambda x: torch.permute(x, (2, 0, 1)).float(), env.observation_space)
    env = TransformObservation(env, lambda x: x[:, :84, :84], env.observation_space)
    env = RecordVideo(env, video_folder="./videos/deeper-net-3", episode_trigger=lambda t: t % 50 == 0, 
                    disable_logger=True)


    agent = DQNAgent(env, memory_size=MEM_LEN, batch_size=BATCH_SIZE)
    print("Using device:", device)

    agent.fillMemory()
    import matplotlib.pyplot as plt
    
    # _ = env.reset(seed=SEED)
    # track_pts = np.array([[pt[-2], pt[-1]] for pt in env.unwrapped.track])
    # track_tree = KDTree(track_pts)

    # skip_frame = 2
    action = -1 #env.action_space.sample()
    # action = torch.tensor(action)

    for e in range(NUM_EPISODES):

        print("Starting episode:", e+1, "/", NUM_EPISODES)
        prev_observation, info = env.reset(seed=SEED)
        action = -1

        terminated = truncated = False
        consec_neg = 0
        step = 0
        while not (terminated or truncated):
            step += 1
            observation, reward, terminated, truncated, info = env.step(agent.action_space[action])
            
            # for _ in range(skip_frame):
            #     _, r, terminated, truncated, info = env.step(agent.action_space[action])
            #     reward += r
                        
            if step < 30:
                continue
            
            # if action == 0:
            #     reward += 0.1

            if reward < 0:
                consec_neg += 1
                reward = 0
            if reward > 0:
                consec_neg = 0
            
            if consec_neg > 60:
                truncated = True
                info["Continuous Negatives"] = True
            
            # print("Car Position:", env.unwrapped.car.hull.position)
            # print("Closest Track Point:", track_tree.query(np.array(env.unwrapped.car.hull.position))
            # # if track_tree.query(np.array(env.unwrapped.car.hull.position))[0] > 40/3:
            # if track_tree.query(np.array(env.unwrapped.car.hull.position))[0] > 6.6*4:
            #     truncated = True
            #     info["Too Far From Track"] = True
            
            # if consec_neg > 60:
            #     reward -= (consec_neg - 60)
                
            # if track_tree.query(np.array(env.unwrapped.car.hull.position))[0] > 6.6*2:
            #     reward = -0.1
                
            # reward = max(reward, -100)
            
            action = agent(prev_observation, action, reward, observation, terminated or truncated)
                                    
            if terminated or truncated:
                print("Finished Lap:", info)
                # fig, ax = plt.subplots(2, 2)
                # plt.title("Step" + str(step))
                # ax.imshow(observation.squeeze(), cmap='gray')
                # for idx, a in enumerate(ax.ravel()):
                    # a.imshow(observation[idx], cmap='gray')
                # plt.show()
                
                print(15*"=", "Episode End", 15*"=")
                continue

            prev_observation = observation.detach().clone()

    env.close()

if __name__ == "__main__":
    train("")