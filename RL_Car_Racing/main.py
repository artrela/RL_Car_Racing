import gymnasium as gym
from models.dqn import DQNAgent
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, \
                            TransformObservation, ClipAction, numpy_to_torch, RecordVideo
import torch
import numpy as np

NUM_EPISODES = int(1000)
MEM_LEN = int(1e5)
SEED = 1
BATCH_SIZE = 64

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
    env = RecordVideo(env, video_folder="./model_out", episode_trigger=lambda t: t % 50 == 0, 
                    disable_logger=True)


    agent = DQNAgent(env, memory_size=MEM_LEN, batch_size=BATCH_SIZE)
    print("Using device:", device)

    agent.fillMemory()
    import matplotlib.pyplot as plt

    action = -1 #env.action_space.sample()
    # action = torch.tensor(action)

    for e in range(NUM_EPISODES):

        print("Starting episode:", e+1, "/", NUM_EPISODES)
        prev_observation, info = env.reset(seed=SEED)
        # breakpoint()

        terminated = truncated = False
        # ep_reward = 0.
        step = 0
        while not (terminated or truncated):
            step += 1
            observation, reward, terminated, truncated, info = env.step(agent.action_space[action])

            if step < 30:
                # action = np.zeros(3)
                continue
            # print(step)
            
                # continue
            # ep_reward += reward
            # if action == 3:
            #     reward += 10
            # elif action == 0:
            #     reward -= 10
            
            # if step % 20 == 0:
            #     fig, ax = plt.subplots(2, 2)
            #     plt.title("Step" + str(step))
            #     # ax.imshow(observation.squeeze(), cmap='gray')
            #     for idx, a in enumerate(ax.ravel()):
            #         a.imshow(observation[idx], cmap='gray')
            #     plt.show()
                
            
            # print("Step:", step, "Action:", action, "Reward:", reward, "Observation:", observation.shape)

            action = agent(prev_observation, action, reward, observation, terminated or truncated)
            # if step < 500:
                # action = 3
            
            if terminated or truncated:
                # print(info, truncated, terminated)
                # finished_lap = True if terminated and (not truncated and not info) else False
                # print(env.env.tile_visited_count)
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