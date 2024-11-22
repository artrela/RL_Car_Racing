import gymnasium as gym
from models.dqn import DQNAgent
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, TransformObservation, ClipAction, numpy_to_torch
import torch

NUM_EPISODES = int(1000)
MEM_LEN = int(1e5)
SEED = 1
BATCH_SIZE = 64

def train(config: str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False)
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    # env = numpy_to_torch.NumpyToTorch(env, device='cpu')
    env = TransformObservation(env, lambda x: torch.tensor(x).float(), env.observation_space)
    # env = TransformObservation(env, lambda x: torch.permute(x, (2, 0, 1)).float(), env.observation_space)

    agent = DQNAgent(env, memory_size=MEM_LEN, batch_size=BATCH_SIZE)
    print("Using device:", device)

    agent.fillMemory()

    action = env.action_space.sample()
    # action = torch.tensor(action)

    for e in range(NUM_EPISODES):

        print("Starting episode:", e+1, "/", NUM_EPISODES)
        prev_observation, info = env.reset(seed=SEED)

        terminated = truncated = False

        while not (terminated or truncated):

            observation, reward, terminated, truncated, info = env.step(action)

            if action == 3:
                reward += 10
            elif action == 0:
                reward -= 10

            action = agent(prev_observation, action, reward, observation, terminated or truncated)
            
            if terminated or truncated:
                print(15*"=", "Episode End", 15*"=")
                continue

            prev_observation = observation.detach().clone()

    env.close()

if __name__ == "__main__":
    train("")