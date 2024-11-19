import gymnasium as gym
from models.dqn import DQNAgent
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, TransformObservation, ClipAction, numpy_to_torch
import torch
    
NUM_EPISODES = int(1e6)
MEM_LEN = int(1e5)
SEED = 1
BATCH_SIZE = 32

def train(config: str):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("CarRacing-v3")
    env = numpy_to_torch.NumpyToTorch(env, device=device)
    env = TransformObservation(env, lambda x: torch.permute(x, (2, 0, 1)).float(), env.observation_space)

    agent = DQNAgent(env, memory_size=MEM_LEN, batch_size=BATCH_SIZE)
    print("Using device:", device)

    agent.fillMemory()

    terminated = truncated = False

    prev_observation, info = env.reset(seed=SEED)
    # prev_observation = prev_observation.float()

    action = env.action_space.sample()
    action = torch.tensor(action).float().to(agent.q_net.device)

    for e in range(NUM_EPISODES):

        print("Starting episode:", e+1, "/", NUM_EPISODES)

        while not terminated or truncated:

            observation, reward, terminated, truncated, info = env.step(action)

            action = agent(prev_observation, action, reward, observation, terminated or truncated)
            
            if terminated or truncated:
                prev_observation, info = env.reset(seed=SEED)

            prev_observation = observation.detach().clone()

    env.close()

if __name__ == "__main__":
    train("")