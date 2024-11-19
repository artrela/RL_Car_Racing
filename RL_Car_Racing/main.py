import gymnasium as gym
from models.dqn import DQNAgent
    
NUM_EPISODES = 100
MEM_LEN = 1e5
SEED = 1

def train(config: str):

    env = gym.make("CarRacing-v3", obs_type="")

    agent = DQNAgent(env, memory_length=MEM_LEN)

    terminated = truncated = False

    prev_observation, info = env.reset(seed=SEED)

    for e in range(NUM_EPISODES):

        print("Starting episode:", e+1, "/", NUM_EPISODES)

        while not terminated or truncated:

            observation, reward, terminated, truncated, info = env.step(action)

            action = agent(action, prev_observation, observation, reward, terminal=terminated or truncated)
            
            if terminated or truncated:
                prev_observation, info = env.reset(seed=SEED)

            prev_observation = observation.copy()

    env.close()

if __name__ == "__main__":
    train()