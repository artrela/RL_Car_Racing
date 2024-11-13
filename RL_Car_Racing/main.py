import gymnasium as gym

NUM_EPISODES = 100

def train(config: str):

    env = gym.make("CarRacing-v3", render_mode="rgb_array") 

    agent = DQN()

    observation, info = env.reset(seed=42)

    for e in range(NUM_EPISODES):

        # this is where you would insert your policy
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()