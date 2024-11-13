import gymnasium as gym
import pytest
import torch


@pytest.mark.gpu
def test_gpu():
    '''
    Ensure that pytorch can access a gpu, so that training is feasible
    '''
    print("Your GPU is not accessable, this will need to be resolved to make local testing feasible!")
    assert torch.cuda.is_available()

@pytest.mark.gym_install 
def test_gym_install():
    '''
    Follows the simple gymnasium script given at the homepage (https://gymnasium.farama.org/)
    to ensure users have correctly setup the environment
    '''
    env = gym.make("CarRacing-v3", render_mode="human") 

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for _ in range(100):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()