from ..RL_Car_Racing.models import dqn
from ..RL_Car_Racing.utils import parse_config
import gymnasium as gym
import pytest
import random
import torch
import pickle

"""
- Does the function yj work for terminal & non-terminal states?
- does selecting an action 1. provide a return for a state estimate 2. 
    give a random action when its suppose to, 3. give a non random answer otherwise
- can you forward prop your network as needed
"""

@pytest.mark.dqn
def test_epsilon_decay():
    '''
    Ensure decay call degrades epsilon as intended
    '''
    
    config = parse_config("../RL_Car_Racing/config/default.yaml")
    agent = dqn.DQNAgent(gym.Env(), config, log=False)

    with open("tests/assets/decay_vals.pkl", "rb") as f:
        decay_values = pickle.load(f)
    
    agent.episode_decay = len(decay_values)
    
    for correct_epsilon in decay_values:
        assert agent.epsilon == correct_epsilon
        agent._epsilonDecay()
        

@pytest.mark.exp_replay
def test_exp_store():
    '''
    Ensures memory works to grow to a certain size and remove as needed
    '''
    
    memory_length = random.randint(5, 20)
    
    exp_replay = dqn.ExperienceReplay(memory_length=memory_length)
    
    random_experiences = [ exp_replay.Experience(
        torch.rand(size=5), # state, 
        random.randint(5, 20), # action, 
        random.random() * random.randint(-2, 2), # reward
        torch.rand(size=5), # next state
        True if random.rand() > 0.5 else False # terminal
    ) for _ in range(memory_length * 2) ]
    
    for idx, exp in enumerate(random_experiences):
        
        exp_replay.storeExperience(exp)
        
        assert len(exp_replay) <= memory_length

        if idx > memory_length: 
            
            stored_experiences = list(exp_replay._replay_memory)
            expected_experiences = random_experiences[idx - memory_length + 1 : idx + 1]
            
            for stored, expected in zip(stored_experiences, expected_experiences):
                assert torch.equal(stored.state, expected.state)
                assert stored.action == expected.action
                assert stored.reward == expected.reward
                assert torch.equal(stored.next_state, expected.next_state)
                assert stored.terminal == expected.terminal
                
                

            
        
    
    
    
    