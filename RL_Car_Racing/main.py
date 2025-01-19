# ==============================================================================
# Created by: Alec Trela
# GitHub: https://github.com/artrela
# Description: The main training and evaluation protocol for the training RL agents
# on Gymnasium (formerly OpenAI Gym)
#
# This is included as a part of the MRSD bootcamp, meant to be a primer for students
# entering their first year of the program at Carnegie Mellon University 
# 
# Feel free to use, modify, and share this file. Attribution is appreciated! 
# For more information, visit my GitHub or 
# https://github.com/RoboticsKnowledgebase/mrsd-software-bootcamp.
# ==============================================================================

import argparse
import gymnasium as gym

import utils
from models.dqn import DQNAgent
from models.ddqn import DDQNAgent


def main(experiment: dict, debug: bool)->None:
    """
    Train an RL agent to drive in the Gymnasium environment 'CarRacing-v3', using hyperparameters specified in a 
    configuration yaml file. See 'RL_Car_Racing/config/default.yaml' as a starting point.

    Args:
        experiment (dict): A set of parameters to define the experiement. High level markers are 'params' and 
        'name'. 
        debug (bool): Forgoes wandb logging for debugging purposes 
    """

    train_env = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    train_env = utils.wrap_env(train_env, experiment['name'], record_t=experiment['record_video'])
    
    test_env = gym.make("CarRacing-v3", render_mode='rgb_array', domain_randomize=False, continuous=True)
    test_env = utils.wrap_env(test_env, experiment['name'], split="eval", record_t=experiment['record_video'])

    if experiment['params']['model'] == 'DQN':
        agent = DQNAgent(train_env, experiment, log=False if debug else True)
    elif experiment['params']['model'] == 'DDQN':
        agent = DDQNAgent(train_env, experiment, log=False if debug else True)
    else:
        raise NotImplementedError
    
    for e in range(experiment['params']['num_episodes']):

        start_str = 15*"=" + f"Episode {e+1}/{experiment['params']['num_episodes']} Start" + 15*"="
        print(start_str)
        
        train(agent, train_env, 
                start_skip=experiment['params']['start_skip'],
                stacked_neg=experiment['params']['stacked_neg'])
                    
        eval(agent, test_env, experiment['params']['start_skip'])
        
        if agent.logger:
            agent.logger.sendLog()
            
        print("=" * len(start_str), "\n")

    train_env.close()
    test_env.close()
    

def train(agent: DQNAgent, env: gym.Env, start_skip: int, stacked_neg: int)->None:
    """ Train the agent, which may include steps not seen during evaluatoin
    
    Args:
        agent (DQNAgent): The RL agent
        env (gym.Env): The train environment
        start_skip (int): To skip frames at the beginning of the episode
        stack_neg (int): The amount of allowable negative rewards in a row before the
            environment resets. 
    """
    prev_observation, info = env.reset(seed=experiment['params']['random_seed'])
    terminated = truncated = False
    
    action, consec_neg, step = -1, 0, 0
    while not (terminated or truncated):
        
        step += 1
        observation, reward, terminated, truncated, info = env.step(agent.action_space[action])
        if step < start_skip:
            continue
        
        if reward < 0:
            consec_neg += 1
            reward = 0
        else:
            consec_neg = 0
        
        if consec_neg > stacked_neg:
            truncated = True
            info["Continuous Negatives"] = True
        
        action = agent(prev_observation, action, reward, observation, terminated or truncated)
                                
        if terminated or truncated:
            print("[TRAIN] | Info:", info)
            print("[TRAIN] | Steps:", step - start_skip)
            print("[TRAIN] | Tiles Visited:", env.unwrapped.tile_visited_count)
            return

        prev_observation = observation.detach().clone()
        

def eval(agent: DQNAgent, env: gym.Env, start_skip: int)->None:
    """ Evaluate the agent, using the agent that has amassed the highest training tiles during training,
    with no exploration. 

    Args:
        agent (DQNAgent): The trained RL agent
        env (gym.Env): The test environment
        start_skip (int): To skip frames at the beginning of the episode
    """
    
    env.reset(seed=experiment['params']['random_seed'])
    terminated = truncated = False
    
    action, step = -1, 0
    while not (terminated or truncated):
        
        step += 1
        observation, reward, terminated, truncated, info = env.step(agent.action_space[action])
        if step < start_skip:
            continue
        
        action = agent.selectAction(state=observation)
        
        if agent.logger:
            agent.logger.setStatistic('eval_tiles_visited', env.unwrapped.tile_visited_count)
                                
        if terminated or truncated:
            print("[EVAL] | Info:", info)
            print("[EVAL] | Tiles Visited:", env.unwrapped.tile_visited_count)
            return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Running training cycle for RL agent on Car Racing Gymnasium Environment")
    
    parser.add_argument("--config", "-c", default="./RL_Car_Racing/config/default.yaml", type=str,
                    help="Path to yaml file used to establish an experiment.")
    parser.add_argument("--debug", "-d", 
                        help="Removes wandb logging for debugging purposes.", action="store_true")
    
    args = parser.parse_args()  
    
    experiment: dict = utils.parse_config(args.config)
    
    main(experiment, args.debug)