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
        
        # ====== Your code goes here ========  
        
        
        
        # ====== Your code goes here ========  
        
        if agent.logger:
            agent.logger.sendLog()
            
        print("=" * len(start_str), "\n")

    train_env.close()
    test_env.close()
    

def train(agent: DQNAgent, env: gym.Env)->None:
    """ Train the agent, which may include steps not seen during evaluation
    
    **May need to add another parameter(s) to this function
    
    Args:
        agent (DQNAgent): The RL agent
        env (gym.Env): The train environment
    """
    raise NotImplementedError
        

def eval(agent: DQNAgent, env: gym.Env)->None:
    """ Evaluate the agent, using the agent that has amassed the highest training tiles during training,
    with no exploration. 
    
    **May need to add another parameter to this function

    Args:
        agent (DQNAgent): The trained RL agent
        env (gym.Env): The test environment
    """
    raise NotImplementedError


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Running training cycle for RL agent on Car Racing Gymnasium Environment")
    
    parser.add_argument("--config", "-c", default="./RL_Car_Racing/config/default.yaml", type=str,
                    help="Path to yaml file used to establish an experiment.")
    parser.add_argument("--debug", "-d", 
                        help="Removes wandb logging for debugging purposes.", action="store_true")
    
    args = parser.parse_args()  
    
    experiment: dict = utils.parse_config(args.config)
    
    main(experiment, args.debug)