# ==============================================================================
# Created by: Alec Trela
# GitHub: https://github.com/artrela
# Description: A set of utilies functions that may be helpful for training an RL 
# agent.
#
# This is included as a part of the MRSD bootcamp, meant to be a primer for students
# entering their first year of the program at Carnegie Mellon University 
# 
# Feel free to use, modify, and share this file. Attribution is appreciated! 
# For more information, visit my GitHub or 
# https://github.com/RoboticsKnowledgebase/mrsd-software-bootcamp.
# ==============================================================================

import gymnasium as gym
import os, torch, wandb, yaml
from typing import Dict, Optional


class WandBLogger:
    def __init__(self, experiment: dict, project_name: str="RL_Car_Racing"):
        """
        A helper class to handle WandB related functionalities. 
        
        Key behaviors include: creating a wandb log, tracking individual numbers, 
        tracking statistics over an episode, and sending the log out when requested. 
        
        Args:
            experiment (dict): A high level dictionary that define an experiment
            project_name (str, optional): The name for the WandB project. Defaults to "RL_Car_Racing".
        """
        self.stats: Dict[str, float] = {
                    "eps": 0.,
                    "tot_steps": 0.,
                    "epi_avg_rets": 0.,
                    "epi_avg_q": 0.,
                    "epi_avg_loss": 0.,
                    "tiles_visited": 0.,
                    "eval_tiles_visited": 0.
        }
        self.epi_stats: Dict[str, list] = {
                    "returns": [],
                    "q_values": [],
                    "losses": []
        }
        
        wandb.login()
        self.run = wandb.init(project=project_name,
                        name = experiment['name'],
                        config = experiment['params'])
        
        
    def setStatistic(self, name: str, val: float=1, step: bool=False)->None:
        """ For the statistics given in the __init__ function, which will be send to 
        WandB logs at a self.sendLog() call, update the value. 

        Args:
            name (str): The key to update in the statistics tracker
            val (float, optional): What to set the statistic to. Defaults to 1.
            step (bool, optional): Should I step the statistic or override it? 
            A val of 1 and a step==True will increment self.stats[name] += 1. Defaults to True.

        Raises:
            KeyError: If the statistic is not existing, then throw and error showing which
            stats are tracked
        """
        if name not in self.stats.keys():
            raise KeyError(f" '{name}' not in statistics tracker! Valid options are {self.stats.keys()}")
        else:
            if step:
                self.stats[name] += val
            else:
                self.stats[name] = val
        return
    
    
    def trackStatistic(self, name: str, val: Optional[float]=None)->None:
        """Meant to track statistics that occur over an episode run. 

        Args:
            name (str): The key to update in the *episode* statistics tracker 
            val (float | list, optional): If a float value given, append to the list for 
            tracking episode statistics. If a list is given, override the dictionary value. 
            Provide no value to reset the list at the given key. Defaults to [].

        Raises:
            KeyError: If the statistic is not existing, then throw and error showing which
            stats are tracked
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.stats.keys()}")
        else:
            if val is not None: 
                try: 
                    val = float(val)
                    self.epi_stats[name].append(val)
                except:
                    raise NotImplementedError(f"No handling for type {type(val)} exists")
            else:
                self.epi_stats[name].clear()
            
        return
    
    def averageStatistic(self, name: str)->float:
        """Calculate the average of a given statistic being tracked over an episode. 

        Args:
            name (str): The key to obtain the average for in the *episode* statistics tracker

        Raises:
            KeyError:  If the statistic is not existing, then throw and error showing which
            stats are tracked

        Returns:
            float: return the average of the statistic
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.stats.keys()}")
        else:
            return sum(self.epi_stats[name]) / len(self.epi_stats[name])
        
    def sumStatistic(self, name: str)->float:
        """Calculate the sum of a given statistic being tracked over an episode. 

        Args:
            name (str): The key to obtain the average for in the *episode* statistics tracker

        Raises:
            KeyError:  If the statistic is not existing, then throw and error showing which
            stats are tracked

        Returns:
            float: return the sum of the statistic
        """
        if name not in self.epi_stats.keys():
            raise KeyError(f" '{name}' not in episode statistics tracker! Valid options are {self.stats.keys()}")
        else:
            return sum(self.epi_stats[name])
            
            
    def sendLog(self)->None:
        """
        Send the log to the wandb server
        """
        wandb.log(self.stats)
        

def parse_config(path: str)->dict:
    """Given a path to a yaml file, return a dictionary object

    Args:
        path (str): A proposed path to a configuration path

    Raises:
        FileNotFoundError: If the file is not found, prints the path given. 

    Returns:
        dict: A parsed configuration yaml file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration not present at {path}!")
    else:
        with open(path, "r") as config_file:
            config_dict = yaml.safe_load(config_file)
            config_file.close()
    
        return config_dict
    
    
def wrap_env(env: gym.Env, experiment_name: str, record_t: int=1)->gym.Env:
    """ Wrap an environment with given wrappers from the Gymnasium library. See 
    https://gymnasium.farama.org/api/wrappers/table/ for a list of wrappers given. 
    
    At minimum, the environment will record every 'record_t' episodes. 
    
    Args:
        env (gymnasium.Env): A gymnasium environment. 
        experiment_name (str): The name to save the recoded videos that are trigger
        record_t (int): If not provided, save each time the env is reset. Useful for evaluation. 
            Defaults to 1.

    Returns:
        gymnasium.Env: A wrapped environment
    """
    
    # place wrapper class calls here
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.TransformObservation(env, lambda x: torch.tensor(x).float(), env.observation_space)
    env = gym.wrappers.TransformObservation(env, lambda x: x[:, :84, :84], env.observation_space)
    # place wrapper class calls here
    
    
    split = "train" if record_t != 1 else "eval"
    env = gym.wrappers.RecordVideo(env, video_folder=f"./videos/{experiment_name}/{split}/", episode_trigger=lambda t: t % record_t == 0, 
                    disable_logger=True)
    
    return env



