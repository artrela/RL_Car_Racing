# RL_Car_Racing
An introduction to common reinforcement methods (RL) leveraging Gymnasium (formerly OpenAI Gym), as a featured part of Carnegie Mellon University's MRSD Summer Software Bootcamp. 

## Learning Objectives:
1. Introduction to RL & Algorithsm  
    1.1 Deep Q-Learning (DQN)  
    1.2 Double DQN (DDQN)  
    1.3 Proximal Policy Optimization (PPO) - `COMING SOON`
2. Getting started with and understanding Gymnasium (previously OpenAI Gym)
3. Getting exposure to some MLOps tools 
    3.1 Weights and Biases (WandB)  
    3.2 Data Version Control (DVC) - `COMING SOON`  

## Assignment Output: 

When completed, you should obtain an RL agent that learned to drive completely on its own
accord. The agent below was trained using a DDQN model with nothing but images of the environment
and a set of actions to choose from. 

<figure>
    <img src="docs/assets/ddqn-agent.gif"
         alt="ddqn-agent">
    <figcaption><i>DDQN agent trained from scratch on the Gymnasium Car Racing Environment</i></figcaption>
</figure>

## Getting Started
Head over to `docs` to get started!

Note that since this is an assigment, there are solutions which can be found at the `solutions` branch. However, the solution to this assignment is not unique, there are likely countless ways to obtain a driving agent. I would urge you to attempt as much of this assignment on your own as possible and only resort to the solutions after countless attempts. 

If you are looking for an even harder approach of this problem, head to the `hard` branch where I remove informative docstrings and typehinting which requires you to really know your stuff. 

