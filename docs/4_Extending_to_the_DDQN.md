# Part 4: Extending to the DDQN

## 4.0 Predicting target values...and regular q values?
If you have completed the DQN agent, you may also be asking this question as I was. Why would I update my Q-network and also compute targets for it, seem like a moving target (pun intended). From a numerical standpoint this means DQN are prone to overestimation of Q values. Others noticed this as well. Their idea was to introduce a second Q-network to stabilize training in the first. Every so often, let's update the target function. 

## 4.1 Implementation
You have already done most of the work, as the DDQN is a simple extension over the DQN. Travel to `RL_Car_Racing/models/ddqn.py`. Here you will be leveraging the advantages of OOP meaning you will reimplement methods from the base class. If you don't know what the `super()` keyword does in python, I would recommend learning ([example resource](https://www.geeksforgeeks.org/python-super/)). 

When combining the target network, one can either do a soft update (weighted sum with current weights and new weights) or a hard update (replace weights). 

## 4.2 Hints
These are only meant to help if you are stuck. Attempt to learn the core algorithm before looking into these. 

<details>
<summary>4.2.1 Great Article w/ Implementation Details; only look if you want the answers</summary>

[Double Deep Q Networks
Tackling maximization bias in Deep Q-learning](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

</details>

## Sources

```
@misc{vanhasselt2015deepreinforcementlearningdouble,
      title={Deep Reinforcement Learning with Double Q-learning}, 
      author={Hado van Hasselt and Arthur Guez and David Silver},
      year={2015},
      eprint={1509.06461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1509.06461}, 
}
```