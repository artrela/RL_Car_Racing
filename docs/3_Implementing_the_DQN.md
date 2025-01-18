# Part 3: Implementing the DQN

## 3.0 Background:
This introduction to DQN's will be brief. If you require a more exhaustive explanation, I recommend starting with the [Hugging Face Tutorials](https://huggingface.co/learn/deep-rl-course/en/unit2/introduction) with a Focus on Unit 2 `Introduction to Q-Learning` and Unit 3 `Deep Q-Learning with Atari Games`. 

Q-Learning is a tabular based RL method which uses the Bellman Equation to find the optimal policy; it looks to find the best set of actions given state-action pairs. 

<br>
V(s) = max_a Σ P(s' | s, a) [ R(s, a, s') + γ * V(s') ]

Where:
- V(s): The value of state s.
- a: Action taken in state s.
- P(s' | s, a): Transition probability to state s' given state s and action a.
- R(s, a, s'): Reward received after transitioning from s to s' via action a.
- γ: Discount factor (0 ≤ γ < 1).
<br>

Take a look at the following example below, which provides an example of tabular Q-Learning to build your intuition on what is actually happening. As we explore new states, we update the Q-Table, which is known as value iteration. For an large number of iterations, we will approach the optimal policy by following the maximum Q values in each state-action transition. 

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*DOv2T74U6C3fd1EoEN7LoA.gif">

Deep Q-Networks (DQNs) are an extension to the generic Q-Learning, which arises as representations for a state become more complex. In the above example, notice how large the table for tracking Q-values is for 6 states. Imagine how many states would need to be stored for our example, which out of the box has an observation space of `Box(0, 255, (96, 96, 3), uint8)`. This expands to `256^(96 * 96 * 3) = 2^221,184` which is far beyond our ability to store. So what do we do? We can use a neural network to approximate the Q-Table! 

## 3.1 Wrapping the Environment

## 3.2 Creating the Q-Network

## 3.3 What is Experience Replay?

## 3.4 Creating a DQN Agent

### 3.4.1 Balancing exploration & exploitation with an Epsilon-Greedy strategy

### 3.4.2 Should the ratio of exploration & exploitation change over time?

### 3.4.3 Training the model with the Bellman Equation

