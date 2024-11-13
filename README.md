# RL_Car_Racing
An introduction to common reinforcement methods (RL) leveraging Gymnasium (formerly OpenAI Gym), as a featured part of Carnegie Mellon University's MRSD Summer Software Bootcamp. 

**Learning Objectives:**
1. Introduction to RL & Algorithsm  
    1.1 Deep Q-Learning (DQN)  
    1.2 Double DQN (DDQN)  
    1.3 Proximal Policy Optimization (PPO)
2. Getting started with and understanding Gymnasium (previously OpenAI Gym)
3. Getting exposure to some MlOps tools used in industry  
    3.1 Weights and Biases (WandB)  
    3.2 MLFlow  
    3.3 Data Version Control (DVC)  

## Part 0: Environment Setup

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) and setup an environment:
```bash
conda env create -n rl-mrsd python=3.10 -y 
conda activate rl-mrsd
```

2. Install a cuda enabled distribution of [PyTorch](https://pytorch.org/get-started/locally/) by matching your version of cuda to their download interface. *Note: this assumes you already have a version of cuda downloaded on your platform. If this is not the case, a good place to start is [here](https://developer.nvidia.com/cuda-toolkit-archive).* In my case, with cuda 12.2 I will download the closest stable version. One can verify their version of cuda with a call to `nvidia smi`:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

3. After installing PyTorch, install the rest of the requirements with:
```bash
pip install requirments.txt
# Confirm that installation has been completed successfully by running the following test(s):
pytest -m gpu 
```

## Part 1: Setup Gymnasium & Car Racing Environment

If you have not used Gymnasium before, or would like a more involved explanation of its use case and history, please refer [here](www.fillintheblank.com). For these assignments, we will be using the [Box2D Car Racing Environment](https://gymnasium.farama.org/environments/box2d/car_racing/). A sample for what this environment looks like can be found below, along with installation instructions.


<p align="center">
    <img src=https://gymnasium.farama.org/_images/car_racing.gif alt=car-racing width=400 height=300/> 
</p>



```bash
pip install swig
pip install gymnasium[box2d]
pytest -m gym_install # test if install was successful
```

Before moving forward, it is crucial that ones understands the Gymnasium API. It fairly straightforward, a great starting place is to reference the [Gymnasium website](https://gymnasium.farama.org/), at minumum the `Basic Usage` and `Training an Agent` sections. 