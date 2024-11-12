# RL_Car_Racing
An introduction to common reinforcement methods (RL) leveraging Gymnasium (formerly OpenAI Gym), as a featured part of Carnegie Mellon University's MRSD Summer Software Bootcamp. 

# Setup the Environment

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) and setup an environment:
```bash
conda env create -n rl-mrsd python=3.10 -y 
conda activate rl-mrsd
```

2. Install a cuda enabled distribution of [PyTorch](https://pytorch.org/get-started/locally/) by matching your version of cuda to their download interface. *Note: this assumes you already have a version of cuda downloaded on your platform. If this is not the case, a good place to start is [here](https://developer.nvidia.com/cuda-toolkit-archive).* 

In my case, with cuda 12.2 I will download the closest stable version:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

3. After installing PyTorch, install the rest of the requirements with:
```
pip install requirments.txt
```

Confirm that installation has been completed successfully by running the following test(s):
```bash
pytest -m gpu
```


