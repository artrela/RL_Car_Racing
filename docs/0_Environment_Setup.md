# Part 0: Environment Setup

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) and setup an environment:
```bash
conda create -n rl-mrsd python=3.10 -y 
conda activate rl-mrsd
```

2. Install a cuda enabled distribution of [PyTorch](https://pytorch.org/get-started/locally/) by matching your version of cuda to their download interface. *Note: this assumes you already have a version of cuda downloaded on your platform. If this is not the case, a good place to start is [here](https://developer.nvidia.com/cuda-toolkit-archive).* In my case, with cuda 12.2 I will download the closest stable version. One can verify their version of cuda with a call to `nvidia-smi`:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

3. After installing PyTorch, install the rest of the requirements with:
```bash
pip install -r requirments.txt
# Confirm that installation has been completed successfully by running the following test(s):
pytest -m gpu 
```