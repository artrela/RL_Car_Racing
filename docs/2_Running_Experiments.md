# Part 2: Running Experiments

This repository was setup to expose students to some good practices when building/training models. We want you to get into the good habit of creating a paper trail for experiments, using tools to visualize outcomes, and saving weights of your models to return to in the future. 

## 2.1 Configuration Files
A good practice for running experiements that include Learning-Based models is to keep a paper trail. 
This allows one to refer to past experiments, and infer the effects of the hyperparameters on the model. 
To aid you in this, we have provided a starter configuration file at `RL_Car_Racing/config/default.md`. Within 
this markdown you will find the purpose of each hyperparameter. A suggested range for each parameter is given below:


### Suggested Hyperparameter Values for DQN Training
| Hyperparameter         | Suggested Range/Values             |
|------------------------|-------------------------------------|
| Start Skip             | `10-50`                            |
| Stacked Frames         | `30-100`                           |
| Number of Episodes     | `3000-10000`                       |
| Episode Decay          | `1000-5000`                        |
| Memory Length          | `50,000-500,000`                   |
| Random Seed            | `1`                                |
| Batch Size             | `32-256`                           |
| Step Update            | `10-100`                           |
| Target Update Frequency| `1-10`                             |
| Gamma (Discount Factor)| `0.90-0.99`                        |
| Learning Rate          | `0.0001-0.001`                     |
| Epsilon (Exploration)  | `0.05-0.5`                         |
| Model                  | `'DQN'`, `'DoubleDQN'`             |
| Optimizer              | `'Adam'`, `'AdamW'`, `'RMSprop'`    |
| Loss Function          | `'MSE'`, `'Huber'`, `'SmoothL1Loss'` |


The intended use for these configuration follows is as follows:
1. Change the `name` entry, which will be reflected as a run in your Weights and Biases project. 
2. Change the parameters as needed.
3. Reference the paremeter file while running the main script

```bash
python3 RL_Car_Racing/main.py -c RL_Car_Racing/config/<name>.yaml
```

## 2.2 Metrics Tracking
Another great habit to get into when tracking experiments, is to use an off-the-shelf experiment tracker. This may include a variety of tools: `Tensorboard`, `MLFlow`, or `Weights and Biases`. For this assignment, we will be exposing you to Weights and Biases (WandB). 

### Installing WandB
First, create a free account [here](https://wandb.ai/site).

If you followed the instructions given in `0_Environment_Setup.md`, you should already have WandB setup in your conda environment. With you environment enabled, execute the following command in the terminal:

```bash
wandb login
```

If you are having trouble, start at the [quickstart guide](https://docs.wandb.ai/quickstart/), and go from there. 

### Using WandB
WandB is a very powerful tool that allows you to do a variety of things, some examples being:
1. Experiment tracking (metrics, hyperparameters, and system logs).
2. Real-time visualizations of metrics and comparisons across runs.
3. Hyperparameter tuning with sweeps (grid search, random search, Bayesian optimization).
4. Model and dataset versioning.
5. Logging and visualizing media (images, videos, and audio).

For this assignment, we are simply providing you with the functionality for application 1. This requires no effort on your behalf, check the `WandBLogger` class in `RL_Car_Racing/utils.py` to understand more. 

## 2.3 Data Versioning Control (DVC) -- `COMING SOON`
TODO



