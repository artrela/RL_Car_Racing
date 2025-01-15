# Part 2: Running Experiments

This repository was setup to expose students to some good practices when building/training models. We want you to get into the good habit of creating a paper trail for experiments, using tools to visualize outcomes, and saving weights of your models to return to in the future. 

## 2.1 Configuration Files
TODO


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


## 2.2 Metrics Tracking
TODO 

## 2.3 Data Versioning Control (DVC) -- `COMING SOON`
TODO



