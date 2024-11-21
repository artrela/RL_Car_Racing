import wandb

class WandBLogger:
    def __init__(self, project_name="RL_Car_Racing"):

        wandb.login()

        self.run = wandb.init(project=project_name,
                        name = "ddqn-explore-fix",
                        config = {
                            "learning_rate": 0.0005,
                            "gamma": 0.96,
                            "episodes": 1000,
                            "step_update": 10,
                            "epsilon": 0.1,
                            "memory_size": 1e5,
                            "batch_size": 64,
                            "episode_decay": 350,
                            "loss": "mse",
                            "optimizer": "AdamW",
                            "target_net_update": 5, 
                            "notes": "reward gas, penalize nothing, dec lr, increase mem buffer, increase eps decay, introduce domain randomization in env"
                            })
        
    def send_log(self, statistics):
        
        wandb.log(statistics)


