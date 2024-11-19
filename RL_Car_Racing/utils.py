import wandb

class WandBLogger:
    def __init__(self, project_name="RL_Car_Racing"):

        wandb.login()

        self.run = wandb.init(project=project_name,
                        name = "baseline",
                        config = {
                            "learning_rate": 0.001,
                            "gamma": 0.9,
                            "episodes": 1e6,
                            "step_update": 100,
                            "epsilon": 0.1,
                            "memory_size": 1e6,
                            "batch_size": 256,
                            "loss": "mse",
                            "optimizer": "AdamW"
                            })
        
    def send_log(self, statistics):
        
        wandb.log(statistics)


