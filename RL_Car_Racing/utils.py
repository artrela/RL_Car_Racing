import wandb

class WandBLogger:
    def __init__(self, project_name="RL_Car_Racing"):

        wandb.login()

        self.run = wandb.init(project=project_name,
                        name = "local-custom-actions",
                        config = {
                            "learning_rate": 0.0005,
                            "gamma": 0.95,
                            "episodes": 1000,
                            "step_update": 10,
                            "epsilon": 0.1,
                            "memory_size": 1e5,
                            "batch_size": 64,
                            "episode_decay": 300,
                            "loss": "mse",
                            "optimizer": "AdamW",
                            "target_net_update": 2, 
                            "notes": "action space changes"
                            })
        
    def send_log(self, statistics):
        
        wandb.log(statistics)


