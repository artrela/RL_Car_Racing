import wandb

class WandBLogger:
    def __init__(self, project_name="RL_Car_Racing"):

        wandb.login()

        self.run = wandb.init(project=project_name,
                        name = "deeper-net-2-2",
                        # name = "ssh-reward-clip",
                        config = {
                            "learning_rate": 0.0005,
                            "gamma": 0.95,
                            "episodes": 5000,
                            "step_update": 20,
                            "epsilon": 0.1,
                            "memory_size": 1e5,
                            "batch_size": 64,
                            "episode_decay": 2000,
                            "loss": "mse",
                            "optimizer": "AdamW",
                            "target_net_update": 2, 
                            "notes": "2 extra linear layer"
                            })
        
    def send_log(self, statistics):
        
        wandb.log(statistics)


