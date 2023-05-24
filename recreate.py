from config.config import ExpConfig
from train import Experiment

root_path = "experiment_logs/BestDeepVAE"
old_config = f"{root_path}/config.json"
class_order = f"{root_path}/class_order.txt"


task_comp = []
with open(class_order, "r") as f:
    for line in f.readlines():
        task_comp.extend(map(int, line.split(",")))

cfg = ExpConfig.from_json(old_config)

cfg.tensorboard_dir = "experiment_logs/BestRecreation"
cfg.hvae_loss_kwargs["free_nat_constant_epochs"] = 200 // 2
cfg.hvae_loss_kwargs["free_nat_cooldown_epochs"] = 200 // 2
cfg.hvae_loss_kwargs["beta_warmup"] = 100
cfg.fixed_class_order = task_comp

cfg.network_cfg["base_channels"] = 64

cfg.label = None
cfg.name = "BestRecreationOldReInit"

print("Fixed class order: ", cfg.fixed_class_order)
Experiment(cfg).train()
