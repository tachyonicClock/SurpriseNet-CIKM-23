
import sys
from config.halton import generate_search
sys.path.append("network/hvae")

from train import Experiment
from config.config import ExpConfig

cfg = ExpConfig()
cfg.scenario_fmnist()
cfg.arch_deep_vae()
cfg.strategy_surprisenet()
cfg.name = "LrHparamSearch"

search_space = {
    'learning_rate': {"min": 1e-6, "max": 0.002, "scaling": "linear"},
    'beta_warmup':   {"min": 0,    "max": 400,   "scaling": "linear"},
}
search = generate_search(search_space, 100)

for hyper_params in search:
    run_cfg = cfg.copy()

    print("=-"*40)
    print(f"Running {run_cfg.name} with:")
    for k, v in hyper_params._asdict().items():
        print(f"  {k:20} {v}")
    print("=-"*40)

    # Set hyper parameters
    run_cfg.learning_rate = hyper_params[0]
    run_cfg.HVAE_schedule["warmup_epochs"] = int(hyper_params[1])

    exp = Experiment(run_cfg)
    exp.logger.writer.add_hparams({
        'lr': run_cfg.learning_rate,
        'beta_warmup': run_cfg.HVAE_schedule["warmup_epochs"],
    },
    {
        'TaskIdAccuracy': -1.0,
    })

    try:
        result = exp.train()
    except Exception as e:
        print()
        print(f"{run_cfg.name} failed")
        continue

exp = Experiment(cfg)
