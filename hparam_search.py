import sys
from config.halton import generate_search

sys.path.append("network/hvae")

from train import Experiment
from config.config import ExpConfig
import click
import numpy as np


@click.command()
@click.option("--log-directory", type=click.Path(exists=True), required=True)
@click.option("--num-trials", type=int, default=1)
def main(log_directory: str, num_trials: int):
    cfg = ExpConfig()
    cfg.scenario_fmnist()
    cfg.arch_deep_vae()
    cfg.strategy_surprisenet()
    cfg.name = "LrHparamSearch"
    cfg.tensorboard_dir = log_directory

    learning_rate_search = np.random.uniform(1e-6, 0.005, num_trials).astype(float)
    beta_warmup_search = np.random.uniform(0, 400, num_trials).astype(int)

    for learning_rate, beta_warmup in zip(learning_rate_search, beta_warmup_search):
        run_cfg = cfg.copy()

        print("=-" * 40)
        print(f"Running {run_cfg.name} with:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Beta warmup: {beta_warmup}")
        print("=-" * 40)

        # Set hyper parameters
        run_cfg.learning_rate = float(learning_rate)
        run_cfg.HVAE_schedule["warmup_epochs"] = int(beta_warmup)

        exp = Experiment(run_cfg)
        exp.logger.writer.add_hparams(
            {
                "lr": run_cfg.learning_rate,
                "beta_warmup": run_cfg.HVAE_schedule["warmup_epochs"],
            },
            {
                "TaskIdAccuracy": -1.0,
            },
        )

        try:
            exp.train()
        except Exception:
            print()
            print(f"{run_cfg.name} failed")
            continue

    exp = Experiment(cfg)


if __name__ == "__main__":
    main()
