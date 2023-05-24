import sys

sys.path.append("network/hvae")

from experimenter import SurpriseNetExperimenter
from study import vizier_client_via_ssh, vz
from config.config import ExpConfig
import click
import os


@click.command()
@click.option("--log-directory", type=click.Path(exists=True), required=True)
@click.option("--num-trials", type=int, default=1)
@click.option("--test", is_flag=True)
def main(log_directory: str, num_trials: int, test: bool):
    name = "InitialCifar10" + ("Test" if test else "")

    cfg = ExpConfig()
    cfg.name = name
    cfg.scenario_cifar10()
    cfg.arch_deep_vae()
    cfg.strategy_surprisenet()
    cfg.fixed_class_order = list(range(10))
    cfg.tensorboard_dir = log_directory

    if test:
        cfg.retrain_epochs = 1
        cfg.total_task_epochs = 2

    experimenter = SurpriseNetExperimenter(cfg)
    problem = experimenter.problem_statement()
    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "QUASI_RANDOM_SEARCH"

    study = vizier_client_via_ssh(
        study_config=study_config, owner="SurpriseNet", study_id=name
    )

    client_id = f"{os.uname().nodename}::{os.getpid()}"

    print("Client ID: ", client_id)
    print("Study Resource Name: ", study.resource_name)
    for _ in range(num_trials):
        suggestions = study.suggest(count=1, client_id=client_id)
        experimenter.evaluate(suggestions)


if __name__ == "__main__":
    main()
