from cli import SCENARIOS
from experimenter import MlpArchitectureSearch
from study import vz, vizier_client, via_ssh
from config.config import ExpConfig
import click
import os
from vizier.service import clients


@click.command()
@click.option("--loader-workers", type=int, default=2)
@click.option("--log-directory", type=click.Path(exists=True), required=True)
@click.option("--num-trials", type=int, default=1)
@click.option("--test", is_flag=True)
@click.option("--dataset", type=click.Choice(SCENARIOS.keys()), required=True)
def main(
    log_directory: str, num_trials: int, test: bool, dataset: str, loader_workers: int
):
    endpoint = via_ssh()
    name = f"{dataset}_Arch" + ("Test" if test else "")
    cfg = ExpConfig()
    cfg.name = name

    cfg = SCENARIOS[dataset](cfg)

    cfg.arch_autoencoder()
    cfg.strategy_surprisenet()
    cfg.loader_workers = loader_workers

    cfg.fixed_class_order = list(range(cfg.n_classes))
    cfg.tensorboard_dir = log_directory

    # Create the experimenter
    experimenter = MlpArchitectureSearch(cfg)
    problem = experimenter.problem_statement()
    clients.environment_variables.server_endpoint = endpoint
    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "QUASI_RANDOM_SEARCH"

    study = vizier_client(study_config=study_config, owner="SurpriseNet", study_id=name)
    pid = os.getpid()
    for _ in range(num_trials):
        suggestions = study.suggest(count=1, client_id=str(pid))
        experimenter.evaluate(suggestions)


if __name__ == "__main__":
    main()
