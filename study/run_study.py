from torch.multiprocessing import Process, set_start_method
from experimenter import SurpriseNetExperimenter
from study import vz, vizier_client, via_ssh
from config.config import ExpConfig
import click
import os
import time
from vizier.service import clients


def start_worker(endpoint: str, num_trials, name, experimenter):
    clients.environment_variables.server_endpoint = endpoint
    problem = experimenter.problem_statement()
    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "QUASI_RANDOM_SEARCH"
    study = vizier_client(study_config=study_config, owner="SurpriseNet", study_id=name)
    pid = os.getpid()
    for _ in range(num_trials):
        suggestions = study.suggest(count=1, client_id=str(pid))
        experimenter.evaluate(suggestions)


@click.command()
@click.option("--log-directory", type=click.Path(exists=True), required=True)
@click.option("--num-trials", type=int, default=1)
@click.option("--test", is_flag=True)
@click.option("--worker-count", type=int, default=1)
def main(log_directory: str, num_trials: int, test: bool, worker_count: int):
    endpoint = via_ssh()
    name = "S-DSADS_Exploit_2" + ("Test" if test else "")
    set_start_method("spawn")
    cfg = ExpConfig()
    cfg.name = name
    cfg.scenario_dsads()
    cfg.arch_autoencoder()
    cfg.strategy_surprisenet()
    # cfg.fixed_class_order = list(range(cfg.n_classes))
    cfg.tensorboard_dir = log_directory

    if test:
        cfg.retrain_epochs = 1
        cfg.total_task_epochs = 2

    experimenter = SurpriseNetExperimenter(cfg)

    processes = []
    for i in range(worker_count):
        click.secho(f"Starting worker {i}", fg="green")
        processes.append(
            Process(
                target=start_worker,
                args=[endpoint, num_trials // worker_count, name, experimenter],
            )
        )
        processes[-1].start()
        time.sleep(10)

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
