import typing as t
import click
import git
import os
from packnet.plugin import equal_capacity_prune_schedule
from train import Experiment
from config.config import ExpConfig


def get_experiment_name(repo_hash, experiment, scenario, architecture, strategy):
    hostname = os.uname()[1]
    return f"{hostname}_{repo_hash}_{experiment}_{scenario}_{architecture}_{strategy}"


SCENARIOS: t.Dict[str, t.Callable[[ExpConfig], ExpConfig]] = {
    "S-FMNIST": ExpConfig.scenario_fmnist,
    "S-CIFAR10": ExpConfig.scenario_cifar10,
    "S-CIFAR100": ExpConfig.scenario_cifar100,
    "S-CORe50": ExpConfig.scenario_core50,
    "SE-CIFAR100": ExpConfig.scenario_embedded_cifar100,
    "SE-CORe50": ExpConfig.scenario_embedded_core50,
}

ARCHITECTURES: t.Dict[str, t.Callable[[ExpConfig], ExpConfig]] = {
    "AE": ExpConfig.arch_autoencoder,
    "VAE": ExpConfig.arch_variational_auto_encoder,
}


def run(cfg: ExpConfig):
    click.secho("-"*80, fg="green")
    click.secho(f"Running Experiment '{cfg.name}'", fg="green")
    click.secho("-"*80, fg="green")
    experiment = Experiment(cfg)
    experiment.train()


@click.group()
@click.option("--ignore-dirty", is_flag=True, default=False,
              help="Do NOT abort when uncommitted changes exist")
@click.option("--epochs", type=int, default=None, 
    help="Number of epochs to train each task on. Default varies based on the" +
    " given scenario.")
@click.argument("label", type=str)
@click.argument("scenario", type=click.Choice(SCENARIOS.keys()), required=True)
@click.argument("architecture", type=click.Choice(ARCHITECTURES.keys()), required=True)
@click.pass_context
def cli(ctx, ignore_dirty: bool, epochs: int, label: str, scenario: str, architecture: str):
    # Start building an experiment configuation
    cfg = ExpConfig()

    # Abort on dirty?
    # Store the repository version
    project_repo = git.Repo('')
    cfg.repo_hash = project_repo.head.commit.hexsha[:8]
    if project_repo.is_dirty() and not ignore_dirty:
        click.echo(
            "Please commit your changes before running experiments. This is best practice")
        click.echo("Use --ignore-dirty to ignore this")
        exit(1)
    elif ignore_dirty and project_repo.is_dirty():
        cfg.repo_hash += "D"
        click.secho("Warning: running with uncommitted changes", fg="yellow")

    cfg = SCENARIOS[scenario](cfg)
    cfg.scenario_name = scenario
    cfg.total_task_epochs = epochs
    cfg = ARCHITECTURES[architecture](cfg)
    cfg.label = label
    ctx.obj = cfg

@cli.command()
@click.option("--prune-proportion", type=float, default=0.5,
              help="PackNet prunes the network to this proportion each experience")
@click.pass_obj
def packNet(cfg: ExpConfig, prune_proportion: float):
    """Use task incremental PackNet

    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a 
    Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer 
    Vision and Pattern Recognition, 7765–7773.
    https://doi.org/10.1109/CVPR.2018.00810
    """
    cfg.strategy_packnet()
    cfg.prune_proportion = prune_proportion
    cfg.name = get_experiment_name(
        cfg.repo_hash, cfg.label, cfg.scenario_name, cfg.architecture, "taskOracle")
    run(cfg)

@cli.command()
@click.option("--prune-proportion", type=float, default=None,
                help="PackNet prunes the network to this proportion each experience")
@click.option("--equal-prune", is_flag=True, default=False,
                help="Prune such that each task has the same number of parameters")
@click.pass_obj
def ci_packnet(cfg: ExpConfig, prune_proportion: float, equal_prune: bool):
    """Use  CI-PackNet (ours). CI-PackNet performs the same pruning as PackNet,
    but uses anomaly detection inspired task inference to infer task labels, 
    removing the reliance on a task oracle. Additionally it can be pruned
    such that each task has the same number of parameters.
    """
    cfg.strategy_ci_packnet()

    # Setup pruning scheme
    if not equal_prune:
        cfg.prune_proportion = prune_proportion if prune_proportion != None else 0.5
    else:
        if prune_proportion is not None:
            click.secho("Warning: --prune-proportion is ignored when using equal pruning", fg="yellow")
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)

    cfg.name = get_experiment_name(
        cfg.repo_hash, cfg.label, cfg.scenario_name, cfg.architecture, "taskInference")
    run(cfg)


@cli.command()
@click.option("--buffer-size", type=int, default=1000, 
    help="Size of the replay buffer")
@click.pass_obj
def replay(cfg: ExpConfig, buffer_size: int):
    """Use replay buffer strategy, where the network is trained using a 
    replay buffer of previous experiences.
    """
    cfg.strategy_replay()
    cfg.replay_buffer = buffer_size
    cfg.name = get_experiment_name(
        cfg.repo_hash, cfg.label, cfg.scenario_name, cfg.architecture, "replay")
    run(cfg)

@cli.command()
@click.pass_obj
def non_continual(cfg: ExpConfig):
    """Train a model on all experiences at once. This is a baseline for
    comparison with continual learning methods.
    """
    cfg.strategy_not_cl()
    cfg.name = get_experiment_name(
        cfg.repo_hash, cfg.label, cfg.scenario_name, cfg.architecture, "nonContinual")
    run(cfg)

if __name__ == "__main__":
    cli()
