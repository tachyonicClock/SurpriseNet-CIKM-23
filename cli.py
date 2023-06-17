import os
import random
import typing as t

import click
import git
import numpy as np
import torch

from config.config import ExpConfig
from surprisenet.plugin import equal_capacity_prune_schedule
from train import Experiment
from matplotlib import pyplot as plt

# Use non interactive backend
plt.switch_backend("agg")
plt.ioff()


def get_experiment_name(repo_hash, experiment, scenario, architecture, strategy):
    hostname = os.uname()[1]
    return f"{hostname}_{repo_hash}_{experiment}_{scenario}_{architecture}_{strategy}"


SCENARIOS: t.Dict[str, t.Callable[[ExpConfig], ExpConfig]] = {
    "S-FMNIST": ExpConfig.scenario_fmnist,
    "SE-FMNIST": ExpConfig.scenario_embedded_fmnist,
    "S-CIFAR10": ExpConfig.scenario_cifar10,
    "SE-CIFAR10": ExpConfig.scenario_embedded_cifar10,
    "S-CIFAR100": ExpConfig.scenario_cifar100,
    "S-CORE50": ExpConfig.scenario_core50,
    "SE-CIFAR100": ExpConfig.scenario_embedded_cifar100,
    "SE-CORE50": ExpConfig.scenario_embedded_core50,
    "S-DSADS": ExpConfig.scenario_dsads,
    "S-PAMAP2": ExpConfig.scenario_pamap2,
}

ARCHITECTURES: t.Dict[str, t.Callable[[ExpConfig], ExpConfig]] = {
    "AE": ExpConfig.arch_autoencoder,
    "VAE": ExpConfig.arch_variational_auto_encoder,
}


def run(cfg: ExpConfig):
    click.secho("-" * 80, fg="green")
    click.secho(f"Running Experiment '{cfg.name}'", fg="green")
    click.secho("-" * 80, fg="green")
    experiment = Experiment(cfg)
    experiment.train()


@click.group()
@click.option(
    "--ignore-dirty",
    is_flag=True,
    default=False,
    help="Do NOT abort when uncommitted changes exist",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Number of epochs to train each task on. Default varies based on the"
    + " given scenario.",
)
@click.option(
    "-z",
    "--latent-dim",
    type=int,
    default=None,
    help="Latent dimension of the autoencoder. Default varies based on the"
    + " given scenario.",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Learning rate. Default varies based on the" + " given scenario.",
)
@click.option(
    "--log-mini-batches",
    type=bool,
    default=False,
    is_flag=True,
    help="Log each mini-batches",
)
@click.option(
    "--no-reconstruction",
    is_flag=True,
    default=False,
    help="Set reconstruction loss to zero",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="The training and evaluation batch size.",
)
@click.option(
    "--std-order",
    type=bool,
    default=False,
    is_flag=True,
    help="Use the standard order for task composition e.g class [0, 1], [2, 3], etc",
)
@click.option(
    "--task-count",
    type=int,
    default=None,
    help="The number of tasks to train on. Default varies based on the"
    + " given scenario.",
)
@click.option(
    "--class-loss-type",
    type=click.Choice(["CrossEntropy", "LogitNorm"]),
    default="CrossEntropy",
    help="The loss function to use for the classifier",
)
@click.option("--log-directory", type=str, default=None)
@click.argument("label", type=str)
@click.argument("scenario", type=click.Choice(SCENARIOS.keys()), required=True)
@click.argument("architecture", type=click.Choice(ARCHITECTURES.keys()), required=True)
@click.pass_context
def cli(
    ctx,
    ignore_dirty: bool,
    epochs: t.Optional[int],
    label: str,
    scenario: str,
    architecture: str,
    lr: t.Optional[float],
    latent_dim: t.Optional[int],
    log_mini_batches: bool,
    no_reconstruction: bool,
    log_directory: str,
    batch_size: t.Optional[int],
    task_count: t.Optional[int],
    std_order: bool,
    class_loss_type: str,
):
    # Start building an experiment configuation
    cfg = ExpConfig()
    cfg.log_mini_batch = log_mini_batches

    # Abort on dirty?
    # Store the repository version
    project_repo = git.Repo("")
    cfg.repo_hash = project_repo.head.commit.hexsha[:8]
    if project_repo.is_dirty() and not ignore_dirty:
        click.echo(
            "Please commit your changes before running experiments. This is best"
            + " practice"
        )
        click.echo("Use --ignore-dirty to ignore this")
        exit(1)
    elif ignore_dirty and project_repo.is_dirty():
        cfg.repo_hash += "D"
        click.secho("WARN: Running with uncommitted changes", fg="yellow")

    cfg = SCENARIOS[scenario](cfg)
    cfg = ARCHITECTURES[architecture](cfg)
    cfg.scenario_name = scenario
    cfg.label = label
    cfg.tensorboard_dir = log_directory or cfg.tensorboard_dir

    # Override the default order if given
    if std_order:
        cfg.fixed_class_order = list(range(cfg.n_classes))
    # Override the default latent dimension if given
    if latent_dim is not None:
        cfg.latent_dims = latent_dim
    # Override the default number of epochs if given
    if epochs is not None:
        cfg.total_task_epochs = epochs
    # Override the default learning rate if given
    if lr is not None:
        cfg.learning_rate = lr
    if no_reconstruction:
        cfg.reconstruction_loss_weight = None

    if batch_size is not None:
        cfg.batch_size = batch_size

    if task_count is not None:
        click.secho(f"WARN: Overriding task count to {task_count}", fg="yellow")
        cfg.n_experiences = task_count

    cfg.classifier_loss_type = class_loss_type

    ctx.obj = cfg


@cli.command()
@click.option(
    "-p",
    "--prune-proportion",
    type=float,
    default=None,
    help="SurpriseNet prunes the network to this proportion each experience",
)
@click.option(
    "--equal-prune",
    is_flag=True,
    default=False,
    help="Prune such that each task has the same number of parameters",
)
@click.option(
    "--retrain-epochs",
    type=int,
    default=None,
    help="Override the default number of epochs to retrain the"
    + "network after pruning",
)
@click.pass_obj
def surprise_net(
    cfg: ExpConfig,
    prune_proportion: float,
    equal_prune: bool,
    retrain_epochs: t.Optional[int],
):
    """SurpriseNet uses anomaly detection inspired task inference to infer task labels,
    removing the reliance on a task oracle.
    """
    cfg.strategy_surprisenet()
    assert not (
        equal_prune and prune_proportion is not None
    ), "Cannot use both equal pruning and prune proportion"

    # Setup pruning scheme
    if prune_proportion is not None:
        cfg.prune_proportion = prune_proportion if prune_proportion is not None else 0.5
    elif equal_prune:
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)
    else:
        raise ValueError("Must specify either prune proportion or equal prune schedule")

    if retrain_epochs is not None:
        click.secho("INFO: Overriding default retrain epochs", fg="yellow")
        cfg.retrain_epochs = retrain_epochs

    assert cfg.retrain_epochs < cfg.total_task_epochs, (
        f"Retrain epochs ({cfg.retrain_epochs}) must be less than total task"
        + f" epochs ({cfg.total_task_epochs})"
    )

    cfg.name = get_experiment_name(
        cfg.repo_hash, cfg.label, cfg.scenario_name, cfg.architecture, "surpriseNet"
    )
    run(cfg)


if __name__ == "__main__":
    cli()
