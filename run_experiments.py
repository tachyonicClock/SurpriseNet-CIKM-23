import itertools
from config.config import ExperimentConfiguration
from packnet.plugin import equal_capacity_prune_schedule
from train import Experiment
import git
import os
import click

REPOSITORY = git.Repo('')
REPO_HASH = REPOSITORY.head.commit.hexsha[:8]
if REPOSITORY.is_dirty():
    REPO_HASH += "D"

PRUNE_LEVELS = [0.2, 0.4, 0.5, 0.6, 0.8]
LATENT_DIMS = [32, 64, 128, 256, 512]

ALL_SCENARIOS = ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50", "splitEmbeddedCIFAR100", "splitEmbeddedCORe50"]

def get_experiment_name(experiment, scenario, architecture, strategy):
    hostname = os.uname()[1]
    return f"{hostname}_{REPO_HASH}_{experiment}_{scenario}_{architecture}_{strategy}"

def run(cfg: ExperimentConfiguration):
        print("--------------------------------------------------------------")
        print(f"Running Experiment '{cfg.name}'")
        print("--------------------------------------------------------------")
        experiment = Experiment(cfg)
        experiment.train()

def choose_scenario(cfg: ExperimentConfiguration, scenario: str):
    if scenario == "splitFMNIST":
        cfg = cfg.use_fmnist()
    elif scenario == "splitCIFAR10":
        cfg = cfg.use_cifar10()
    elif scenario == "splitCIFAR100":
        cfg = cfg.use_cifar100()
    elif scenario == "splitCORe50":
        cfg = cfg.use_core50()
    elif scenario == "splitEmbeddedCIFAR100":
        cfg = cfg.use_embedded_cifar100()
    elif scenario == "splitEmbeddedCORe50":
        cfg = cfg.use_embedded_core50()
    else:
        raise NotImplementedError(f"Unknown scenario {scenario}")
    return cfg

def choose_strategy(cfg: ExperimentConfiguration, strategy: str):
    if strategy == "cumulative":
        cfg = cfg.use_cumulative_learning()
    elif strategy == "finetuning":
        cfg.use_packnet = False
    elif strategy == "taskOracle":
        cfg.enable_packnet()
    elif strategy == "taskInference":
        cfg.enable_packnet()
        cfg.task_inference_strategy = "task_reconstruction_loss"
    else:
        raise NotImplementedError(f"Unknown variant {strategy}")
    return cfg

def choose_architecture(cfg: ExperimentConfiguration, architecture: str):
    if architecture == "AE":
        cfg = cfg.use_auto_encoder()
    elif architecture == "VAE":
        cfg = cfg.use_variational_auto_encoder()
    else:
        raise NotImplementedError(f"Unknown architecture {architecture}")
    return cfg


# 
# Experiments
#

@click.group()
@click.option("--ignore-dirty", is_flag=True, default=False)
def cli(ignore_dirty):
    if REPOSITORY.is_dirty() and not ignore_dirty:
        print("Please commit your changes before running experiments. This is best practice")
        print("Use --ignore-dirty to ignore this")
        exit(1)

@cli.command()
def prune_levels():
    for strategy, architecture, scenario, prune_level in itertools.product(
            ["taskInference"],
            ["AE"],
            ["splitEmbeddedCIFAR100", "splitEmbeddedCORe50"],
            PRUNE_LEVELS):
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("PL", scenario, architecture, strategy)
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = prune_level
        run(cfg)

@cli.command()
def equal_prune():
    for strategy, architecture, scenario in itertools.product(
            ["taskOracle" ,"taskInference"],
            ["AE", "VAE"],
            ALL_SCENARIOS):
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("EP", scenario, architecture, strategy)

        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()

@cli.command()
def best_results():
    for strategy, architecture, scenario in itertools.product(
            ["cumulative", "finetuning", "taskOracle", "taskInference"],
            ["AE", "VAE"],
            ALL_SCENARIOS):
        cfg = ExperimentConfiguration()
        cfg.name = f"{scenario}_{architecture}_{strategy}"

        # Select the dataset to use for the experiment
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()

@cli.command()
def other_strategies():

    for scenario, strategy in itertools.product(
        ALL_SCENARIOS,
        ["SI", "LwF"]):
        architecture = "AE" if strategy != "genReplay" else "VAE"
            

        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("OS", scenario, architecture, strategy)
        cfg.use_packnet = False

        if strategy in ["SI", "LwF"]:
            cfg.use_adam = False

        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)

        if strategy == "replay":
            cfg.use_experience_replay = True
        elif strategy == "genReplay":
            cfg.use_generative_replay = True
        elif strategy == "SI":
            cfg.use_synaptic_intelligence = True
        elif strategy == "LwF":
            cfg.use_learning_without_forgetting = True

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()


# Main
if __name__ == '__main__':
    cli()