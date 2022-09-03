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
    elif strategy == "genReplay":
        cfg.use_generative_replay = True
        cfg.use_packnet = False
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

def run(cfg: ExperimentConfiguration):
    print("--------------------------------------------------------------")
    print(f"Running Experiment '{cfg.name}'")
    print("--------------------------------------------------------------")
    experiment = Experiment(cfg)
    experiment.train()


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
            ["AE", "VAE"],
            ALL_SCENARIOS,
            PRUNE_LEVELS):
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("PL", scenario, architecture, strategy)
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = prune_level
        run(cfg)


@cli.command()
def latent_sizes():
    """
    Try different latent dimension sizes while holding other variables
    constant
    """
    for scenario, latent_size in itertools.product(
            ALL_SCENARIOS,
            [32, 64, 128, 256, 512, 1024]):
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("LS", scenario, "AE", "taskInference")
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, "AE")
        cfg = choose_strategy(cfg, "taskInference")
        cfg.prune_proportion = 0.5
        cfg.latent_dims = latent_size
        run(cfg)

@cli.command()
@click.option("--shuffle-tasks", is_flag=True, default=False)
@click.option("--n-runs", type=int, default=1)
def equal_prune(shuffle_tasks: bool, n_runs: int):
    for strategy, architecture, scenario, _ in itertools.product(
            ["taskInference"],
            ["AE", "VAE"],
            ALL_SCENARIOS,
            range(n_runs)):
        cfg = ExperimentConfiguration()
        cfg.fixed_class_order = not shuffle_tasks
        cfg.name = get_experiment_name("EP", scenario, architecture, strategy)

        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)

        run(cfg)


@cli.command()
@click.option("--shuffle-tasks", is_flag=True, default=False)
@click.option("--n-runs", type=int, default=1)
def baselines(shuffle_tasks, n_runs):
    for strategy, architecture, scenario, _ in itertools.product(
            ["cumulative", "finetuning"],
            ["AE", "VAE"],
            ALL_SCENARIOS,
            range(n_runs)):
        cfg = ExperimentConfiguration()
        cfg.fixed_class_order = not shuffle_tasks
        cfg.name = get_experiment_name("BL", scenario, architecture, strategy)

        # Select the dataset to use for the experiment
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        run(cfg)

@cli.command()
def gen_replay():
    for scenario in ["splitFMNIST"]:
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("OS", scenario, "VAE", "genReplay")
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, "VAE")
        cfg.use_generative_replay = True
        cfg.use_packnet = False
        # cfg.total_task_epochs = 5
        cfg.use_classifier_loss = True
        run(cfg)

@cli.command()
def test_all():
    for scenario, arch, strategy in itertools.product(["splitFMNIST"], ["AE", "VAE"], ["taskInference", "taskOracle", "genReplay"]):
        # Blacklist some combinations
        if strategy == "genReplay" and arch == "AE":
            continue

        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("TEST", scenario, arch, strategy)
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, arch)
        cfg = choose_strategy(cfg, strategy)
        cfg.total_task_epochs = 2
        cfg.retrain_epochs = 1
        run(cfg)

@cli.command()
def other_strategies():

    # for scenario, si_lambda in itertools.product(ALL_SCENARIOS, [10, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]):
    #     cfg = ExperimentConfiguration()
    #     cfg.name = get_experiment_name("OS", scenario, "SI", "AE")
    #     cfg.use_packnet = False
    #     cfg = choose_scenario(cfg, scenario)
    #     cfg = choose_architecture(cfg, "AE")

    #     cfg.use_synaptic_intelligence = True

    #     cfg.use_adam = False
    #     cfg.learning_rate = 0.01
    #     cfg.si_lambda = si_lambda
    #     run(cfg)

    for scenario, lwf_alpha in itertools.product(ALL_SCENARIOS, [8, 4, 2, 1, 0.5]):
        cfg = ExperimentConfiguration()
        cfg.name = get_experiment_name("OS", scenario, architecture="AE", strategy="LwF")
        cfg.use_packnet = False
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, "AE")

        cfg.use_learning_without_forgetting = True

        cfg.use_adam = False
        cfg.learning_rate = 0.01
        cfg.lwf_alpha = lwf_alpha
        run(cfg)

    # for scenario, strategy in itertools.product(
    #     ["splitFMNIST"],
    #     ["SI", "LwF"]):
    #     architecture = "AE" if strategy != "genReplay" else "VAE"
            

    #     cfg = ExperimentConfiguration()
    #     cfg.name = get_experiment_name("OS", scenario, architecture, strategy)
    #     cfg.use_packnet = False


    #     cfg = choose_scenario(cfg, scenario)
    #     cfg = choose_architecture(cfg, architecture)

    #     if strategy == "replay":
    #         cfg.use_experience_replay = True
    #     elif strategy == "genReplay":
    #         cfg.use_generative_replay = True
    #     elif strategy == "SI":
    #         cfg.use_synaptic_intelligence = True
    #     elif strategy == "LwF":
    #         cfg.use_learning_without_forgetting = True

    #     if strategy in ["SI", "LwF"]:
    #         cfg.use_adam = False
    #         cfg.learning_rate = 0.01
    #         cfg.use_reconstruction_loss = False

    #     print("################")
    #     print(f"{cfg.name}")
    #     print("################")
    #     experiment = Experiment(cfg)
    #     experiment.train()


# Main
if __name__ == '__main__':
    cli()