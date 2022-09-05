import functools
from itertools import product
from config.config import ExpConfig
from packnet.plugin import equal_capacity_prune_schedule
from train import Experiment
import git
import os
import click
import typing as t

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

def choose_scenario(cfg: ExpConfig, scenario: str):
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

def choose_strategy(cfg: ExpConfig, strategy: str):
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
    elif strategy == "LwF":
        cfg.use_learning_without_forgetting = True
        cfg.use_packnet = False
    elif strategy == "replay":
        cfg.use_experience_replay = True
        cfg.use_packnet = False
    else:
        raise NotImplementedError(f"Unknown variant {strategy}")
    return cfg

def choose_architecture(cfg: ExpConfig, architecture: str):
    if architecture == "AE":
        cfg = cfg.use_auto_encoder()
    elif architecture == "VAE":
        cfg = cfg.use_variational_auto_encoder()
    else:
        raise NotImplementedError(f"Unknown architecture {architecture}")
    return cfg

def setup_experiment(cfg, experiment_name,  scenario, architecture, strategy):
    cfg.name = get_experiment_name(experiment_name, scenario, architecture, strategy)
    cfg = choose_scenario(cfg, scenario)
    cfg = choose_strategy(cfg, strategy)
    cfg = choose_architecture(cfg, architecture)
    return cfg

def run(cfg: ExpConfig):
    print("--------------------------------------------------------------")
    print(f"Running Experiment '{cfg.name}'")
    print("--------------------------------------------------------------")
    experiment = Experiment(cfg)
    experiment.train()

class GenericFunctionality():
    def __init__(self):
        pass
    

    def __call__(self, f):
        @functools.wraps(f)
        def decorator(*args, **kwargs):
            n_runs = kwargs["n_runs"]
            fixed_class_order = kwargs["fixed_class_order"]
            scenario = kwargs["scenario"]
            del kwargs["n_runs"]
            del kwargs["fixed_class_order"]
            del kwargs["scenario"]

            if scenario == "all":
                scenarios = ALL_SCENARIOS
            else:
                scenarios = [scenario]

            for _, scenario in product(range(n_runs), scenarios):
                cfg = ExpConfig()
                cfg.fixed_class_order = fixed_class_order
                f(cfg, scenario, *args, **kwargs)

        decorator = click.option("--n-runs", default=1, type=int)(decorator)
        decorator = click.option("--fixed-class-order", is_flag=True, default=False)(decorator)
        decorator = click.option(
            "--scenario", 
            type=click.Choice(["all", *ALL_SCENARIOS]),
            default="all"
        )(decorator)

        return decorator


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
@GenericFunctionality()
def prune_levels(base_cfg: ExpConfig, scenario: str):
    for strategy, architecture, prune_level in product(
            ["taskInference"],
            ["AE", "VAE"],
            PRUNE_LEVELS):
        cfg = base_cfg.copy()
        cfg = setup_experiment(cfg, "PL", scenario, architecture, strategy)
        cfg.prune_proportion = prune_level
        run(cfg)

@cli.command()
@GenericFunctionality()
def baselines(base_cfg: ExpConfig, scenario: str):
    for strategy, architecture in product(
            ["cumulative", "finetuning"],
            ["AE", "VAE"]):
        cfg = base_cfg.copy()
        cfg = setup_experiment(cfg, "BL", scenario, architecture, strategy)
        run(cfg)

@cli.command()
@GenericFunctionality()
def equal_prune(base_cfg: ExpConfig, scenario: str):
    for strategy, architecture in product(
            ["taskInference"],
            ["AE", "VAE"]):
        cfg = base_cfg.copy()
        cfg = setup_experiment(cfg, "EP", scenario, architecture, strategy)
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)
        run(cfg)

@cli.command()
@GenericFunctionality()
def gen_replay(cfg: ExpConfig, scenario: str):
    cfg = setup_experiment(cfg, "OS", scenario, "VAE", "genReplay")
    run(cfg)


@cli.command()
@click.option("--shuffle-tasks", is_flag=True, default=False)
@click.option("--n-runs", type=int, default=1)
@click.argument("experiment_name")
@click.argument("strategy", type=click.Choice(["cumulative", "finetuning", "taskOracle", "taskInference", "genReplay", "LwF", "replay"]))
@click.argument("architecture", type=click.Choice(["AE", "VAE"]))
@click.argument("scenario", type=click.Choice(ALL_SCENARIOS))
def custom(experiment_name, strategy, architecture, scenario, shuffle_tasks, n_runs):
    for _ in range(n_runs):
        cfg = ExpConfig()
        cfg.fixed_class_order = not shuffle_tasks
        cfg = setup_experiment(cfg, experiment_name, scenario, architecture, strategy)
        run(cfg)



@cli.command()
@click.option("--shuffle-tasks", is_flag=True, default=False)
@click.option("--n-runs", type=int, default=1)
def replay(shuffle_tasks, n_runs):
    for scenario, buffer_sizes, _ in product(
            ALL_SCENARIOS, 
            [100, 1000, 10000],
            range(n_runs)):
        cfg = ExpConfig()
        cfg.fixed_class_order = not shuffle_tasks
        cfg = setup_experiment(cfg, "OS", scenario, "AE", "replay")
        cfg.replay_buffer = buffer_sizes
        run(cfg)

@cli.command()
def lwf():
    for scenario, lwf_alpha in product(ALL_SCENARIOS, [32, 8, 2, 1, 0.5]):
        cfg = ExpConfig()
        cfg = setup_experiment(cfg, "OS", scenario, "AE", "LwF")
        cfg.lwf_alpha = lwf_alpha
        run(cfg)

@cli.command()
def test_all():
    for scenario, arch, strategy in product(
        ALL_SCENARIOS, 
        ["AE", "VAE"], 
        ["cumulative", "finetuning", "taskInference", "taskOracle", "genReplay", "replay"]):
        # Blacklist some combinations
        if strategy == "genReplay" and arch == "AE":
            continue
        cfg = ExpConfig()
        cfg = setup_experiment(cfg, "TEST", scenario, arch, strategy)
        cfg.replay_buffer = 100
        cfg.total_task_epochs = 2
        cfg.retrain_epochs = 1
        run(cfg)

@cli.command()
@GenericFunctionality()
def test_command(cfg, scenario):
    print(scenario)


# @cli.command()
# def other_strategies():

    # for scenario, si_lambda in product(ALL_SCENARIOS, [10, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]):
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


    # for scenario, strategy in product(
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