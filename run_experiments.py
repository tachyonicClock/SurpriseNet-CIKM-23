import itertools
from config.config import ExperimentConfiguration
from packnet.plugin import equal_capacity_prune_schedule
from train import Experiment
import git
import os


PRUNE_LEVELS = [0.2, 0.4, 0.5, 0.6, 0.8]
LATENT_DIMS = [32, 64, 128, 256, 512]

ALL_SCENARIOS = ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50", "splitEmbeddedCIFAR100", "splitEmbeddedCORe50"]


def get_experiment_name(repo_hash, experiment, scenario, architecture, strategy):
    hostname = os.uname()[1]
    return f"{hostname}_{repo_hash}_{experiment}_{scenario}_{architecture}_{strategy}"

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

def choose_architecture(cfg: ExperimentConfiguration, architecture: str):
    if architecture == "AE":
        cfg = cfg.use_auto_encoder()
    elif architecture == "VAE":
        cfg = cfg.use_variational_auto_encoder()
    else:
        raise NotImplementedError(f"Unknown architecture {architecture}")
    return cfg


def prune_levels(repo_hash):
    cfg = ExperimentConfiguration()

    for strategy, architecture, scenario, prune_level in itertools.product(
            ["taskInference"],
            ["AE"],
            ["splitEmbeddedCIFAR100", "splitEmbeddedCORe50"],
            PRUNE_LEVELS):
        cfg.name = get_experiment_name(repo_hash, "PL", scenario, architecture, strategy)
        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = prune_level
        run(cfg)


def equal_prune():
    cfg = ExperimentConfiguration()

    for strategy, architecture, scenario in itertools.product(
            ["taskOracle", "taskInference"],
            ["AE", "VAE"],
            ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50"]):
        cfg.name = f"EP_{scenario}_{architecture}_{strategy}"

        cfg = choose_scenario(cfg, scenario)
        cfg = choose_architecture(cfg, architecture)
        cfg = choose_strategy(cfg, strategy)
        cfg.prune_proportion = equal_capacity_prune_schedule(cfg.n_experiences)

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()






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

def main():
    cfg = ExperimentConfiguration()

    for strategy, architecture, scenario in itertools.product(
            ["cumulative", "finetuning", "taskOracle", "taskInference"],
            ["AE", "VAE"],
            ["splitEmbeddedCIFAR100", "splitEmbeddedCORe50"]):
            # ["fmnist"]):
    # for strategy, architecture, scenario in itertools.product(
    #         ["finetuning", "taskOracle", "taskInference"],
    #         ["AE", "VAE"],
    #         ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50"]):
    #         # ["fmnist"]):
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


# Main
if __name__ == '__main__':
    # Ensure that we can only run an experiment on a clean commit to better 
    # track the results
    repo = git.Repo('')
    assert repo.is_dirty() is False, "Cannot run experiment on dirty repo"
    repo_hash = repo.head.commit.hexsha[:8]

    prune_levels(repo_hash)