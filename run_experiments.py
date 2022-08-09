import itertools
from config.config import ExperimentConfiguration
from train import Experiment




def main():
    cfg = ExperimentConfiguration()

    for strategy, architecture, scenario in itertools.product(
            ["taskInference"],
            ["AE", "VAE"],
            ["splitEmbeddedCIFAR100"]):
            # ["fmnist"]):
    # for strategy, architecture, scenario in itertools.product(
    #         ["finetuning", "taskOracle", "taskInference"],
    #         ["AE", "VAE"],
    #         ["splitFMNIST", "splitCIFAR10", "splitCIFAR100", "splitCORe50"]):
    #         # ["fmnist"]):
        cfg.name = f"{scenario}_{architecture}_{strategy}"

        # Select the dataset to use for the experiment
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
        else:
            raise NotImplementedError(f"Unknown scenario {scenario}")

        if architecture == "AE":
            cfg = cfg.use_auto_encoder()
        elif architecture == "VAE":
            cfg = cfg.use_variational_auto_encoder()
        else:
            raise NotImplementedError(f"Unknown architecture {architecture}")

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

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()


# Main
if __name__ == '__main__':
    main()