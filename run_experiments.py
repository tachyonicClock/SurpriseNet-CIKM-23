import itertools
from config.config import ExperimentConfiguration
from train import Experiment




def main():
    cfg = ExperimentConfiguration()

    for variant, architecture, scenario in itertools.product(
            ["finetuning", "task_oracle", "task_inference"],
            ["ae", "vae"],
            ["core50"]):
            # ["fmnist"]):
        cfg.name = f"{scenario}_{architecture}_{variant}"

        # Select the dataset to use for the experiment
        if scenario == "fmnist":
            cfg = cfg.use_fmnist()
        elif scenario == "cifar10":
            cfg = cfg.use_cifar10()
        elif scenario == "cifar100":
            cfg = cfg.use_cifar100()
        elif scenario == "core50":
            cfg = cfg.use_core50()
        else:
            raise NotImplementedError(f"Unknown scenario {scenario}")

        if architecture == "ae":
            cfg = cfg.use_auto_encoder()
        elif architecture == "vae":
            cfg = cfg.use_variational_auto_encoder()
        else:
            raise NotImplementedError(f"Unknown architecture {architecture}")

        if variant == "transient":
            cfg = cfg.use_transient_learning()
        elif variant == "finetuning":
            cfg.use_packnet = False
        elif variant == "task_oracle":
            cfg.enable_packnet()
        elif variant == "task_inference":
            cfg.enable_packnet()
            cfg.task_inference_strategy = "task_reconstruction_loss"
        else:
            raise NotImplementedError(f"Unknown variant {variant}")

        print("################")
        print(f"{cfg.name}")
        print("################")
        experiment = Experiment(cfg)
        experiment.train()


# Main
if __name__ == '__main__':
    main()