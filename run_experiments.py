import itertools
from config.config import ExperimentConfiguration
from train import Experiment

def main():
    cfg = ExperimentConfiguration()

    for variant, architecture, dataset in itertools.product(
            ["transient", "finetuning", "task_oracle", "task_inference"],
            ["ae", "vae"],
            ["cifar10"]):
            # ["fmnist"]):
        cfg.name = f"{dataset}_{architecture}_{variant}"

        if dataset == "fmnist":
            cfg = cfg.fmnist()
        elif dataset == "cifar10":
            cfg = cfg.cifar10()
        else:
            raise NotImplementedError(f"Unknown dataset {dataset}")

        if architecture == "ae":
            cfg = cfg.configure_ae()
        elif architecture == "vae":
            cfg = cfg.configure_vae()
        else:
            raise NotImplementedError(f"Unknown architecture {architecture}")

        if variant == "transient":
            cfg = cfg.configure_transient()
        elif variant == "finetuning":
            cfg.use_packnet = False
        elif variant == "task_oracle":
            cfg.configure_packnet()
        elif variant == "task_inference":
            cfg.configure_packnet()
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