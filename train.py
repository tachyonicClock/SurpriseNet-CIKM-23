import typing as t
import click

import numpy as np
import torch
from network.embedding_network import ResNet18FeatureExtractor
from experiment.experiment import BaseExperiment
import avalanche as cl
from config.config import ExperimentConfiguration
from experiment.loss import BCEReconstructionLoss, MSEReconstructionLoss, MultipleObjectiveLoss, ClassifierLoss, VAELoss
from packnet.plugin import PackNetPlugin
from packnet.task_inference import TaskInferenceStrategy, TaskReconstruction, UseTaskOracle
from experiment.scenario import scenario
from experiment.strategy import Strategy
from network.trait import AutoEncoder, VariationalAutoEncoder
from torch import nn
from network.architectures import *
import packnet.packnet as pn


class Experiment(BaseExperiment):

    def make_scenario(self) -> cl.benchmarks.NCScenario:
        """Create a scenario from the config"""
        return scenario(
            self.cfg.dataset_name,
            self.cfg.dataset_root,
            self.cfg.n_experiences,
            self.cfg.fixed_class_order)

    def make_task_inference_strategy(self) -> TaskInferenceStrategy:
        if self.cfg.task_inference_strategy == "task_oracle":
            return UseTaskOracle(self)
        elif self.cfg.task_inference_strategy == "task_reconstruction_loss":
            return TaskReconstruction(self)
        else:
            raise NotImplementedError("Unknown task inference strategy")

    def setup_packnet(self, network: nn.Module) -> nn.Module:
        """Setup packnet"""

        if not self.cfg.use_packnet:
            return network

        # Wrap network in packnet
        if isinstance(network, VariationalAutoEncoder):
            network = pn.PackNetVariationalAutoEncoder(
                network,
                self.make_task_inference_strategy()
            )
        elif isinstance(network, AutoEncoder):
            network = pn.PackNetAutoEncoder(
                network,
                self.make_task_inference_strategy()
            )

        self.plugins.append(
            PackNetPlugin(network, self, self.cfg.prune_proportion,
                          self.cfg.retrain_epochs)
        )
        return network

    def make_network(self) -> nn.Module:
        architecture = self.cfg.network_architecture
        is_vae = self.cfg.deep_generative_type == "VAE"

        if architecture == "vanilla_cnn":
            assert self.cfg.vanilla_cnn_config is not None, \
                "Vanilla CNN config must be provided when using vanilla CNN"
            vanilla_cnn_config = self.cfg.vanilla_cnn_config
            network = vanilla_cnn(
                self.n_classes,
                self.cfg.input_shape[0],
                self.cfg.latent_dims,
                vanilla_cnn_config.base_channels,
                is_vae)
            return self.setup_packnet(network)
        elif architecture == "residual_network":
            network = residual_network(
                self.n_classes,
                self.cfg.latent_dims,
                self.cfg.input_shape,
                is_vae)
            return self.setup_packnet(network)
        elif architecture == "mlp":
            network = mlp_network(
                self.n_classes,
                self.cfg.latent_dims,
                self.cfg.input_shape,
                is_vae)
            return self.setup_packnet(network)
        elif architecture == "rectangular_network":
            network = rectangular_network(
                self.n_classes,
                self.cfg.latent_dims,
                self.cfg.input_shape,
                depth=4, width=1024,
                is_vae=is_vae)
            return self.setup_packnet(network)

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        if self.cfg.use_classifier_loss:
            loss.add(ClassifierLoss(self.cfg.classifier_loss_weight))
        if self.cfg.use_reconstruction_loss:

            # Pick a type of reconstruction loss
            if self.cfg.recon_loss_type == "mse":
                loss.add(MSEReconstructionLoss(self.cfg.reconstruction_loss_weight))
            elif self.cfg.recon_loss_type == "bce":
                loss.add(BCEReconstructionLoss(self.cfg.reconstruction_loss_weight))
            else:
                raise NotImplementedError("Unknown reconstruction loss type")

        if self.cfg.use_vae_loss:
            loss.add(VAELoss(self.cfg.vae_loss_weight))
        return loss

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(parameters, self.cfg.learning_rate)
        return optimizer

    def make_strategy(self) -> Strategy:
        cfg = self.cfg

        # Ensure that total epochs takes the retrain_epochs into account
        train_epochs = cfg.total_task_epochs - cfg.retrain_epochs \
            if cfg.use_packnet else cfg.total_task_epochs

        strategy = Strategy(
            self.network,
            self.optimizer,
            criterion=self.make_criterion(),
            device=cfg.device,
            train_mb_size=cfg.batch_size,
            train_epochs=train_epochs,
            eval_mb_size=cfg.batch_size,
            eval_every=-1,
            plugins=[self, *self.plugins],
            evaluator=self.evaluator
        )

        if cfg.embedding_module == "None":
            return strategy
        elif cfg.embedding_module == "ResNet18":
            strategy.batch_transform = ResNet18FeatureExtractor().to(cfg.device)
            return strategy
        else:
            raise NotImplementedError("Unknown embedding module")


    def dump_config(self):
        with open(f"{self.logdir}/config.json", "w") as f:
            f.writelines(self.cfg.toJSON())
        self.logger.writer.add_text(
            "Config", f"<pre>{self.cfg.toJSON()}</pre>")

    # def save_checkpoint(self, checkpoint_name: str):
    #     torch.save({
    #         "network": self.network,
    #         "gin_config": gin.operative_config_str()
    #     }, f"{self.logdir}/{checkpoint_name}.pt")

    # def load_checkpoint(self, path: str):
    #     print(f"Attempting to load checkpoint {path}")
    #     checkpoint = torch.load(path)
    #     self.network = checkpoint["network"]

    # def after_training_epoch(self, strategy: Strategy, *args, **kwargs) -> "CallbackResult":
    #     self.save_checkpoint(f"experience_{self.clock.train_exp_counter:04d}")


@click.command()
@click.argument("dataset", nargs=1)
@click.argument("architecture", nargs=1)
@click.argument("variant", nargs=1)
def main(dataset, architecture, variant):
    cfg = ExperimentConfiguration()
    cfg.name = f"{dataset}_{architecture}_{variant}"

    if dataset == "fmnist":
        cfg = cfg.configure_fmnist()
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    if architecture == "ae":
        cfg = cfg.configure_ae()
    elif architecture == "vae":
        cfg = cfg.configure_vae()
    else:
        raise NotImplementedError(f"Unknown architecture {architecture}")

    if variant == "cumulative":
        cfg = cfg.configure_cumulative()
    elif variant == "finetuning":
        cfg.use_packnet = False
    elif variant == "task_oracle":
        cfg.enable_packnet()
    elif variant == "task_inference":
        cfg.enable_packnet()
        cfg.task_inference_strategy = "task_reconstruction_loss"
    else:
        raise NotImplementedError(f"Unknown variant {variant}")

    experiment = Experiment(cfg)
    experiment.train()

    # Get all configurable parameters


if __name__ == '__main__':
    main()
