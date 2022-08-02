from difflib import context_diff
import json
import os
import typing as t
import click

import numpy as np
import torch
from experiment.experiment import BaseExperiment
import avalanche as cl
from config.config import ExperimentConfiguration
from experiment.loss import LossObjective, MultipleObjectiveLoss, ReconstructionLoss, ClassifierLoss, VAELoss
from experiment.plugins import PackNetPlugin
from experiment.scenario import scenario
from experiment.strategy import Strategy
from experiment.task_inference import TaskInferenceStrategy, TaskReconstruction, UseTaskOracle
from network.mlp import MPLRectangularClassifierHead
from network.trait import AutoEncoder, Classifier, Decoder, Encoder, VariationalAutoEncoder
from torch import nn
from network.architectures import *
import network.module.packnet as pn
from network.vanilla_cnn import ClassifierHead, VAEBottleneck

# Make loss parts configurable
# gin.external_configurable(ReconstructionLoss)
# gin.external_configurable(ClassifierLoss)
# gin.external_configurable(VAELoss)

# gin.external_configurable(vanilla_cnn)
# gin.external_configurable(wide_residual_network)
# gin.external_configurable(residual_network)
# gin.external_configurable(mlp_network)
# gin.external_configurable(rectangular_network)

# Setup configured functions
# scenario = gin.configurable(scenario, "Experiment")


random_search_hp: t.Dict[str, float] = dict()

def uniform_rand(variable_name, var_range: t.Tuple[int, int], is_int=False) -> float:
    value = np.random.uniform(var_range[0], var_range[1])
    if is_int:
        value = int(value)
    random_search_hp[variable_name] = value
    return value

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
            PackNetPlugin(network, self, self.cfg.prune_proportion, self.cfg.retrain_epochs)
        )
        return network

    # def make_classifier_head(self, latent_dims, width: int) -> nn.Module:
    #     return ClassifierHead(self.network.latent_dim, self.network.num_classes, width)

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
        

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        if self.cfg.use_classifier_loss:
            loss.add(ClassifierLoss(self.cfg.classifier_loss_weight))
        if self.cfg.use_reconstruction_loss:
            loss.add(ReconstructionLoss(self.cfg.reconstruction_loss_weight))
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

        return Strategy(
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

    def dump_config(self):
        with open(f"{self.logdir}/config.json", "w") as f:
            f.writelines(self.cfg.toJSON())
        self.logger.writer.add_text("Config", f"<pre>{self.cfg.toJSON()}</pre>")

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

    if variant == "transient":
        cfg = cfg.configure_transient()
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

# Experiment.make_network()
