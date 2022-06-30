from difflib import context_diff
import os
import gin
import typing as t
import click

import numpy as np
import torch
from experiment.experiment import BaseExperiment
import avalanche as cl
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
gin.external_configurable(ReconstructionLoss)
gin.external_configurable(ClassifierLoss)
gin.external_configurable(VAELoss)

gin.external_configurable(vanilla_cnn)
gin.external_configurable(wide_residual_network)
gin.external_configurable(residual_network)
gin.external_configurable(mlp_network)
gin.external_configurable(rectangular_network)

# Setup configured functions
scenario = gin.configurable(scenario, "Experiment")


random_search_hp: t.Dict[str, float] = dict()

@gin.configurable()
def uniform_rand(variable_name, var_range: t.Tuple[int, int], is_int=False) -> float:
    value = np.random.uniform(var_range[0], var_range[1])
    if is_int:
        value = int(value)
    random_search_hp[variable_name] = value
    return value

@gin.configurable
class Experiment(BaseExperiment):


    def make_scenario(self) -> cl.benchmarks.NCScenario:
        """Create a scenario from the config"""
        return scenario()

    @gin.configurable("task_oracle", "Experiment.task_inference")
    def task_oracle(self) -> TaskInferenceStrategy:
        return UseTaskOracle(self)

    @gin.configurable("reconstruction", "Experiment.task_inference")
    def task_reconstruction(self) -> TaskInferenceStrategy:
        return TaskReconstruction(self)

    @gin.configurable("packnet", "Experiment")
    def setup_packnet(self, 
            network: nn.Module,
            use_packnet: bool = False, 
            prune_proportion: float = 0.5, 
            post_prune_epochs: int = 1,
            task_inference_strategy: t.Callable[[BaseExperiment],TaskInferenceStrategy] = task_oracle
            ) -> nn.Module:

        if not use_packnet:
            return network

        if isinstance(network, VariationalAutoEncoder):
            network = pn.PackNetVariationalAutoEncoder(network, task_inference_strategy(self))
        elif isinstance(network, AutoEncoder):
            network = pn.PackNetAutoEncoder(network, task_inference_strategy(self))
        self.plugins.append(
            PackNetPlugin(network, self, prune_proportion, post_prune_epochs)
        )
        return network

    # @gin.configurable("classifier_head", "Experiment")
    # def make_classifier_head(self, latent_dims, width: int) -> nn.Module:
    #     return ClassifierHead(self.network.latent_dim, self.network.num_classes, width)

    @gin.configurable("network", "Experiment")
    def make_network(self,
        deep_generative_type: t.Literal["AE", "VAE"],
        ae_architecture: t.Callable[[t.Any], AEArchitecture]) -> nn.Module:

        ae_architecture: AEArchitecture = ae_architecture(self.n_classes, vae=deep_generative_type=="VAE")
        
        if deep_generative_type == "AE":
            return self.setup_packnet(AutoEncoder(ae_architecture.encoder, ae_architecture.decoder, ae_architecture.head))
        elif deep_generative_type == "VAE":
            bottleneck = VAEBottleneck(ae_architecture.latent_dims*2, ae_architecture.latent_dims)
            return self.setup_packnet(VariationalAutoEncoder(ae_architecture.encoder, bottleneck, ae_architecture.decoder, ae_architecture.head))
        else:
            return NotImplemented

    @gin.configurable("loss", "Experiment")
    def make_objective(self, loss_objectives: t.Sequence[LossObjective]) -> MultipleObjectiveLoss:
        loss = MultipleObjectiveLoss()
        for objective in loss_objectives:
            loss.add(objective)
        return loss

    @gin.configurable("optimizer", "Experiment", denylist=["parameters"])
    def make_optimizer(self, parameters, lr) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(parameters, lr)
        return optimizer

    @gin.configurable("train", "Experiment")
    def make_strategy(self,
        device,
        train_mb_size,
        eval_mb_size,
        eval_every,
        train_epochs) -> Strategy:
        return Strategy(
            self.network,
            self.optimizer,
            criterion=self.make_criterion(),
            device=device,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            eval_every=eval_every,
            plugins=[self, *self.plugins],
            evaluator=self.evaluator
        )

    def dump_config(self):
        with open(f"{self.logdir}/config.gin", "w") as f:
            f.write(gin.operative_config_str())

        self.logger.writer.add_text("Config", gin.markdown(gin.operative_config_str()))

    def save_checkpoint(self, checkpoint_name: str):
        torch.save({
            "network": self.network,
            "gin_config": gin.operative_config_str()
        }, f"{self.logdir}/{checkpoint_name}.pt")

    # def load_checkpoint(self, path: str):
    #     print(f"Attempting to load checkpoint {path}")
    #     checkpoint = torch.load(path)
    #     self.network = checkpoint["network"]

    # def after_training_epoch(self, strategy: Strategy, *args, **kwargs) -> "CallbackResult":
    #     self.save_checkpoint(f"experience_{self.clock.train_exp_counter:04d}")

@click.command()
@click.option("--n-runs", default=1, help="Run the configuration n times")
@click.argument("experiment_name", nargs=1)
@click.argument("gin_configs", nargs=-1, type=click.File())
def main(experiment_name: str, gin_configs: t.List[str], n_runs):
    for _ in range(n_runs):
        gin.clear_config(True)

        # Apply the configuration files in order
        for gin_config in gin_configs:
            print(f"Applying config: {gin_config}")
            gin.parse_config(gin_config)

        # Add the experiment name to the configuration
        gin.bind_parameter("Experiment.name", experiment_name)

        experiment = Experiment()
        experiment.add_hparams(random_search_hp)
        experiment.train()

if __name__ == '__main__':
    main()

# Experiment.make_network()
