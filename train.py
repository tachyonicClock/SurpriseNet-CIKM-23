from difflib import context_diff
import sys
import gin
import typing as t

import torch
from experiment.experiment import BaseExperiment
import avalanche as cl
from experiment.loss import LossObjective, MultipleObjectiveLoss, ReconstructionLoss, ClassifierLoss, VAELoss
from experiment.plugins import PackNetPlugin
from experiment.scenario import scenario
from experiment.strategy import Strategy
from experiment.task_inference import TaskInferenceStrategy, TaskReconstruction, UseTaskOracle
from network.trait import AutoEncoder, Decoder, Encoder
from torch import nn
from network.architectures import *
import network.module.packnet as pn

# Make loss parts configurable
gin.external_configurable(ReconstructionLoss)
gin.external_configurable(ClassifierLoss)
gin.external_configurable(VAELoss)

gin.external_configurable(vanilla_cnn)
gin.external_configurable(wide_residual_network)


# Setup configured functions
scenario = gin.configurable(scenario, "Experiment")

@gin.configurable
class Experiment(BaseExperiment):


    def make_scenario(self) -> cl.benchmarks.NCScenario:
        """Create a scenario from the config"""
        return scenario()


    @gin.configurable("task_oracle", "Experiment.task_inference")
    def task_oracle(self) -> TaskInferenceStrategy:
        return UseTaskOracle(self)

    @gin.configurable("reconstruction", "Experiment.task_inference")
    def task_oracle(self) -> TaskInferenceStrategy:
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

        if isinstance(network, AutoEncoder):
            network = pn.PackNetAutoEncoder(network, task_inference_strategy(self))
        else:
            assert False, "Couldn't figure out how to make it a pack net"
            
        self.plugins.append(
            PackNetPlugin(network, self, prune_proportion, post_prune_epochs)
        )
        return network

    @gin.configurable("network", "Experiment")
    def make_network(self,
        deep_generative_type: t.Literal["AE", "VAE"],
        network_architecture: t.Callable[[t.Any], t.Tuple[Encoder, Decoder]]) -> nn.Module:

        encoder, decoder = network_architecture()
        

        if deep_generative_type == "AE":
            return self.setup_packnet(AutoEncoder(encoder, decoder))
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

    # def save_checkpoint(self, checkpoint_name: str):
    #     torch.save({
    #         "network": self.network,
    #         "gin_config": gin.operative_config_str()
    #     }, f"{self.logdir}/{checkpoint_name}.pt")

    # def load_checkpoint(self, path: str):
    #     print(f"Attempting to load checkpoint {path}")
    #     checkpoint = torch.load(path)
    #     self.network = checkpoint["network"]

    # def after_eval_exp(self, strategy: Strategy, *args, **kwargs) -> "CallbackResult":
    #     self.save_checkpoint(f"experience_{self.clock.train_exp_counter:04d}")



gin.parse_config_file("/Scratch/al183/dynamic-dropout/config/base.gin")
Experiment().train()


# Experiment.make_network()
