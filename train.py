import gin
import typing as t

import torch
from experiment.experiment import BaseExperiment
import avalanche as cl
from experiment.loss import LossObjective, MultipleObjectiveLoss, ReconstructionLoss, ClassifierLoss, VAELoss
from experiment.scenario import scenario
from experiment.strategy import Strategy
from network.trait import AutoEncoder, Decoder, Encoder
from torch import nn
from network.architectures import *

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

    @gin.configurable("network", "Experiment")
    def make_network(self,
        deep_generative_type: t.Literal["AE", "VAE"],
        network_architecture: t.Callable[[t.Any], t.Tuple[Encoder, Decoder]]) -> nn.Module:

        encoder, decoder = network_architecture()

        if deep_generative_type == "AE":
            return AutoEncoder(encoder, decoder)
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
            plugins=[self, *self.add_plugins()],
            evaluator=self.evaluator
        )


    def dump_config(self):
        with open(f"{self.logdir}/config.gin", "w") as f:
            f.write(gin.operative_config_str())

        self.logger.writer.add_text("Config", gin.markdown(gin.operative_config_str()))
        



gin.parse_config_file("/Scratch/al183/dynamic-dropout/config/base.gin")
Experiment().train()



# Experiment.make_network()
