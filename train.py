import torch
from avalanche.benchmarks import NCScenario
from torch import nn

import surprisenet.packnet as pn
from experiment.experiment import BaseExperiment
from experiment.loss import (
    BCEReconstructionLoss,
    CrossEntropy,
    MSEReconstructionLoss,
    MultipleObjectiveLoss,
    SurpriseNetLoss,
    VAELoss,
)
from experiment.scenario import split_scenario
from experiment.strategy import Strategy
from network.feature_extractor import r18_extractor
from network.networks import construct_network
from surprisenet.plugin import SurpriseNetPlugin
from surprisenet.task_inference import (
    TaskInferenceStrategy,
    TaskReconstruction,
    UseTaskOracle,
)


class Experiment(BaseExperiment):
    def make_scenario(self) -> NCScenario:
        """Create a scenario from the config"""
        return split_scenario(
            self.cfg.dataset_name,
            self.cfg.dataset_root,
            self.cfg.n_experiences,
            self.cfg.fixed_class_order,
            self.cfg.normalize,
        )

    def make_task_inference_strategy(self) -> TaskInferenceStrategy:
        if self.cfg.task_inference_strategy == "task_reconstruction_loss":
            return TaskReconstruction(self)
        elif self.cfg.task_inference_strategy == "task_oracle":
            return UseTaskOracle(self)

    def make_network(self) -> nn.Module:
        network = construct_network(self.cfg)
        # If not using PackNet/SurpriseNet then return the network
        if not self.cfg.use_packnet:
            return network

        # Wrap network in packnet
        if self.cfg.architecture == "VAE":
            network = pn.SurpriseNetVariationalAutoEncoder(
                network,
                self.make_task_inference_strategy(),
            )
        elif self.cfg.architecture == "AE":
            network = pn.SurpriseNetAutoEncoder(
                network,
                self.make_task_inference_strategy(),
            )
        else:
            raise ValueError("Unknown architecture")

        self.plugins.append(
            SurpriseNetPlugin(self.cfg.prune_proportion, self.cfg.retrain_epochs)
        )
        return network

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        # Add cross entropy loss
        if self.cfg.classifier_loss_weight:
            if self.cfg.classifier_loss_type == "CrossEntropy":
                loss.add(
                    CrossEntropy(
                        self.cfg.classifier_loss_weight,
                        **self.cfg.classifier_loss_kwargs,
                    )
                )
            else:
                raise ValueError("Unknown classifier loss type")

        # Add reconstruction loss
        if self.cfg.reconstruction_loss_weight:
            # Pick a type of reconstruction loss
            if self.cfg.reconstruction_loss_type == "mse":
                loss.add(MSEReconstructionLoss(self.cfg.reconstruction_loss_weight))
            elif self.cfg.reconstruction_loss_type == "bce":
                loss.add(BCEReconstructionLoss(self.cfg.reconstruction_loss_weight))
            elif self.cfg.reconstruction_loss_type == "SurpriseNetLoss":
                relative_mse = SurpriseNetLoss(self.cfg.reconstruction_loss_weight)
                loss.add(relative_mse)
                self.plugins.append(relative_mse)
            else:
                raise NotImplementedError(
                    "Unknown reconstruction loss type",
                    self.cfg.reconstruction_loss_type,
                )

        # Add VAE loss
        if self.cfg.vae_loss_weight:
            loss.add(VAELoss(self.cfg.vae_loss_weight))
        return loss

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        if self.cfg.optimizer == "Adam":
            return torch.optim.Adam(parameters, self.cfg.learning_rate)
        elif self.cfg.optimizer == "SGD":
            return torch.optim.SGD(parameters, self.cfg.learning_rate)
        else:
            raise ValueError("Unknown optimizer")

    def make_strategy(self) -> Strategy:
        cfg = self.cfg

        # Ensure that total epochs takes the retrain_epochs into account
        train_epochs = (
            cfg.total_task_epochs - cfg.retrain_epochs
            if cfg.use_packnet
            else cfg.total_task_epochs
        )

        strategy = self.strategy_type(
            self.network,
            self.optimizer,
            criterion=self.make_criterion(),
            device=cfg.device,
            train_mb_size=int(cfg.batch_size),
            train_epochs=int(train_epochs),
            eval_mb_size=int(cfg.batch_size),
            eval_every=-1,
            plugins=[self, *self.plugins],
            evaluator=self.evaluator,
        )

        # Enable a feature extractor
        if cfg.embedding_module == "ResNet18":
            print("! Using ResNet18 as embedding module")
            strategy.batch_transform = r18_extractor().to(cfg.device)
        elif cfg.embedding_module != "None":
            raise NotImplementedError("Unknown embedding module")

        return strategy

    def dump_config(self):
        with open(f"{self.logdir}/config.json", "w") as f:
            f.writelines(self.cfg.toJSON())
        self.logger.writer.add_text("Config", f"<pre>{self.cfg.toJSON()}</pre>")
