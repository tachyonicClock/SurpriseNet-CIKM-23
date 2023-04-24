import torch
from network.feature_extractor import r18_extractor
from experiment.experiment import BaseExperiment
import avalanche as cl
import avalanche.training.plugins as cl_plugins

from experiment.loss import BCEReconstructionLoss, MSEReconstructionLoss, MultipleObjectiveLoss, ClassifierLoss, VAELoss
from network.networks import construct_network
from surprisenet.plugin import PackNetPlugin
from surprisenet.task_inference import TaskInferenceStrategy, TaskReconstruction, UseTaskOracle
from experiment.scenario import split_scenario, gaussian_schedule_scenario
from experiment.strategy import Strategy, CumulativeTraining
from experiment.chf import CHF_SurpriseNet
from torch import nn

import surprisenet.packnet as pn

TASK_INFERENCE_STRATEGIES = {
    "task_reconstruction_loss": TaskReconstruction,
    "task_oracle": UseTaskOracle
}


class Experiment(BaseExperiment):

    def make_scenario(self) -> cl.benchmarks.NCScenario:
        """Create a scenario from the config"""

        if self.cfg.task_free:
            return gaussian_schedule_scenario(
                self.cfg.dataset_root,
                self.cfg.dataset_name,
                self.cfg.task_free_microtask_size,
                self.cfg.normalize,
            )
        else:
            return split_scenario(
                self.cfg.dataset_name,
                self.cfg.dataset_root,
                self.cfg.n_experiences,
                self.cfg.fixed_class_order,
                self.cfg.normalize)

    def make_task_inference_strategy(self) -> TaskInferenceStrategy:
        return TASK_INFERENCE_STRATEGIES[self.cfg.task_inference_strategy](self)

    def make_network(self) -> nn.Module:
        network = construct_network(self.cfg)
        # If using PackNet/SurpriseNet then wrap the network in a PackNet
        # and add a PackNetPlugin
        if self.cfg.use_packnet:
            # Wrap network in packnet
            if self.cfg.architecture == "VAE":
                network = pn.PackNetVariationalAutoEncoder(
                    network,
                    self.make_task_inference_strategy()
                )
            elif self.cfg.architecture == "AE":
                network = pn.PackNetAutoEncoder(
                    network,
                    self.make_task_inference_strategy()
                )

            self.plugins.append(
                PackNetPlugin(self.cfg.prune_proportion, self.cfg.retrain_epochs)
            )
        return network

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        # Add cross entropy loss
        if self.cfg.classifier_loss_weight:
            loss.add(ClassifierLoss(self.cfg.classifier_loss_weight))
        
        # Add reconstruction loss
        if self.cfg.reconstruction_loss_weight:
            # Pick a type of reconstruction loss
            if self.cfg.reconstruction_loss_type == "mse":
                loss.add(MSEReconstructionLoss(
                    self.cfg.reconstruction_loss_weight))
            elif self.cfg.reconstruction_loss_type == "bce":
                loss.add(BCEReconstructionLoss(
                    self.cfg.reconstruction_loss_weight))
            else:
                raise NotImplementedError("Unknown reconstruction loss type")

        # Add VAE loss
        if self.cfg.vae_loss_weight:
            loss.add(VAELoss(self.cfg.vae_loss_weight))
        return loss

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, self.cfg.learning_rate)

    def add_strategy_plugins(self):
        cfg = self.cfg
        if cfg.si_lambda:
            print("! Using Synaptic Intelligence")
            self.plugins.append(
                cl_plugins.SynapticIntelligencePlugin(cfg.si_lambda)
            )
        if cfg.lwf_alpha:
            print("! Using Learning without Forgetting")
            self.plugins.append(
                cl_plugins.LwFPlugin(cfg.lwf_alpha, temperature=2)
            )
        if cfg.replay_buffer:
            print(
                f"! Using Experience Replay. Buffer size={cfg.replay_buffer}")
            self.plugins.append(cl_plugins.ReplayPlugin(cfg.replay_buffer))
            # The replay buffer provides double the batch size. This
            # is because it provides a combined batch of old experiences and a
            # new experiences. To ensure that it fits on the GPU, we need to
            # halve the batch size.
            cfg.batch_size = cfg.batch_size//2

    def make_strategy(self) -> Strategy:
        cfg = self.cfg

        if cfg.continual_hyperparameter_framework:
            print("! Using CHF")
            assert cfg.use_packnet, "Our CHF is only compatible with SurpriseNet/PackNet"
            self.strategy_type = CHF_SurpriseNet
        elif cfg.cumulative:
            print("! Using Cumulative")
            # Replace the strategy with a cumulative strategy
            self.strategy_type = CumulativeTraining

        # Ensure that total epochs takes the retrain_epochs into account
        train_epochs = cfg.total_task_epochs - cfg.retrain_epochs \
            if cfg.use_packnet else cfg.total_task_epochs

        self.add_strategy_plugins()

        strategy = self.strategy_type(
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

        # Set CHF parameters
        if isinstance(strategy, CHF_SurpriseNet):
            strategy.set_chf_params(
                cfg.chf_validation_split_proportion,
                cfg.chf_lr_grid,
                cfg.chf_accuracy_drop_threshold,
                cfg.chf_stability_decay
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
        self.logger.writer.add_text(
            "Config", f"<pre>{self.cfg.toJSON()}</pre>")
