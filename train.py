import torch
from network.feature_extractor import ResNet50FeatureExtractor
from experiment.experiment import BaseExperiment
import avalanche as cl
import avalanche.training.plugins as cl_plugins

from experiment.loss import BCEReconstructionLoss, MSEReconstructionLoss, MultipleObjectiveLoss, ClassifierLoss, VAELoss
from network.networks import construct_network
from packnet.plugin import PackNetPlugin
from packnet.task_inference import TaskInferenceStrategy, TaskReconstruction, UseTaskOracle
from experiment.scenario import scenario
from experiment.strategy import Strategy
from torch import nn

import packnet.packnet as pn

TASK_INFERENCE_STRATEGIES = {
    "task_reconstruction_loss": TaskReconstruction,
    "task_oracle": UseTaskOracle
}

class Experiment(BaseExperiment):

    def make_scenario(self) -> cl.benchmarks.NCScenario:
        """Create a scenario from the config"""
        return scenario(
            self.cfg.dataset_name,
            self.cfg.dataset_root,
            self.cfg.n_experiences,
            self.cfg.fixed_class_order)

    def make_task_inference_strategy(self) -> TaskInferenceStrategy:
        return TASK_INFERENCE_STRATEGIES[self.cfg.task_inference_strategy](self)

    def setup_packnet(self, network: nn.Module) -> nn.Module:
        """Setup packnet"""

        if not self.cfg.use_packnet:
            return network

        # Wrap network in packnet
        if self.cfg.deep_generative_type == "VAE":
            network = pn.PackNetVariationalAutoEncoder(
                network,
                self.make_task_inference_strategy()
            )
        elif self.cfg.deep_generative_type == "AE":
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
        network = construct_network(self.cfg)
        return self.setup_packnet(network)

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        # Add cross entropy loss
        if self.cfg.use_classifier_loss:
            loss.add(ClassifierLoss(self.cfg.classifier_loss_weight))
        # Add reconstruction loss
        if self.cfg.use_reconstruction_loss:

            # Pick a type of reconstruction loss
            if self.cfg.recon_loss_type == "mse":
                loss.add(MSEReconstructionLoss(
                    self.cfg.reconstruction_loss_weight))
            elif self.cfg.recon_loss_type == "bce":
                loss.add(BCEReconstructionLoss(
                    self.cfg.reconstruction_loss_weight))
            else:
                raise NotImplementedError("Unknown reconstruction loss type")
        # Add VAE loss
        if self.cfg.use_vae_loss:
            loss.add(VAELoss(self.cfg.vae_loss_weight))
        return loss

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        if self.cfg.use_adam:
            optimizer = torch.optim.Adam(parameters, self.cfg.learning_rate)
        else:
            optimizer = torch.optim.SGD(parameters, self.cfg.learning_rate)
        return optimizer

    def add_strategy_plugins(self):
        cfg = self.cfg
        if cfg.use_synaptic_intelligence:
            print("! Using Synaptic Intelligence")
            self.plugins.append(
                cl_plugins.SynapticIntelligencePlugin(cfg.si_lambda)
            )
        if cfg.use_learning_without_forgetting:
            print("! Using Learning without Forgetting")
            self.plugins.append(
                cl_plugins.LwFPlugin(cfg.lwf_alpha, temperature=1)
            )
        if cfg.use_experience_replay:
            print(f"! Using Experience Replay. Buffer size={cfg.replay_buffer}")
            self.plugins.append(
                cl_plugins.ReplayPlugin(
                    cfg.replay_buffer, 
                    storage_policy=cl.training.storage_policy.ClassBalancedBuffer(cfg.replay_buffer))
            )

    def make_strategy(self) -> Strategy:
        cfg = self.cfg

        # Ensure that total epochs takes the retrain_epochs into account
        train_epochs = cfg.total_task_epochs - cfg.retrain_epochs \
            if cfg.use_packnet else cfg.total_task_epochs

        self.add_strategy_plugins()

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


        if cfg.use_generative_replay:
            # This is a little hacky because our VAE is connected to the
            # classifier. `GenerativeReplayPlugin` needs a strategy but we
            # dont have the strategy until after. Therefore we append the plugin
            # after the strategy is created.
            print("! Using Deep Generative Replay")
            strategy.plugins.append(
                cl_plugins.GenerativeReplayPlugin(
                    generator_strategy=strategy,
                    increasing_replay_size=True
                )
            )

        if cfg.embedding_module == "None":
            return strategy
        elif cfg.embedding_module == "ResNet50":
            strategy.batch_transform = ResNet50FeatureExtractor().to(cfg.device)
            return strategy
        else:
            raise NotImplementedError("Unknown embedding module")

    def dump_config(self):
        with open(f"{self.logdir}/config.json", "w") as f:
            f.writelines(self.cfg.toJSON())
        self.logger.writer.add_text(
            "Config", f"<pre>{self.cfg.toJSON()}</pre>")
