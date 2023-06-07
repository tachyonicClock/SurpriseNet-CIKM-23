import surprisenet.packnet as pn
import torch
from avalanche.benchmarks import NCScenario
from avalanche.training.plugins import (
    LwFPlugin,
    ReplayPlugin,
    SynapticIntelligencePlugin,
)
from experiment.chf import CHF_SurpriseNet
from experiment.experiment import BaseExperiment
from experiment.loss import (
    BCEReconstructionLoss,
    CrossEntropy,
    ClassifierLossMasked,
    LogitNorm,
    LossObjective,
    MSEReconstructionLoss,
    MultipleObjectiveLoss,
    SurpriseNetLoss,
    VAELoss,
)
from experiment.scenario import gaussian_schedule_scenario, split_scenario
from experiment.strategy import CumulativeTraining, Strategy
from network.deep_vae import DeepVAELoss
from network.feature_extractor import r18_extractor
from network.networks import construct_network
from surprisenet.activation import (
    ActivationStrategy,
    NaiveSurpriseNetActivation,
    SurpriseNetTreeActivation,
)
from surprisenet.drift_detection import (
    ClockOracle,
    DriftDetector,
    DriftDetectorPlugin,
    SurpriseNetDriftHandler,
)
from surprisenet.plugin import SurpriseNetPlugin
from surprisenet.task_inference import (
    HierarchicalVAEOOD,
    TaskInferenceStrategy,
    TaskReconstruction,
    UseTaskOracle,
)
from avalanche.training.storage_policy import ClassBalancedBuffer
from torch import nn


class Experiment(BaseExperiment):
    def make_scenario(self) -> NCScenario:
        """Create a scenario from the config"""

        if self.cfg.task_free:
            return gaussian_schedule_scenario(
                dataset_root=self.cfg.dataset_root,
                dataset=self.cfg.dataset_name,
                instances_in_task=self.cfg.task_free_instances_in_task,
                width=self.cfg.task_free_width,
                microtask_count=self.cfg.n_experiences,
                normalize=self.cfg.normalize,
            )
        else:
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
        elif self.cfg.task_inference_strategy == "log_likelihood_ratio":
            return HierarchicalVAEOOD(**self.cfg.task_inference_strategy_kwargs)
        elif self.cfg.task_inference_strategy == "task_oracle":
            return UseTaskOracle(self)

    def make_activation_strategy(self) -> ActivationStrategy:
        if self.cfg.activation_strategy == "SurpriseNetTreeActivation":
            return SurpriseNetTreeActivation(self.logger.writer)
        elif self.cfg.activation_strategy == "NaiveSurpriseNetActivation":
            return NaiveSurpriseNetActivation()
        else:
            raise ValueError("Unknown activation strategy")

    def make_drift_detection_plugin(self) -> DriftDetectorPlugin:
        # Get the metric used for reconstruction loss
        metric: LossObjective
        if self.cfg.reconstruction_loss_type == "mse":
            metric = self.objective.objectives["MSEReconstruction"]
        elif self.cfg.reconstruction_loss_type == "bce":
            metric = self.objective.objectives["BCEReconstruction"]

        # Construct the drift detector
        detector: DriftDetector
        if self.cfg.task_free_drift_detector == "clock_oracle":
            detector = ClockOracle(**self.cfg.task_free_drift_detector_kwargs)
        else:
            raise ValueError("Unknown drift detector")

        # What todo when a drift is detected
        drift_handler = SurpriseNetDriftHandler(self.cfg.prune_proportion)

        # Construct the plugin
        return DriftDetectorPlugin(detector, metric, drift_handler)

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
                self.make_activation_strategy(),
            )
        elif self.cfg.architecture == "AE":
            network = pn.SurpriseNetAutoEncoder(
                network,
                self.make_task_inference_strategy(),
                self.make_activation_strategy(),
            )
        elif self.cfg.architecture == "DeepVAE":
            network = pn.SurpriseNetDeepVAE(
                network,
                self.make_task_inference_strategy(),
                self.make_activation_strategy(),
            )
        else:
            raise ValueError("Unknown architecture")

        if self.cfg.task_free:
            self.plugins.append(self.make_drift_detection_plugin())
        else:
            self.plugins.append(
                SurpriseNetPlugin(self.cfg.prune_proportion, self.cfg.retrain_epochs)
            )
        return network

    def make_objective(self) -> MultipleObjectiveLoss:
        """Create a loss objective from the config"""
        loss = MultipleObjectiveLoss()
        # Add cross entropy loss
        if self.cfg.classifier_loss_weight:
            if self.cfg.classifier_loss_type == "LogitNorm":
                loss.add(
                    LogitNorm(
                        self.cfg.classifier_loss_weight,
                        **self.cfg.classifier_loss_kwargs,
                    )
                )
            elif self.cfg.classifier_loss_type == "CrossEntropy":
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
            elif self.cfg.reconstruction_loss_type == "DeepVAE_ELBO":
                deep_vae_loss = DeepVAELoss(
                    logger=self.logger,
                    **self.cfg.hvae_loss_kwargs,
                )
                # DeepVAELoss contains schedules requiring callbacks to be
                # called at the end of each epoch
                self.plugins.append(deep_vae_loss)
                loss.add(deep_vae_loss)
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

    def add_strategy_plugins(self):
        cfg = self.cfg
        if cfg.si_lambda:
            print("! Using Synaptic Intelligence")
            self.plugins.append(SynapticIntelligencePlugin(cfg.si_lambda))
        if cfg.lwf_alpha:
            print("! Using Learning without Forgetting")
            self.plugins.append(LwFPlugin(cfg.lwf_alpha, temperature=2))
        if cfg.replay_buffer:
            print(f"! Using Experience Replay. Buffer size={cfg.replay_buffer}")
            self.plugins.append(
                ReplayPlugin(
                    cfg.replay_buffer,
                    storage_policy=ClassBalancedBuffer(cfg.replay_buffer),
                )
            )
            # The replay buffer provides double the batch size. This
            # is because it provides a combined batch of old experiences and a
            # new experiences. To ensure that it fits on the GPU, we need to
            # halve the batch size.
            cfg.batch_size = cfg.batch_size // 2

    def make_strategy(self) -> Strategy:
        cfg = self.cfg

        if cfg.continual_hyperparameter_framework:
            print("! Using CHF")
            assert (
                cfg.use_packnet
            ), "Our CHF is only compatible with SurpriseNet/PackNet"
            self.strategy_type = CHF_SurpriseNet
        elif cfg.cumulative:
            print("! Using Cumulative")
            # Replace the strategy with a cumulative strategy
            self.strategy_type = CumulativeTraining

        # Ensure that total epochs takes the retrain_epochs into account
        train_epochs = (
            cfg.total_task_epochs - cfg.retrain_epochs
            if cfg.use_packnet
            else cfg.total_task_epochs
        )

        self.add_strategy_plugins()

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
        strategy.loader_workers = cfg.loader_workers

        # Set CHF parameters
        if isinstance(strategy, CHF_SurpriseNet):
            strategy.set_chf_params(
                cfg.chf_validation_split_proportion,
                cfg.chf_lr_grid,
                cfg.chf_accuracy_drop_threshold,
                cfg.chf_stability_decay,
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
