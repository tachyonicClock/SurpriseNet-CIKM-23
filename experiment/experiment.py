import os
import typing as t

import avalanche as av
import torch
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.evaluation.metrics import (accuracy_metrics,
                                          confusion_matrix_metrics,
                                          forgetting_metrics, loss_metrics)
from avalanche.evaluation.metric_utils import default_cm_image_creator
from avalanche.training.plugins import EvaluationPlugin
from config.config import ExpConfig
from metrics.metrics import (ConditionalMetrics, EpochClock,
                             EvalLossObjectiveMetric,
                             ExperienceIdentificationCM, LossObjectiveMetric,
                             SubsetRecognition)
from metrics.reconstructions import GenerateReconstruction, GenerateSamples
from network.trait import (NETWORK_TRAITS, AutoEncoder, Classifier,
                           ConditionedSample, InferTask, Samplable)
from setproctitle import setproctitle
from torch import Tensor, nn
from functools import partial
from matplotlib import pyplot as plt

from experiment.loss import MultipleObjectiveLoss
from experiment.strategy import ForwardOutput, Strategy


def count_parameters(model, verbose=True):
    '''
    Count number of parameters, print to screen.

    This snippet is taken from https://github.com/GMvandeVen/brain-inspired-replay
    '''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims == 0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


class BaseExperiment():
    """
    Py-lightning inspired for continual learning with avalanche
    """

    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()
        self.plugins: t.List[BasePlugin] = []
        self.strategy: Strategy
        self.network:  nn.Module
        self.logger:   av.logging.TensorboardLogger
        self.scenario: av.benchmarks.NCScenario
        self.optimizer: torch.optim.Optimizer
        self.evaluator: EvaluationPlugin
        self.objective: MultipleObjectiveLoss
        self.plugins: t.List[BasePlugin]
        self.strategy_type: t.Type[SupervisedPlugin]
        self.cfg: ExpConfig
        self.strategy_type = Strategy
        self.cfg = cfg

        os.makedirs(self.cfg.tensorboard_dir, exist_ok=True)
        self.label = f"{max(self._get_log_numbers())+1:04d}_{self.cfg.name}"
        setproctitle(os.getlogin() + "::" + self.label)

        # Create a new logger with sequential names
        self.logdir = self.cfg.tensorboard_dir+"/"+self.label
        self.logger = av.logging.TensorboardLogger(self.logdir)

        self.scenario = self.make_scenario()
        self.objective = self.make_objective()
        self.network = self.make_network()
        self.evaluator = self.make_evaluator(
            [self.logger], self.scenario.n_classes)
        self.optimizer = self.make_optimizer(self.network.parameters())
        self.strategy = self.make_strategy()

    def train(self):
        self.preflight()
        results = []
        for i, exp in enumerate(self.scenario.train_stream):
            unique_classes = set(map(int, exp.classes_in_this_experience))
            print(f"Start of experience: {exp.current_experience}")
            print(f"Current Classes:     {unique_classes}")
            print(f"Experience size:     {len(exp.dataset)}")

            self.strategy.train(exp)
            test_subset = self.scenario.test_stream

            if (i+1) % self.cfg.test_every == 0:
                results.append(self.strategy.eval(test_subset))

        self.logger.writer.flush()
        return results

    def make_strategy(self) -> SupervisedPlugin:
        return NotImplemented

    def make_criterion(self):
        def _loss_function(output: Tensor, target: Tensor) -> Tensor:
            self.objective.update(self.last_mb_output, target)
            return self.objective.sum
        return _loss_function

    def make_objective(self) -> MultipleObjectiveLoss:
        return NotImplemented

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        """Overload to define the evaluation plugin"""
        cm_image_creator = partial(default_cm_image_creator,
                                   cmap=plt.rcParams['image.cmap'])

        plugins = []
        is_images = self.cfg.is_image_data
        if isinstance(self.network, AutoEncoder) and is_images:
            plugins.append(GenerateReconstruction(self.scenario, 2, 1))

        if isinstance(self.network, Samplable) and is_images:
            rows_are_experiences = isinstance(self.network, ConditionedSample)
            plugins.append(GenerateSamples(5, 4, rows_are_experiences))

        if isinstance(self.network, InferTask) and not self.cfg.task_free:
            plugins.append(ConditionalMetrics())
            plugins.append(ExperienceIdentificationCM(self.n_experiences))

        if isinstance(self.network, InferTask):
            plugins.append(SubsetRecognition(self.cfg.n_classes))

        if isinstance(self.network, Classifier):
            plugins.append(accuracy_metrics(epoch=True, stream=True,
                           experience=True, trained_experience=True))
            plugins.append(confusion_matrix_metrics(
                normalize='true',
                image_creator=cm_image_creator,
                stream=True,
                num_classes=num_classes
            ))
            plugins.append(forgetting_metrics(experience=True, stream=True))

        # Add metric for each objective
        for name, objective in self.objective:
            plugins.append(LossObjectiveMetric(name, objective,
                           on_iteration=self.cfg.log_mini_batch))
            plugins.append(EvalLossObjectiveMetric(name, objective))

        return EvaluationPlugin(
            loss_metrics(epoch=True, experience=True,
                         stream=True, minibatch=self.cfg.log_mini_batch),
            EpochClock(),
            *plugins,
            loggers=loggers
        )

    def dump_config(self):
        pass

    def _save_class_order(self):
        class_order = []
        for task_classes in self.scenario.classes_in_experience["train"]:
            task_classes = list(set(map(int, task_classes)))
            class_order.append(task_classes)

        with open(self.logdir+"/class_order.txt", "w") as f:
            for classes in class_order:
                f.write(",".join(map(str, classes))+"\n")

    def preflight(self):
        print(f"Network: {type(self.network)}")
        print(f"Traits:")
        for trait in NETWORK_TRAITS:
            if isinstance(self.network, trait):
                print(f" > Has the `{trait.__name__}` trait")
        print(f"Objectives:")
        for name, _ in self.objective:
            print(f" > Has the `{name}` objective")

        print("-"*80)
        count_parameters(self.network)
        print("-"*80)

        print()
        self._save_class_order()
        self.dump_config()

    def make_network(self) -> nn.Module:
        raise NotImplemented

    def make_dependent_variables(self):
        return {
            "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000": 0,
            "Loss_Exp/eval_phase/test_stream/Task000/Exp000": 0,
            "EvalLossPart/Experience_0/Classifier": 0,
            "EvalLossPart/Experience_0/Reconstruction": 0
        }

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        raise NotImplemented

    def make_scenario(self) -> av.benchmarks.NCScenario:
        raise NotImplemented

    @property
    def lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def n_experiences(self) -> int:
        return len(self.scenario.train_stream)

    @property
    def n_classes(self) -> int:
        return self.scenario.n_classes

    @property
    def last_mb_output(self) -> ForwardOutput:
        return self.strategy.last_forward_output

    @property
    def clock(self):
        return self.strategy.clock

    def _get_log_numbers(self):
        for filename in os.listdir(self.cfg.tensorboard_dir):
            name, _ = os.path.splitext(filename)
            yield int(name[:4])
        yield 0
