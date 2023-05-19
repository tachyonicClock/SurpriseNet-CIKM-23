import typing as t

from config.config import ExpConfig
from train import Experiment

import vizier.pyvizier as vz
from vizier.benchmarks import experimenters
from vizier.client.client_abc import TrialInterface

ACC_TID = "TaskIdAccuracy"


class SurpriseNetExperimenter(experimenters.Experimenter):
    def __init__(self, base_cfg: ExpConfig) -> None:
        super().__init__()
        self.cfg = base_cfg

    def _evaluate(self, cfg: ExpConfig, suggestion: TrialInterface):
        """Evaluates a single Trial."""
        # Log the hyperparameters
        print("=-" * 40)
        for k, v in suggestion.parameters.items():
            print(f"  {k}: {v}")
        print("=-" * 40)

        # Create an experiment incorporating the suggested hyperparameters
        for k, v in suggestion.parameters.items():
            setattr(cfg, k, v)
        exp = Experiment(cfg)

        # Add the hyperparameters to TensorBoard
        exp.logger.writer.add_hparams(
            suggestion.parameters,
            {
                ACC_TID: 0.0,
            },
        )
        final_measurement = vz.Measurement({ACC_TID: 0.0})

        try:
            result = exp.train()
        except Exception:
            print()
            print(f"{cfg.name} failed")

            suggestion.complete(
                final_measurement, infeasible_reason="Error during training"
            )
            return

        final_measurement.metrics[ACC_TID] = result[-1][ACC_TID]
        suggestion.complete(final_measurement)

    def evaluate(self, suggestions: t.Sequence[vz.Trial]):
        """Evaluates and mutates the Trials in-place."""
        for suggestion in suggestions:
            self._evaluate(self.cfg.copy(), suggestion)

    def problem_statement(self) -> vz.ProblemStatement:
        """The search configuration generated by this experimenter."""

        problem = vz.ProblemStatement()
        search = problem.search_space.root

        # SEARCH SPACE --------------------------------------------------------
        search.add_discrete_param("classifier_loss_weight", [0.0])
        search.add_discrete_param("latent_dims", [8])
        search.add_float_param("learning_rate", 1e-6, 0.003)
        search.add_int_param("total_task_epochs", 180, 500)
        search.add_int_param("retrain_epochs", 50, 100)

        # OBJECTIVES ----------------------------------------------------------
        problem.metric_information.append(
            vz.MetricInformation(name=ACC_TID, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        )
        return problem
