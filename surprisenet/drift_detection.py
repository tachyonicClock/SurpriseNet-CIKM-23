import numbers
from avalanche.core import SupervisedPlugin
import typing as t

from experiment.loss import LossObjective
from network.trait import PackNet
from click import secho

class DriftDetector(t.Protocol):

    @property
    def drift_detected(self) -> bool:
        pass

    @property
    def warning_detected(self) -> bool:
        pass

    def update(self, x: numbers.Number, strategy: t.Any):
        pass


class ClockOracle(DriftDetector):

    def __init__(self, drift_period: int, drift_in_advance: int, warn_in_advance: int) -> None:
        """The `ChunkOracle` cheats by looking at the clock to determine if a
        drift should have occurred or not.

        :param clock: An Avalanche clock
        :param drift_period: The number of microtasks to train before
            triggering a drift
        :param warn_duration: The number of microtasks to train before
            triggering a warning
        """
        super().__init__()
        self.drift_period = drift_period
        self.warn_in_advance = warn_in_advance
        self.drift_in_advance = drift_in_advance
        self.task_counter = 0
        self.last_triggered = 0

    @property
    def drift_detected(self) -> bool:
        # Ensure we are not triggered multiple times
        if self.task_counter == self.last_triggered:
            return False
        if self.task_counter % self.drift_period == self.drift_period - self.drift_in_advance:
            self.last_triggered = self.task_counter
            return True

    @property
    def warning_detected(self) -> bool:
        # Ensure we are not triggered multiple times
        if self.task_counter == self.last_triggered:
            return False
    
        if self.task_counter % self.drift_period == self.drift_period - self.warn_in_advance:
            self.last_triggered = self.task_counter
            return True

    def update(self, x: numbers.Number, strategy: t.Any):
        self.task_counter = strategy.clock.train_exp_counter + 1


class DriftHandler(t.Protocol):

    def on_drift_warning(self, strategy):
        print("- "*40)
        secho(" ==> RECEIVED DRIFT WARNING", fg="yellow")

    def on_drift(self, strategy):
        print("- "*40)
        secho(" ==> RECEIVED DRIFT", fg="yellow")


class SurpriseNetDriftHandler(DriftHandler):

    def __init__(self, prune_amount: float) -> None:
        super().__init__()
        self.prune_amount = prune_amount
        self.capacity = 1.0

    def on_drift_warning(self, strategy):
        super().on_drift_warning(strategy)
        self.capacity *= self.prune_amount
        network: PackNet = strategy.model

        secho(f"Pruning {self.prune_amount*100:0.1f}% reclaiming {self.capacity*100:0.1f}% capacity", fg="green")
        network.prune(self.capacity)
        strategy.reset_optimizer()


    def on_drift(self, strategy):
        super().on_drift(strategy)
        secho("Freezing Task-Specific Subset", fg="green")
        network: PackNet = strategy.model
        network.push_pruned()
        strategy.reset_optimizer()


class DriftDetectorPlugin(SupervisedPlugin):

    def __init__(self, detector: DriftDetector, metric: LossObjective, on_drift: DriftHandler):
        super().__init__()
        self.detector = detector
        self.metric = metric
        self.on_drift = on_drift

    def after_training_iteration(self, strategy, *args, **kwargs):
        if self.detector.drift_detected:
            self.on_drift.on_drift(strategy)
        elif self.detector.warning_detected:
            self.on_drift.on_drift_warning(strategy)
        self.detector.update(float(self.metric.loss), strategy)

    def after_training_epoch(self, strategy, *args, **kwargs):
        self.after_training_iteration(strategy, *args, **kwargs)

