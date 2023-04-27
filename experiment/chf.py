
import copy
import itertools
import typing as t

import torch
from avalanche.benchmarks.scenarios.new_classes import NCExperience
from avalanche.evaluation.metrics.accuracy import StreamAccuracy
from surprisenet.plugin import SurpriseNetPlugin

from experiment.strategy import Strategy


def _split_experience(experience: NCExperience, split_proportion: float) \
        -> t.Tuple[NCExperience, NCExperience]:
    """
    Split an experience into two mutually exclusive subsets.

    Returns a tuple containing (split_proportion, 1-split_proportion) number
    of samples from the experience.
    """
    n_samples = len(experience.dataset)
    n_train = int(n_samples * (1 - split_proportion))
    n_val = n_samples - n_train

    train_indices, val_indices = torch.utils.data.random_split(
        range(n_samples), [n_train, n_val])

    train_exp = NCExperience(experience.origin_stream,
                             experience.current_experience)
    val_exp = NCExperience(experience.origin_stream,
                           experience.current_experience)
    train_exp.dataset = experience.dataset.subset(train_indices)
    val_exp.dataset = experience.dataset.subset(val_indices)

    return val_exp, train_exp


def partition(pred: t.Callable[[bool], t.Any], iterable: t.Iterable[t.Any]):
    t1, t2 = itertools.tee(iterable)
    return list(itertools.filterfalse(pred, t1)), list(filter(pred, t2))


class CHF_SurpriseNet(Strategy):
    """Continual Hyper-Parameter Framework (CHF) for continual learning.

    This module overloads Strategy to allow for hyper-parameter
    optimization. It is based on the CHF framework by Delange et al. (2021).


    The use of SurpriseNet with CHF raised the question of whether the 
    parameters in Maximal Plasticity Search (MPS) should be kept frozen. If the
    parameters are unfrozen, it creates a problem where Stability Decay (SD)
    cannot be adjusted enough to match MPS. This is because, even though SD
    decreases stability, it cannot reach maximum plasticity if the parameters
    remain frozen. Therefore, MPS had to be frozen for CHF to work properly.

    SurpriseNet only decides on the stability vs plasticity trade-off when it 
    prunes the model, unlike other methods that decide the tradeoff during
    training. SD only needs the prune and retrain steps to tune the trade-off.
    This insight decreases the time of CHF for SurpriseNet.

    Delange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., 
    Slabaugh, G., & Tuytelaars, T. (2021). A continual learning survey: 
    Defying forgetting in classification tasks. IEEE Transactions on Pattern 
    Analysis and Machine Intelligence, 1–1. 
    https://doi.org/10.1109/TPAMI.2021.3057446
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chf_validation_split_proportion: float
        """What fraction of the experience should be used for validation"""
        self.chf_lr_grid: t.List[float]
        """Learning rate grid to search over during maximal plasticity search"""
        self.chf_accuracy_drop_threshold: float
        """Threshold for acceptable accuracy drop"""
        self.chf_stability_decay: float
        """How quickly the stability decays during stability decay search"""
        self.pack_net_plugin: SurpriseNetPlugin
        """The PackNet plugin used to learn continually"""
        self.metric = StreamAccuracy()
        """The metric used to evaluate the model during CHF"""

        for plugin in self.plugins:
            if isinstance(plugin, SurpriseNetPlugin):
                self.pack_net_plugin = plugin
        assert self.pack_net_plugin is not None, \
            "PackNetPlugin is required for SurpriseNetWithCHF"
        assert isinstance(self.pack_net_plugin.prune_amount, float), \
            "SurpriseNetWithCHF only works with PackNetPlugin with prune_amount as a float"
        self.plugins.append(self.metric)

    def set_chf_params(self, validation_split_proportion: float,
                       lr_grid: t.List[float],
                       accuracy_drop_threshold: float,
                       stability_decay: float):
        """Set the CHF parameters"""
        self.chf_validation_split_proportion = validation_split_proportion
        self.chf_lr_grid = lr_grid
        self.chf_accuracy_drop_threshold = accuracy_drop_threshold
        self.chf_stability_decay = stability_decay

    def search_lr(self,
                  train_exp: NCExperience,
                  valid_exp: NCExperience,
                  **kwargs) -> float:
        """ Maximal plasticity search, tries to find the best learning rate
        for the method without constraints, such as PackNet or EWC. This
        gives CHF a baseline to compare against.
        """
        best_lr = None
        best_acc = 0
        best_model: t.OrderedDict[str, torch.Tensor] = None

        # Disable PackNetPlugin, so that we can train without constraints. And
        # disable metrics so that we don't clutter the tensorboard logs.
        self.pack_net_plugin.enabled = False
        original_model = copy.deepcopy(self.model.state_dict())
        original_clock = copy.deepcopy(self.clock.__dict__)

        # Search for the best learning rate
        print("Starting Maximal Plasticity Search")
        for lr in self.chf_lr_grid:
            self.metric.reset()
            self.model.load_state_dict(copy.deepcopy(original_model))
            self.clock.__dict__ = copy.deepcopy(original_clock)
            self.optimizer.state = {}  # Stops optimizer from remembering old state
            self.optimizer.param_groups[0]['lr'] = lr

            print(f"  Testing lr: {lr}",)
            self._inner_train(train_exp, eval_streams=[valid_exp], **kwargs)
            self.experience = valid_exp

            # Use the validation set. We do not log these results because of
            # bugs in my metric implementations.
            self.evaluator.active = False
            self.eval(valid_exp, **kwargs)
            self.evaluator.active = True

            if self.metric.result() > best_acc:
                best_acc = self.metric.result()
                best_lr = lr
                best_model = copy.deepcopy(self.model.state_dict())

        # Re-enable metrics and PackNetPlugin
        self.pack_net_plugin.enabled = True

        # Restore the best model
        self.model.load_state_dict(copy.deepcopy(best_model))
        print(f"  Learning Rate: {best_lr} Accuracy: {best_acc:0.2f}")
        print()
        return best_lr, best_acc

    def search_tradeoff(self,
                        train_exp: NCExperience,
                        valid_exp: NCExperience,
                        reference_accuracy: float,
                        **kwargs) -> float:
        print(f"Starting Stability Decay Search")
        prune = self.pack_net_plugin.prune_amount

        def in_threshold() -> bool:
            """Returns true if the accuracy is within the margin of the reference accuracy"""
            return self.metric.result() > (1-self.chf_accuracy_drop_threshold)*reference_accuracy

        # Store the original state, so that we can rollback failed searches.
        # This is not a great idea because it will have to grow if new
        # state is added to the system. But it's the easiest way to do it for
        # now.
        self.model.use_top_subset()
        original_model = copy.deepcopy(self.model.state_dict())
        original_clock = copy.deepcopy(self.clock.__dict__)
        original_packnet = copy.deepcopy(self.pack_net_plugin.__dict__)

        while True:
            self.metric.reset()
            self.clock.__dict__ = copy.deepcopy(original_clock)
            self.pack_net_plugin.__dict__ = copy.deepcopy(original_packnet)
            self.model.load_state_dict(copy.deepcopy(original_model))

            # Run SurpriseNet prune and retrain
            self.pack_net_plugin.prune_amount = prune
            self.experience = train_exp
            self.pack_net_plugin.after_training_exp(self, **kwargs)
            self._silent_validation(valid_exp, **kwargs)
            print(f"  Accuracy: {self.metric.result():0.2f}")

            if in_threshold():
                break
            else:
                print(
                    f"  Prune Proportion: {prune:.2f} -> {prune*self.chf_stability_decay:.2f}")
                prune *= self.chf_stability_decay

    def _inner_train(self, experiences, eval_streams=None, **kwargs):
        super().train(experiences, eval_streams, **kwargs)

    def _silent_validation(self, valid_exp, **kwargs):
        # Use the validation set. We do not log these results because some
        # of my metrics do not support validation sets durning training phases.
        self.evaluator.active = False
        self.eval(valid_exp, **kwargs)
        self.evaluator.active = True

    def train(self,
              experiences,
              eval_streams=None,
              **kwargs):
        assert isinstance(experiences, NCExperience), \
            f"CHF only works with NCExperience not {type(experiences)})"

        # Split the experience into a validation set and a training set
        valid_exp, train_exp = _split_experience(
            experiences, self.chf_validation_split_proportion)

        # Use Maximal Plasticity Search to find the best learning rate
        lr, reference_accuracy = self.search_lr(
            train_exp, valid_exp, **kwargs)
        self.optimizer.param_groups[0]['lr'] = lr

        # Use Stability Decay Search to find the best pruning proportion
        self.search_tradeoff(
            train_exp, valid_exp, reference_accuracy, **kwargs)

        print("="*80)
