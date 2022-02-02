import copy
import torch
from torch import Tensor
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from network.nn_traits import HasFeatureMap


class BackboneLWF(StrategyPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.teacher: HasFeatureMap = None

    def before_training_epoch(self, strategy, **kwargs):
        assert isinstance(strategy.model, HasFeatureMap)
        self.model: HasFeatureMap = strategy.model

    def _distillation_loss(student_out: Tensor, teacher_out: Tensor, temp: float, alpha: float):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        log_p = torch.log_softmax(student_out / temp, dim=1)
        q = torch.softmax(teacher_out / temp, dim=1)
        loss = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return loss * alpha

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        # No knowledge distillation is done
        if self.teacher == None:
            return

        mb_x = strategy.mb_x

        alpha = self.alpha[strategy.clock.train_exp_counter] \
            if isinstance(self.alpha, (list, tuple)) else self.alpha

        # Teacher gradients are irrelevant and should not be calculated
        with torch.no_grad():
            teacher_out = self.teacher.forward_to_featuremap(mb_x)
        student_out = self.model.forward_to_featuremap(mb_x)

        penalty = BackboneLWF._distillation_loss(
            student_out, teacher_out, self.temperature, alpha)
        strategy.loss += penalty

    def after_training_exp(self, strategy, **kwargs):
        self.teacher: HasFeatureMap = copy.deepcopy(strategy.model)
