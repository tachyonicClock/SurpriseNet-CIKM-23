

from avalanche.evaluation.metrics.loss import LossPluginMetric

class TrainExperienceLoss(LossPluginMetric):

    def __init__(self):
        super(TrainExperienceLoss, self).__init__(
            reset_at='experience', emit_at='iteration', mode='train')

    def __str__(self):
        return "training_loss"