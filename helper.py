from datetime import  datetime
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy


def tb_logger(label, path="./tb_data"):
    """"""
    return TensorboardLogger(f"{path}/{label}_{datetime.now()}")

def get_eval_plugin(loggers, n_classes):
    """Get the eval plugin"""
    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True, trained_experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=n_classes, save_image=True, normalize="true",
                                stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers
    )
