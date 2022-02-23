from typing import Dict
import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import MetricValue

from avalanche.core import SupervisedPlugin
from network.trait import HasFeatureMap
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
import io


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class FeatureMap(PluginMetric):

    sample: int = 0
    feature_map_sums: Dict[int, torch.Tensor] = {}
    to_img = T.ToPILImage()

    @torch.no_grad()
    def after_eval_forward(self, strategy: SupervisedPlugin):

        # No model does not have feature map
        if not isinstance(strategy.model, HasFeatureMap):
            return None

        for f_map, y_class in zip(
                strategy.model.forward_to_featuremap(strategy.mb_x),
                strategy.mb_y):
            y_class = int(y_class)
            if y_class not in self.feature_map_sums:
                self.feature_map_sums[y_class] = torch.zeros(f_map.shape)
            self.feature_map_sums[y_class] += f_map.to("cpu")

    def after_eval(self, strategy):
        metric_name = f"feature_map_{strategy.clock.train_exp_counter}"
        metrics = []

        for y, f_map in self.feature_map_sums.items():
            fig, ax = plt.subplots()
            ax.imshow(f_map.view(8, 8))
            metrics.append(MetricValue(self, metric_name,
                           fig2img(plt), x_plot=float(y)))
        return sorted(metrics, key=lambda i: i.x_plot)

    def reset(self, **kwargs) -> None:
        self.feature_map_sums = {}

    def result(self, **kwargs):
        pass
