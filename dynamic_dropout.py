from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.scenarios.generic_definitions import ScenarioStream
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.strategies.base_strategy import BaseStrategy
from helper import get_eval_plugin, training_loop
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

tblog = TensorboardLogger()
eval_plugin = get_eval_plugin([InteractiveLogger(), TextLogger(open('log.txt', 'a')), tblog])

cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)


training_loop(cl_strategy, scenario)
