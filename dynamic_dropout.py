from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.strategies import Naive
from helper import get_eval_plugin, training_loop, tb_logger
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
from network.simple_network import SimpleDropoutMLP

from network.module.dropout import ConditionedDropout, NaiveDropout

scenario = SplitFMNIST(n_experiences=5)

cond_dropout = ConditionedDropout(512, 5, 0.1, 0.8)
naive_dropout = NaiveDropout()
model = SimpleDropoutMLP(num_classes=scenario.n_classes, hidden_layers=2, dropout_module=naive_dropout)

eval_plugin = get_eval_plugin([tb_logger("naive_dropout"), TextLogger(open('log.txt', 'a')), InteractiveLogger()], scenario.n_classes)

cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin, device="cuda")


# TRAINING LOOP
print('Starting experiment...')
results = []
for i, experience in enumerate(scenario.train_stream):
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    cond_dropout.set_active_group(i)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream[:i+1]))