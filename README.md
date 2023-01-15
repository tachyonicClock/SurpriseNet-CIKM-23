# SurpriseNet

The field of continual learning strives to create neural networks that can
accumulate knowledge and skills throughout their lifetime without forgetting.
A key challenge in the field is that many methods cannot learn to differentiate
between classes that weren't presented side-by-side. Methods that can do this
are said to gain *cross-task knowledge*. Replay and generative replay
strategies dominate class-incremental continual learning benchmarks, in part
because they implicitly gain cross-task knowledge. Instead, we use a technique
from anomaly detection to gain cross-task knowledge without replay.

> **SurpriseNet** is named after the parallel between anomaly detection and
> the human concept of surprise. Anomaly detection is the process of
> identifying instances that are different from the rest of the data. In
> the context of continual learning, this is the process of identifying
> instances that belong to different tasks. *Hence, SurpriseNet*.




## Usage

The CLI follows the following basic structure:
```
python cli.py [OPTIONS] LABEL SCENARIO {AE|VAE} STRATEGY
```
- `OPTIONS` - Override default configurations. Set seed etc
- `LABEL` - A short title for the experiment. This helps differentiate experiments
but is only meaningful to you.
- `SCENARIO` - Choose between: `S-FMNIST`, `S-CIFAR10`, `S-CIFAR100`, `S-CORe50`, `SE-CIFAR100`, or `SE-CORe50`.
- `AE|VAE` - Choose between an autoencoder or variational autoencoder
- `STRATEGY` - Choose a continual learning strategy, where `ci-packnet` is our approach


For full details run:
```
python cli.py --help
```

> For simplicity only some configuration options are available through the
> CLI. You may choose to edit `config/config.py` for greater control. And to
> change default values.

All results are saved to tensoboard files in `experiment_logs/`. You can view
the results using tensorboard:
```
tensorboard --logdir experiment_logs/
```
For your convenience, the experiments are labeled thusly:
```
[NUMBER]_[HOSTNAME]_[GIT HASH]_[LABEL]_[SCENARIO]_[AE/VAE]_[STRATEGY]
e.g
0001_ml-14_54dcf601_myExperiment_S-CORe50_AE_taskInference
```

If you forget what the configuration options were for a particular experiment,
the configuration is saved to the tensorboard file and to a JSON file in
`experiment_logs/[EXPERIMENT]/config.json`.

### Examples

Train and test a non-continual model.
```
python cli.py myExperiment S-FMNIST AE non-continual
```
Fine-tuning:
```
python cli.py myExperiment SE-CIFAR100 AE finetuning
```
SurpriseNet with a specific prune proportion:
```
python cli.py myExperiment SE-CIFAR100 VAE ci-packnet -p 0.5
```
Experience replay:
```
python cli.py myExperiment S-CORe50 AE replay -m 1000
```

Equal prune:
```
python cli.py myExperiment S-CIFAR100 AE equal-prune -p 0.5
```

## Areas of interest

- `packnet/packnet.py` contains our implementation of PackNet. We employ the
decorator pattern to implement PackNet. We define multiple decorator classes
to wrap neural network primitive modules with PackNet logic. For example,
the `packnet._PnLinear` class is a decorator for `nn.Linear` that adds the necessary
functionality to make it PackNet compatible. The decorated class is then
used in place of the original object. The pattern allows us to easily convert
any neural network into a PackNet. We simply wrap the network with the
`packnet.wrap`. The method returns a new network with the same architecture
but with each module wrapped with the appropriate PackNet decorator.
- `packnet/task_inference.py` contains the code responsible for task inference,
which is the process of determining which task a instance belongs to. This is done
by activating each task-specific-subset of the network and comparing the
reconstruction errors. The task with the lowest reconstruction error is considered
the task the instance belongs to.

- `train.py` contains the code responsible for creating the neural network and
attaching all the necessary components before training.






