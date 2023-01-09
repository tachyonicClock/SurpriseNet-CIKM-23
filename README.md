# SurpriseNet

SurpriseNet is a class incremental continual learning technique. It allows
a neural network to learn from a stream or sequence of classes rather than a
traditional static dataset. The main challenge it solves is differentiating classes
that were never presented side-by-side without the need for replay.



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

- `packnet/packnet.py` contains our implementation of PackNet. I implemented
PackNet using a decorator design pattern. It is implemented by creating a
wrapper class that contains the original object and the added PackNet 
functionality, and the wrapper class is then used in place of the original 
object. The pattern allows us to easily convert a neural network into a PackNet,
simply by decorating it with the PackNet version of the module.

- `packnet/task_inference.py` contains the code responsible for task inference,
which is the process of determining which task a instance belongs to. This is done
by activating each task-specific-subset of the network and comparing the
reconstruction errors. The task with the lowest reconstruction error is the
task the instance belongs to.

- `train.py` contains the code responsible for creating the neural network and 
attaching all the necessary components before training.






