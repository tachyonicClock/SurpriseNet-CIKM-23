# SurpriseNet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8247906.svg)](https://doi.org/10.5281/zenodo.8247906)

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

## Othermethods 
Most other method were implemented using the [mammoth](https://github.com/aimagelab/mammoth) codebase.
S-FMNIST, S-DSADS and S-PAMAP2 are added by us to mammoth codebase. Furthermore, we added class order 
shuffling. For all methods we performed a grid search over the hyperparameters.

BIR, and GR were implemented using [brain-inspired replay](https://github.com/GMvandeVen/brain-inspired-replay)
codebase. We added class order shuffling and S-DSADS, S-PAMAP2, S-FMNIST datasets. For their methods, we used the 
default hyperparameters. For S-DSADS and S-PAMAP2, we performed a grid search over the hyperparameters.

Finally CLOM was implemented using [CLOM](https://github.com/k-gyuhak/clom) codebase. 
We added class order shuffling and S-FMNIST datasets. However, their method does not support
S-DSADS and S-PAMAP2 datasets. For their methods, we used the default hyperparameters but had
to adjust the epoch budgets.

The results of our grid search can be found in `SurpriseNet Hyperprarameters.ods` or `SurpriseNet Hyperprarameters.xlsx`.
## Usage

The CLI follows the following basic structure:
```sh
python cli.py [OPTIONS] LABEL SCENARIO {AE|VAE} STRATEGY
```
- `OPTIONS` - Override default configurations. Set seed etc
- `LABEL` - A short title for the experiment. This helps differentiate experiments
but is only meaningful to you.
- `SCENARIO` - Choose between: `S-DSADS`, `S-PAMAP2`, `S-FMNIST`, `S-CIFAR10`, `S-CIFAR100`
   to enable SurpriseNetE you can use `SE-FMNIST`, `SE-CIFAR10`, `SE-CIFAR100`, attaching
   a feature extractor before an MLP SurpriseNet.
- `AE|VAE` - Choose between an autoencoder or variational autoencoder
- `STRATEGY` - Choose a continual learning strategy, where `surprise-net` is our approach

Ensure you setup the `DATASETS` environment variable:
```sh
export DATASETS=/path/to/datasets
```
Make sure you extract the S-DSADS and S-PAMAP2 datasets from the zip file. 
```
 unzip HAR.zip -d $DATASETS
```


For full details run:
```sh
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
SurpriseNet with a specific prune proportion:
```
python cli.py myExperiment SE-CIFAR100 VAE surprise-net -p 0.5
```
Equal prune:
```
python cli.py myExperiment S-CIFAR100 AE surprise-net --equal-prune
```
