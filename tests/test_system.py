from conftest import ROOT_LOG_DIR
from config.config import ExpConfig
from train import Experiment


def _seed_all():
    import random
    import numpy as np
    import torch

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_short(cfg: ExpConfig):
    cfg.total_task_epochs = 2
    cfg.retrain_epochs = 1
    cfg.tensorboard_dir = ROOT_LOG_DIR
    return Experiment(cfg).train(_early_finish_task_id=2)


def test_fmnist_hvae_surprisenet():
    """Test DeepVAE on FMNIST"""
    _seed_all()
    cfg = ExpConfig()
    cfg.scenario_fmnist()
    cfg.arch_deep_vae()
    cfg.strategy_surprisenet()
    cfg.name = "test_fmnist_hvae_surprisenet"
    _run_short(cfg)


def test_fmnist_hvae_surprisenet_tree_activation():
    _seed_all()
    cfg = ExpConfig()
    cfg.scenario_fmnist()
    cfg.arch_deep_vae()
    cfg.strategy_surprisenet()
    cfg.activation_strategy = "SurpriseNetTreeActivation"
    cfg.name = "test_fmnist_hvae_surprisenet_tree_activation"
    _run_short(cfg)


def test_cifar10_hvae_surprisenet():
    """Test DeepVAE on CIFAR10"""
    _seed_all()
    cfg = ExpConfig()
    cfg.scenario_cifar10()
    cfg.arch_deep_vae()
    cfg.strategy_surprisenet()
    cfg.name = "test_cifar10_hvae_surprisenet"
    _run_short(cfg)
