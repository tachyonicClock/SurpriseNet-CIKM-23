from conftest import ROOT_LOG_DIR
from cli import cli
import pytest
import sys


def test_VAE_surprise_net():
    # Expect system exit 0
    with pytest.raises(SystemExit) as e:
        cli(
            [   
                "--log-directory", ROOT_LOG_DIR,
                "--ignore-dirty",
                "--epochs", "2",
                "testExperiment",
                "S-FMNIST",
                "VAE",
                "surprise-net",
                "--retrain-epochs", "1"
            ]
        )
    assert e.value.code == 0

def test_AE_surprise_net():
    # Expect system exit 0
    with pytest.raises(SystemExit) as e:
        cli(
            [   
                "--log-directory", ROOT_LOG_DIR,
                "--ignore-dirty",
                "--epochs", "2",
                "testExperiment",
                "S-FMNIST",
                "AE",
                "surprise-net",
                "--retrain-epochs", "1"
            ]
        )
    assert e.value.code == 0
    

def test_hvae_surprise_net():
    # Expect system exit 0
    with pytest.raises(SystemExit) as e:
        cli(
            [   
                "--log-directory", ROOT_LOG_DIR,
                "--ignore-dirty",
                "--epochs", "2",
                "testExperiment",
                "S-FMNIST",
                "DeepVAE",
                "surprise-net",
                "--retrain-epochs", "1"
            ]
        )
    assert e.value.code == 0
    

