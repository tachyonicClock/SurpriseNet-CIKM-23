from conftest import ROOT_LOG_DIR
from cli import cli

def test_deep_vae():
    cli(
        [
            "--ignore-dirty",
            "--log-mini-batches",
            "--epochs", "1",
            "testExperiment",
            "S-FMNIST",
            "DeepVAE",
            "non-continual"
        ]
    )

def test_DeepVAESurpriseNet():
    cli(
        [   
            "--log-directory", ROOT_LOG_DIR,
            "--ignore-dirty",
            "--log-mini-batches",
            "--epochs", "2",
            "testExperiment",
            "S-FMNIST",
            "DeepVAE",
            "surprise-net",
            "--retrain-epochs", "1"
        ]
    )

