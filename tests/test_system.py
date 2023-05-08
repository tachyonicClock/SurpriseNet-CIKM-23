
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
