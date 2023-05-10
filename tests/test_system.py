
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


def test_DeepVAE_packnet():
    # python cli.py --ignore-dirty --log-mini-batches  --epochs 2  myExperiment S-FMNIST DeepVAE surprise-net --retrain-epochs 1  
    cli(
        [
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

