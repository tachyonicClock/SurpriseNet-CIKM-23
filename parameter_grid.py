import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import nn

from experiment import ExperimentConfig, FashionExperiment, set_all_seeds
from network.simple_network import SimpleDropoutMLP
from avalanche.models.simple_mlp import SimpleMLP

def time_str() -> str:
    return datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
def mp_seed():
    return mp.current_process().pid * int(time.time()) % 123456789 

def vanilla_dropout(args):
    p = args
    set_all_seeds(mp_seed())

    experiment_builder = FashionExperiment("vanilla_dropout", ExperimentConfig(
        lr=0.001,
        momentum=0.8,
        train_mb_size=64,
        eval_mb_size=100
    ))

    print(f"Running p={p}")

    def make_model(n_classes):
        return SimpleMLP(
            num_classes=n_classes,
            hidden_size=512, hidden_layers=3,
            drop_rate=p)

    # def make_lr_schedule(optimizer):
    #     return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    experiment_builder.make_model = make_model
    results = experiment_builder.build().train()

    print(f"Done:{results[-1]['Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000']}")

    return {**results[-1], "P(drop)":p }

def main():
    df = pd.DataFrame()
    # p_actives = np.arange(0.1, 0.9, 0.1)
    # p_inactives = np.arange(0.1, 0.9, 0.1)
    p_drop = np.arange(0.0001, 0.999, 0.02)

    with mp.Pool(10) as p:
        items = [(a) for a in p_drop]
        print(f"Starting {len(items)} experiments")
        rows = p.map(vanilla_dropout, items)

        df = df.append(rows)

    df.to_csv(f"vanilla_grid_{time_str()}.csv")
    print(df)


main()
