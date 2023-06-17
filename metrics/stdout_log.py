from avalanche.core import SupervisedPlugin
from avalanche.logging import BaseLogger
from avalanche.training.templates import SupervisedTemplate
import click
from tqdm import tqdm
import os
import numpy as np


class StdoutLog(SupervisedPlugin, BaseLogger):
    def __init__(self):
        self.pbar_mb = None
        self.interactive = os.isatty(1)

        self.after_training_iteration = self.update_pbar
        self.after_training_epoch = self.close_pbar
        self.after_eval_iteration = self.update_pbar
        self.after_eval_exp = self.close_pbar

    def echo_header(self, s: str):
        click.secho(s, fg="blue", bold=True)

    def before_training(self, strategy: SupervisedTemplate, *args, **kwargs):
        if not self.interactive:
            click.secho("Not an interactive session. Using append mode.", fg="yellow")

    def before_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        i_exp = strategy.experience.current_experience
        classes = np.unique(strategy.experience.classes_in_this_experience)
        self.echo_header("-" * 80)
        self.echo_header(f"Experience {i_exp} containing the classes: {classes}")

    def before_training_epoch(self, strategy: SupervisedTemplate, *args, **kwargs):
        if self.interactive:
            self.pbar_mb = tqdm(
                total=len(strategy.dataloader),
                ncols=80,
                leave=True,
                desc=f"Epoch {strategy.clock.train_exp_epochs: 2d}",
                colour="CYAN",
            )

    def before_eval(self, *args, **kwargs):
        self.eval_counter = 0

    def before_eval_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        if self.interactive:
            self.eval_counter += 1
            self.pbar_mb = tqdm(
                total=len(strategy.dataloader),
                ncols=80,
                leave=True,
                desc=f"Eval {self.eval_counter: 2d}",
                colour="YELLOW",
            )

    def update_pbar(self, strategy: SupervisedTemplate, *args, **kwargs):
        if self.pbar_mb is not None:
            self.pbar_mb.update(1)

    def close_pbar(self, *args, **kwargs):
        if self.pbar_mb is not None:
            self.pbar_mb.close()
            self.pbar_mb = None
