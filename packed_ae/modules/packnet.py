import torch
import torch.nn.functional as F
from torchvision.models.mobilenet import mobilenet_v3_small
import pytorch_lightning as pl
import torchmetrics as tm
import torch.nn as nn

from modules.traits import HasTaskID


class PackNet(pl.LightningModule, HasTaskID):
    def __init__(self):
        super().__init__()
        self.network = mobilenet_v3_small(True)

        self.accuracy = tm.Accuracy()
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out_layer = self(x) # Output layer or logits
        loss = self.loss_func(out_layer, y)

        # Log
        self.log(f"train_loss/{self.task_tag}", loss, on_step=True)
        return loss

    def _validate(self, metric_prefix, batch, batch_idx):
        x, y = batch
        out_layer = self(x)
        loss = self.loss_func(out_layer, y)
        y_hat = torch.argmax(out_layer, dim=1) # 

        self.accuracy(y_hat, y)

        # Log
        self.log(f"{metric_prefix}_loss/{self.task_tag}", loss)
        self.log(f"{metric_prefix}_acc/{self.task_tag}", self.accuracy)
        self.log(f"val_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._validate("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._validate("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}