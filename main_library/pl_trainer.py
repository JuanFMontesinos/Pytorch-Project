import os
from typing import List

# ML libraries
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

# Data science libraries
import matplotlib.pyplot as plt
import numpy as np


class Trainer(LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(hparams)
        self.model = AVCoreModel.from_argparse_args(self.hparams)

